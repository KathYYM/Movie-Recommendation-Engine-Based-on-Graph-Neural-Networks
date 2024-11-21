# Databricks notebook source
import os
import torch
from tqdm import tqdm
import typing_extensions
from importlib import reload
reload(typing_extensions)
# %pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
os.environ['TORCH'] = torch.__version__
# %pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}.html
# %pip install torch_geometric

# COMMAND ----------
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric.data import HeteroData
from torch_geometric.datasets import MovieLens
from torch.nn import Linear
from torch.optim.lr_scheduler import StepLR
from torch_geometric.datasets import MovieLens1M
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch_geometric.transforms import RandomLinkSplit
import copy
from sklearn.model_selection import ParameterGrid

import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

def create_data_splits(data, val_ratio=0.1, test_ratio=0.2, seed=42):
    """
    Create train, validation, and test splits for the MovieLens HeteroData.
    
    Args:
        data: PyG HeteroData object with the structure:
              - movie.x: [num_movies, num_movie_features]
              - user.x: [num_users, num_user_features]
              - (user, rates, movie): edge_index, rating, time, edge_label_index, edge_label
              - (movie, rated_by, user): edge_index, rating, time
        val_ratio: Ratio of validation edges (default: 0.1)
        test_ratio: Ratio of test edges (default: 0.2)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (train_data, val_data, test_data) HeteroData objects
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get edge indices and attributes
    edge_index = data['user', 'rates', 'movie'].edge_index
    ratings = data['user', 'rates', 'movie'].rating
    
    # Create a permutation for random splitting
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)
    
    # Calculate split sizes
    test_size = int(num_edges * test_ratio)
    val_size = int(num_edges * val_ratio)
    train_size = num_edges - (test_size + val_size)
    
    # Split edge indices
    train_indices = perm[:train_size]
    val_indices = perm[train_size:train_size + val_size]
    test_indices = perm[train_size + val_size:]
    
    # Function to create a split
    def create_split(indices):
        split_data = HeteroData()
        
        # Add node features
        split_data['movie'].x = data['movie'].x
        split_data['user'].x = data['user'].x
        
        # Add forward edges
        split_data['user', 'rates', 'movie'].edge_index = edge_index[:, indices]
        split_data['user', 'rates', 'movie'].rating = ratings[indices]
        
        # Add reverse edges
        rev_edge_index = torch.stack([edge_index[1, indices], edge_index[0, indices]])
        split_data['movie', 'rated_by', 'user'].edge_index = rev_edge_index
        split_data['movie', 'rated_by', 'user'].rating = ratings[indices]
        
        # Add edge labels
        split_data['user', 'rates', 'movie'].edge_label = (split_data['user', 'rates', 'movie'].rating >= 4).float()
        
        return split_data
    
    # Create splits
    train_data = create_split(train_indices)
    val_data = create_split(val_indices)
    test_data = create_split(test_indices)
    
    # Print statistics
    print("\nData split statistics:")
    print("Original data:")
    print(f"Number of users: {data['user'].x.size(0)}")
    print(f"Number of movies: {data['movie'].x.size(0)}")
    print(f"Number of ratings: {num_edges}")
    
    print("\nSplit sizes:")
    print(f"Train edges: {train_data['user', 'rates', 'movie'].edge_index.size(1)}")
    print(f"Val edges: {val_data['user', 'rates', 'movie'].edge_index.size(1)}")
    print(f"Test edges: {test_data['user', 'rates', 'movie'].edge_index.size(1)}")
    
    # Verify no overlapping edges
    def get_edge_set(split_data):
        edges = split_data['user', 'rates', 'movie'].edge_index.t()
        return {tuple(edge.tolist()) for edge in edges}
    
    train_edges = get_edge_set(train_data)
    val_edges = get_edge_set(val_data)
    test_edges = get_edge_set(test_data)
    
    print("\nOverlap verification:")
    print(f"Train-Val overlap: {len(train_edges & val_edges)}")
    print(f"Train-Test overlap: {len(train_edges & test_edges)}")
    print(f"Val-Test overlap: {len(val_edges & test_edges)}")
    
    # Verify rating distributions
    print("\nRating distributions:")
    for name, split in zip(['Train', 'Val', 'Test'], 
                         [train_data, val_data, test_data]):
        ratings = split['user', 'rates', 'movie'].rating
        unique_ratings, counts = torch.unique(ratings, return_counts=True)
        dist = counts.float() / len(ratings)
        print(f"\n{name} split:")
        for r, p in zip(unique_ratings, dist):
            print(f"Rating {r.item()}: {p.item():.3f}")

    print("\nLabel distributions:")
    for name, split in zip(['Train', 'Val', 'Test'], 
                         [train_data, val_data, test_data]):
        labels = split['user', 'rates', 'movie'].edge_label
        unique_labels, counts = torch.unique(labels, return_counts=True)
        dist = counts.float() / len(labels)
        print(f"\n{name} split:")
        for r, p in zip(unique_labels, dist):
            print(f"Rating {r.item()}: {p.item():.3f}")
            
    # Additional verification
    def verify_split(name, split_data):
        forward_edges = split_data['user', 'rates', 'movie'].edge_index
        reverse_edges = split_data['movie', 'rated_by', 'user'].edge_index
        
        # Check if reverse edges match forward edges
        forward_set = {tuple(edge.tolist()) for edge in forward_edges.t()}
        reverse_set = {tuple(reversed(edge.tolist())) for edge in reverse_edges.t()}
        
        print(f"\n{name} Split Verification:")
        print(f"Forward edges: {len(forward_set)}")
        print(f"Reverse edges: {len(reverse_set)}")
        print(f"Forward-Reverse match: {forward_set == reverse_set}")
        
        # Verify ratings consistency
        forward_ratings = split_data['user', 'rates', 'movie'].rating
        reverse_ratings = split_data['movie', 'rated_by', 'user'].rating
        ratings_match = torch.all(forward_ratings == reverse_ratings)
        print(f"Ratings consistency: {ratings_match}")
    
    # Verify each split
    verify_split("Train", train_data)
    verify_split("Val", val_data)
    verify_split("Test", test_data)
    
    return train_data, val_data, test_data


class MovieGNN(torch.nn.Module):
    def __init__(self, num_layers, gat_heads, hidden_channels, hetero_aggr, 
                 relu_slope, dropout_rate, sequential_channels):
        super().__init__()
        
        # Graph convolution layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ('user', 'rates', 'movie'): GATConv(
                        in_channels=(-1, -1),
                        out_channels=hidden_channels,
                        heads=gat_heads,
                        add_self_loops=False  # Explicitly ensure self-loops are not added
                    ),
                    ('movie', 'rated_by', 'user'): GATConv(
                        in_channels=(-1, -1),
                        out_channels=hidden_channels,
                        heads=gat_heads,
                        add_self_loops=False  # Explicitly ensure self-loops are not added
                    ),
                },
                aggr=hetero_aggr  # Aggregation across relations
            )
            self.convs.append(conv)
            
        # MLP for final prediction
        self.relu = torch.nn.LeakyReLU(negative_slope=relu_slope)
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        # Prediction layers
        self.sequential = torch.nn.Sequential(
            Linear(hidden_channels * gat_heads * 2, sequential_channels),
            torch.nn.BatchNorm1d(sequential_channels),
            torch.nn.LeakyReLU(negative_slope=relu_slope),
            torch.nn.Dropout(dropout_rate)
        )
        self.out = Linear(sequential_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_dict, edge_index_dict):
        # Message passing layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.relu(x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        return x_dict

    def decode(self, z_dict, edge_index, apply_sigmoid=False):
        # Combine user and movie embeddings
        movie_embeddings = z_dict['movie'][edge_index[1]]
        user_embeddings = z_dict['user'][edge_index[0]]
        
        # Concatenate and predict
        concat_embeddings = torch.cat([user_embeddings, movie_embeddings], dim=-1)
        hidden = self.sequential(concat_embeddings)
        pred = self.out(hidden).squeeze()

        # Apply sigmoid if requested
        if apply_sigmoid:
            pred = self.sigmoid(pred)
        return pred

def train_model(model, train_data, val_data, params, apply_sigmoid=False):
    """
    Train the GNN model with validation monitoring. Skips operations that cause CUDA OOM errors.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=params['lr'], 
        weight_decay=params['weight_decay']
    )
    scheduler = StepLR(
        optimizer, 
        step_size=params['scheduler_step'], 
        gamma=params['scheduler_gamma']
    )
    criterion = torch.nn.BCEWithLogitsLoss() if not apply_sigmoid else torch.nn.BCELoss()

    best_val_loss = float('inf')
    best_model = None
    patience = params['patience']
    counter = 0
    train_losses, val_losses = [], []

    for epoch in tqdm(range(params['num_epochs'])):
        # Training Phase
        model.train()
        try:
            optimizer.zero_grad()
            z_dict = model(train_data.x_dict, train_data.edge_index_dict)
            edge_index = train_data['user', 'rates', 'movie'].edge_index
            train_pred = model.decode(z_dict, edge_index, apply_sigmoid=apply_sigmoid)
            train_loss = criterion(train_pred, train_data['user', 'rates', 'movie'].edge_label)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(train_loss))
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"Skipping batch due to CUDA out-of-memory at epoch {epoch}.")
                torch.cuda.empty_cache()
            else:
                raise e

        # Validation Phase
        model.eval()
        with torch.no_grad():
            try:
                z_dict = model(val_data.x_dict, val_data.edge_index_dict)
                edge_index = val_data['user', 'rates', 'movie'].edge_index
                val_pred = model.decode(z_dict, edge_index, apply_sigmoid=apply_sigmoid)
                val_loss = criterion(val_pred, val_data['user', 'rates', 'movie'].edge_label)
                val_losses.append(float(val_loss))
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"Skipping validation due to CUDA out-of-memory at epoch {epoch}.")
                    torch.cuda.empty_cache()
                else:
                    raise e

        # Early Stopping Logic
        if val_losses and val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_model = copy.deepcopy(model)
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        if (epoch + 1) % params['eval_every'] == 0:
            if train_losses and val_losses:  # Check if losses are not empty
                print(f"Epoch {epoch + 1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")
            else:
                print(f"Epoch {epoch + 1}: No valid batches processed due to memory issues.")

            
    return best_model, train_losses, val_losses, epoch



def evaluate_model(model, data, split_name="Test"):
    """
    Evaluate the model's performance on a specific data split.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    all_probs, all_labels = [], []

    model.eval()
    with torch.no_grad():
        try:
            z_dict = model(data.x_dict, data.edge_index_dict)
            edge_index = data['user', 'rates', 'movie'].edge_index
            probs = model.decode(z_dict, edge_index, apply_sigmoid=True)
            labels = data['user', 'rates', 'movie'].edge_label

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"Skipping evaluation batch due to CUDA out-of-memory on {split_name} data.")
                torch.cuda.empty_cache()
            else:
                raise e

    # Combine all results
    if all_probs and all_labels:
        probs = torch.cat(all_probs)
        labels = torch.cat(all_labels)

        # Convert to numpy for metric computation
        probs_np = probs.numpy()
        labels_np = labels.numpy()
        preds_np = (probs_np >= 0.5).astype(int)  # Binary predictions

        # Metrics
        accuracy = accuracy_score(labels_np, preds_np)
        precision = precision_score(labels_np, preds_np)
        recall = recall_score(labels_np, preds_np)
        f1 = f1_score(labels_np, preds_np)

        # Confusion matrix components
        tp = np.sum((preds_np == 1) & (labels_np == 1))
        tn = np.sum((preds_np == 0) & (labels_np == 0))
        fp = np.sum((preds_np == 1) & (labels_np == 0))
        fn = np.sum((preds_np == 0) & (labels_np == 1))

        # Compile results
        results = {
            "prediction": preds_np.tolist(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

        return results
    else:
        # Return None results if there were no probabilities or labels
        return {
            "prediction": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "true_positives": None,
            "true_negatives": None,
            "false_positives": None,
            "false_negatives": None,
        }


def main():
    # Load MovieLens dataset
    dataset = MovieLens1M('../data/MovieLens1M')
    data = dataset[0].cuda()
    
    # Create data splits
    train_data, val_data, test_data = create_data_splits(
        data,
        val_ratio=0.1,
        test_ratio=0.2,
        seed=42
    )
    
    # Initialize results storage
    results = []
    # Model parameters
    param_grid = {
        'num_layers': [2, 3],
        'gat_heads': [2, 4, 8],
        'hidden_channels': [64, 128],
        'hetero_aggr': ['mean', 'sum', 'max'],
        'relu_slope': [0.1, 0.2],
        'dropout_rate': [0, 0.1],
        'sequential_channels': [32],
        'lr': [0.01, 0.001],
        'weight_decay': [1e-5],
        'scheduler_step': [10],
        'scheduler_gamma': [0.5],
        'num_epochs': [100, 200],
        'eval_every': [10],
        'patience': [10]  # Early stopping patience
    }
    for params in ParameterGrid(param_grid):
        print(f"\nTrying parameters: {params}")
        # Create and train model
        model = MovieGNN(
            num_layers=params['num_layers'],
            gat_heads=params['gat_heads'],
            hidden_channels=params['hidden_channels'],
            hetero_aggr=params['hetero_aggr'],
            relu_slope=params['relu_slope'],
            dropout_rate=params['dropout_rate'],
            sequential_channels=params['sequential_channels'],
        ).cuda()
        
        # Train model
        model, train_losses, val_losses, best_epoch = train_model(
            model, train_data, val_data, params, apply_sigmoid=False
        )

        results.append({
                'params': params,
                'best_val_loss': val_losses,
                'best_epoch': best_epoch,
            })
        
    # Convert results to DataFrame and sort by validation loss
    train_results_df = pd.DataFrame([
        {
        'num_layers': r['params']['num_layers'],
        'gat_heads': r['params']['gat_heads'],
        'hidden_channels': r['params']['hidden_channels'],
        'hetero_aggr': r['params']['hetero_aggr'],
        'relu_slope': r['params']['relu_slope'],
        'dropout_rate': r['params']['dropout_rate'],
        'sequential_channels': r['params']['sequential_channels'],
        'lr': r['params']['lr'],
        'weight_decay': r['params']['weight_decay'],
        'scheduler_step': r['params']['scheduler_step'],
        'scheduler_gamma': r['params']['scheduler_gamma'],
        'num_epochs': r['params']['num_epochs'],
        'eval_every': r['params']['eval_every'],
        'patience': r['params']['patience'],
        'best_val_loss': r['best_val_loss'],
        'best_epoch': r['best_epoch'],
        }
        for r in results
    ])

    train_results_df = train_results_df.sort_values('best_val_loss')

    # Print best parameters
    print("\nBest parameters:")
    best_params = train_results_df.iloc[0]
    print(best_params)
    
    # Evaluate
    model = MovieGNN(
            num_layers=best_params['num_layers'],
            gat_heads=best_params['gat_heads'],
            hidden_channels=best_params['hidden_channels'],
            hetero_aggr=best_params['hetero_aggr'],
            relu_slope=best_params['relu_slope'],
            dropout_rate=best_params['dropout_rate'],
            sequential_channels=best_params['sequential_channels'],
        ).cuda()
        
    # Train model
    best_model, train_losses, val_losses, best_epoch = train_model(
        model, train_data, val_data, best_params, apply_sigmoid=False
    )
    test_results = evaluate_model(best_model, test_data, "Test")

    return train_results_df, test_results


# COMMAND ----------

train_results_df, test_results = main()

# COMMAND ----------

print(test_results)

# COMMAND ----------



# Extract confusion matrix values
tn, fp, fn, tp = test_results['true_negatives'], test_results['false_positives'], test_results['false_negatives'], test_results['true_positives']

# Create confusion matrix
conf_matrix = confusion_matrix(
    [1] * tp + [0] * tn + [1] * fn + [0] * fp, 
    [1] * tp + [0] * tn + [0] * fn + [1] * fp
)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

plt.hist(test_results['prediction'], bins=50, alpha=0.75)
plt.xlabel('Predicted Ratings')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Ratings')
plt.show()