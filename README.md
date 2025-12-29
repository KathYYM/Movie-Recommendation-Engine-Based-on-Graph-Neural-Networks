# Graph Neural Network Recommendation System

This project investigates the behavior, performance, and **numerical stability** of Graph Neural Networks (GNNs) for large-scale recommendation systems.  
The work was developed as part of **CS7643: Deep Learning (Georgia Tech OMSCS)**.

---

## Overview

- Benchmarked **GAT, LightGCN, GraphSAGE, and GCN** on the MovieLens 1M dataset
- Implemented models using **PyTorch Geometric**
- Emphasized numerical stability, evaluation rigor, and reproducibility

---

## Key Insight: Numerical Instability in GATs

During experimentation with **Graph Attention Networks (GAT)**, I observed severe regression instability, where predicted ratings drifted far beyond the valid range.

### Diagnosis
- Unconstrained attention-weighted aggregation
- Gradient amplification during message passing

### Solution
I introduced a **sigmoid-constrained output layer** to naturally bound predictions during training, which:
- Stabilized gradient flow
- Improved convergence
- Significantly increased downstream metrics

**Peak Recall@k: 0.968**

---

## Results

<p float="left">
  <img src="results/figures/Confusion_Matrix.png" width="420"/>
  <img src="results/figures/Distribution_of_Predicted_Ratings.png" width="420"/>
</p>

---

## Methods & Models

The following architectures were implemented and evaluated using **PyTorch Geometric**:

- **GCN (Graph Convolutional Network)**
- **GraphSAGE**
- **LightGCN**
- **Graph Attention Network (GAT)**

**Architectural focus:**
- Multi-head attention for heterogeneous neighborhood aggregation
- Embedding dimensionality analysis (64 → 512)
- Over-smoothing diagnostics

---

## Repository Structure

```text
CS7643_Project/
├── src/
│   ├── GATs_best_model.py      # Final stabilized GAT implementation
│   ├── gnn_test.py             # Evaluation and testing scripts
│   └── utils/                  # Helper functions
├── results/
│   └── figures/
│       ├── Confusion_Matrix.png
│       └── Distribution_of_Predicted_Ratings.png
├── README.md
├── LICENSE
└── .gitignore
```