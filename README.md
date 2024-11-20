# GNN Pipeline

This repository contains a pipeline for training a Graph Neural Network (GNN) on a synthetic social network dataset.

## Files
- `data_preparation.py`: Script to generate and encode the synthetic dataset.
- `train_gnn.py`: Script to train a 2-layer GNN for node classification.
- `Makefile`: Automates the pipeline steps.

## How to Use
1. Clone this repository.
2. Ensure you have Python 3.8+ and install dependencies:
   ```
   pip install torch torch-geometric pandas scikit-learn
   ```
3. Run the pipeline:
   ```
   make
   ```

## Output
The pipeline generates:
- Trained GNN model.
- Training loss printed at each epoch.

## Extend
You can extend this pipeline by adding:
- Advanced GNN architectures (e.g., GAT, GraphSAGE).
- Variational Autoencoders or GANs for graph tasks.

---

Author: [Your Name]