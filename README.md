# GNN_MODEL_TRPM8_DRUG_POTENCY_PREDICTION

Predict TRPM8 ligand potency (classification & IC₅₀ regression) using Graph Neural Networks.

---

## 🚀 Project Overview
The goal of this project is to predict the potency of small-molecule ligands against the TRPM8 ion channel from ChEMBL, by learning directly on their molecular graph representations. We tackle two complementary tasks:

1. **Potency classification** — Low / Medium / High  
2. **IC₅₀ regression** — Continuous pChEMBL values

We compare five GNN architectures (GIN, GCN, MPNN, GraphSAGE, GAT) through:

- **Data prep**: clean raw measurements, compute atom/bond descriptors  
- **Graph construction**: build PyG `Data` objects  
- **Splitting**: stratified (classification) or random (regression) 10-fold CV + held-out test  
- **Training & tuning**: per-fold training, hyperparameter sweeps (hidden_dim, dropout, lr), early stopping, LR schedulers  
- **Ensembling & final eval**: average fold predictions, retrain final model, plot confusion matrices/AUC (classification) and scatter/MSE/R² (regression)  
- **Baseline comparison**: classical QSAR vs. GNN (bar & radar plots)

---
## 🧠 GNN Architectures

- **GIN (Graph Isomorphism Network)**  
 uses MLPs to distinguish different graph structures; strong at capturing subtle subgraph differences.

- **GCN (Graph Convolutional Network)**  
  Classic spectral convolution: aggregates neighbor features via a normalized adjacency; simple and widely used.

- **MPNN (Message Passing Neural Network)**  
  Flexible message‐passing framework: learns custom messages along edges, then updates node embeddings—great for chemistry.

- **GraphSAGE**  
  samples and aggregates a fixed-size neighborhood; efficient for large graphs .

- **GAT (Graph Attention Network)**  
  Attention‐based aggregation: learns per-edge attention coefficients, allowing the model to weigh important neighbors more heavily.

---

## 📂 Repository Structure

```text
GNN_MODEL_TRPM8_DRUG_POTENCY_PREDICTION/
├── 1_data/
│   ├── initial_data/
│   │   ├── orca_outputs/
│   │   └── Pre-process.ipynb
│   └── processed/
│       ├── .gitkeep
│       ├── TRPM8_cleaned_preprocessed.csv
│       ├── TRPM8_graph_classification.csv
│       └── TRPM8_graph_regression.csv
├── 2_feature_extraction/
│   ├── .gitkeep
│   ├── Feature_extraction.ipynb
│   ├── mulliken_charges.json
│   ├── TRPM8_graph_features_class_w_atomic_onehot_encoder.csv
│   └── TRPM8_graph_features_regression_w_atomic_onehot_encoder.csv
├── 3_graph_data/
│   ├── .gitkeep
│   ├── GraphDataset_conversion_for_PyTorch_Geometric.ipynb
│   ├── TRPM8_classification_graph_dataset.pt
│   └── TRPM8_regression_graph_dataset.pt
├── 4_train_test_split/
│   ├── 10fold_cv/
│   │   ├── regression/
│   │   ├── classification/
│   ├── .gitkeep
│   └── Graph_dataset_train_test_val_random_split_and_Kfoldsplit.ipynb
├── 5_model_training/
│   ├── GAT/
│   │   ├── classification_10fold/
│   │   ├── regression_10fold/
│   │   ├── Final_GAT_training_classification.ipynb
│   │   └── Final_GAT_training_regression.ipynb
│   ├── GCN/
│   │   ├── classification_10fold/
│   │   ├── regression_10fold/
│   │   ├── Final_GCN_training_classification.ipynb
│   │   └── Final_GCN_training_regression.ipynb
│   ├── GIN/
│   │   ├── classification_10fold/
│   │   ├── regression_10fold/
│   │   ├── Final_GIN_training_classification.ipynb
│   │   └── Final_GIN_training_regression.ipynb
│   ├── GraphSAGE/
│   │   ├── classification_10fold/
│   │   ├── regression_10fold/
│   │   ├── Final_GraphSAGE_training_classification.ipynb
│   │   └── Final_GraphSAGE_training_regression.ipynb
│   └── MPNN/
│       ├── classification_10fold/
│       ├── regression_10fold/
│       ├── Final_MPNN_training_classification.ipynb
│       └── Final_MPNN_training_regression.ipynb
├── 6_baseline/
│   ├── QSAR_classification_performance_summary.csv
│   └── QSAR_regression_performance_summary.csv
│   └── GNN_vs_QSAR_comparison.ipynb
└── README.md           
