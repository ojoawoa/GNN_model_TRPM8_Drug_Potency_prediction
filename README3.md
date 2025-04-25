```markdown
# Repository Structure


```
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
│   |   ├── classification
│   |   ├── regression
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

```
```