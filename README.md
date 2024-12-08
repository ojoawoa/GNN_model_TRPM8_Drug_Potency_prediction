# Project Overview

The goal of this project is to predict the potency of molecules on the TRPM8 drug target using their physical, chemical, and topological properties. The project addresses two main predictive tasks: potency classification (e.g., high, medium or low potency) and IC50 value prediction (regression). The workflow is divided into five stages: preprocessing the dataset to clean and standardize activity data (`1_preprocess`), generating a comprehensive set of molecular descriptors (`2_descriptors`), splitting the data into training and testing sets for regression and classification tasks (`3_train_test_split`), performing feature selection using five distinct methods to identify the most relevant descriptors (`4_feature_selection`), and training four machine learning models on the reduced feature sets (`5_model_training`). The optimal combination of feature selection method and machine learning model is chosen based on predictive performance metrics.

---

## Directory Structure

This repository contains the following main directories:

```
TRPM8-bootcamp-project/
├─1_preprocess/
│   ├─ preprocess.ipynb
│   ├─ TRPM8-homosapien-compounds-activities.csv
│   └─ TRPM8-homosapien-compounds-activities-processed.csv
├─2_descriptors/
│   ├─ 3D-descriptors.ipynb
│   ├─ physicochemical-descriptors.ipynb
│   ├─ quantum-descriptors.ipynb
│   ├─ topological-descriptors.ipynb
│   └─ *-descriptors.csv *-descriptors-standardized.csv
├─3_train_test_split/
│   ├─ train_test_stratify.ipynb
│   ├─ descriptors_all.csv
│   └─ train*.csv test*.csv val*.csv
├─4_feature_selection/
│   ├─ correlation_variance_filter/
│   │   ├─ correlation_variance_filter.ipynb
│   │   └─ *.csv
│   ├─ genetic_algorithm/
│   │   ├─ genetic_algorithm.ipynb
│   │   ├─ random_forest/*.csv
│   │   ├─ svm/*.csv
│   │   ├─ xgboost/*.csv
│   │   └─ plots/*
│   ├─ Factor_Analysis/
│   │   ├─ FA_feature_selection.ipynb
│   │   ├─ FA_results/
│   │   │   ├─ factor_components
│   │   │   └─ selected_features
│   │   └─ plots/*
│   ├─ PCA/
│   │   ├─ PCA_Feature_selection.ipynb
│   │   ├─ PCA_results/
│   │   │   └─ PCA_components
│   │   └─ plots/*
│   └─ random_forest_elimination/
│       ├─ random_forest_elimination.ipynb
│       └─ plots/*
├─5_model_training/  
│   ├─ svm/  
│   │   ├─ svm.ipynb  
│   │   └─ outputs/*  
│   ├─ random_forest/  
│   │   ├─ random_forest.ipynb  
│   │   └─ outputs/*  
│   └─ xgboost/  
│       ├─ xgboost.ipynb  
│       └─ outputs/*
├─compare_models.ipynb
│   ├─ regression_performance_summary.csv
│   ├─ classification_performance_summary.csv
│   ├─ model_comparisons_bar_plot.png
│   ├─ radar_plot_accuracy.png
│   └─ radar_plot_mse.png
```

---

## Directory Details

### 1_preprocess

This directory contains scripts and data for the initial preprocessing of activity data.

- **`preprocess.ipynb`**: A Jupyter Notebook that processes activity data for Homo sapiens TRPM8 obtained from [Chembl Activity Data](https://www.ebi.ac.uk/chembl/web_components/explore/target/CHEMBL1075319). The preprocessing steps include:
  - Removing empty activity values, salt ions, and small fragments.
  - Ensuring correct SMILES representation.
  - Removing duplicates.
  - Standardizing IC50 data.

- **`TRPM8-homosapien-compounds-activities.csv`**: Input activity data file downloaded from Chembl.

- **`TRPM8-homosapien-compounds-activities-processed.csv`**: Output data file after preprocessing with `preprocess.ipynb`.

---

### 2_descriptors

This directory contains scripts for generating various molecular descriptors.

- **`3D-descriptors.ipynb`**: Extracts 3D molecular descriptors.
- **`physicochemical-descriptors.ipynb`**: Extracts physicochemical descriptors.
- **`quantum-descriptors.ipynb`**: Extracts quantum descriptors.
- **`topological-descriptors.ipynb`**: Extracts 2D topological descriptors.
- **Outputs**:
  - `[descriptor_name]-descriptors.csv`: Raw descriptor data.
  - `[descriptor_name]-descriptors-standardized.csv`: Standardized descriptor data.

---

### 3_train_test_split

This directory contains files and scripts for splitting the data into training and testing datasets.

- **`descriptors_all.csv`**: Combined descriptor data from `2_descriptors`, including columns for potency values (classification) and IC50 values (regression).
- **`train_test_stratify.ipynb`**: A script that splits the data as follows:
  - **Training Set**: 85% of the data.
  - **Test Set**: 15% of the data.
  - **Train/Validation Split**: Further splits the training set into 5-fold cross-validation.
- **Outputs**:
  - `train_reg.csv`, `train_class.csv`: Training sets for regression and classification tasks.
  - `test_reg.csv`, `test_class.csv`: Test sets for regression and classification tasks.
  - `train_[reg_or_class]_k.csv` and `val_[reg_or_class]_k.csv`: Training and validation sets for each fold (k) in cross-validation for regression and classification tasks.

---

### 4_feature_selection

This directory contains subdirectories and scripts for performing feature selection using various methods.

#### correlation_variance_filter
- **`correlation_variance_filter.ipynb`**:
  - Removes features with a variance less than 0.2 and a correlation greater than 0.95.
  - Inputs: Data from `3_train_test_split`.
  - Outputs: Reduced-feature datasets in the same format as `3_train_test_split`.
  - **Outputs**:
    - `*.csv`: Same outputs as in `3_train_test_split`, with reduced features.

#### genetic_algorithm
- **`genetic_algorithm.ipynb`**:
  - Implements a genetic algorithm for feature selection.
  - Inputs: Data from `3_train_test_split`.
  - Outputs: Reduced features for each machine learning model.
  - **Outputs**:
    - `random_forest/*.csv`, `svm/*.csv`, `xgboost/*.csv`: Reduced-feature datasets for each model.
    - `plots/*`: Plots showing scores vs. generations in the genetic algorithm.

#### Factor_Analysis
- **`FA_feature_selection.ipynb`**:
  - Applies factor analysis for feature selection.
  - Inputs: Data from `3_train_test_split`.
  - Outputs: Reduced-feature datasets and plots.
  - **Subdirectories**:
    - `FA_results/`:
      - `factor_components/`: Contains factor components for dimensionality reduction.
      - `selected_features/`: Contains selected features based on factor loadings.
    - `plots/`: Contains plots such as factor loading distributions, pair plots, and scree plots.

#### PCA
- **`PCA_Feature_selection.ipynb`**:
  - Performs feature selection using Principal Component Analysis (PCA).
  - Inputs: Data from `3_train_test_split`.
  - Outputs: Reduced-feature datasets and plots.
  - **Subdirectories**:
    - `PCA_results/`:
      - `PCA_components/`: Contains PCA components for dimensionality reduction.
    - `plots/`: Contains scree plots and principal component pair plots.

#### random_forest_elimination
- **`random_forest_elimination.ipynb`**:
  - Applies random forest with recursive elimination for dimensionality reduction.
  - Inputs: Data from `3_train_test_split`.
  - Outputs: Reduced-feature datasets and plots.
  - **Subdirectories**:
    - `plots/`: Contains relevant plots for the feature elimination process.

---

### 5_model_training

This directory contains subdirectories and scripts for training machine learning models on features generated from `4_feature_selection`. The models are trained for both classification and regression tasks using the following algorithms:

- **Support Vector Machines (SVM)**  
- **Random Forest**  
- **XGBoost**  

Each model directory contains a corresponding Jupyter Notebook file for training the model and generating evaluation metrics.


#### Model Training I/O

- **Inputs:**  
  Features from `4_feature_selection`, generated by the following feature selection methods:  
  - `random_forest_elimination`  
  - `PCA`  
  - `Factor_Analysis`  
  - `correlation_variance_filter`  
  - `genetic_algorithm`  

- **Outputs:**  

  - **Prediction Files:**  
    - `[model]_[regression_or_classification]_[filter_method]_predictions.csv`: Contains true vs. predicted values for each task.  

  - **Performance Summaries:**  
    - `[model]_[regression_or_classification]_[filter_method]_performance_summary.csv`:  
      - **Regression:** Includes metrics such as MSE, R² score, and Pearson correlation.  
      - **Classification:** Includes metrics such as accuracy, precision, recall, and F1-score.  

  - **Figures:**  
    - **Classification:**  
      - `[model]_classification_[filter_method]_roc_curve.png` (ROC Curve)  
      - `[model]_classification_[filter_method]_confusion_matrix.png` (Confusion Matrix)  

    - **Regression:**  
      - `[model]_regression_[filter_method]_residuals.png` (Residuals Plot)  
      - `[model]_regression_[filter_method]_predicted_vs_true.png` (Predicted vs. True Plot)  

### compare_models.ipynb

This notebook compares the performance of models trained in `5_model_training` across different feature selection methods:

#### Outputs:
- **Performance Summary CSV Files:**
  - `regression_performance_summary.csv`: Summary of regression metrics (will be used as input for the plots).
  - `classification_performance_summary.csv`: Summary of classification metrics (will be used as input for the plots).

- **Visualization Files:**
  - `model_comparisons_bar_plot.png`: Bar plots comparing performance metrics across models and feature selection methods.
  - `radar_plot_accuracy.png`: Radar plot showing accuracy scores for different models and feature selection methods.
  - `radar_plot_mse.png`: Radar plot showing 1/MSE values for regression models.
---
