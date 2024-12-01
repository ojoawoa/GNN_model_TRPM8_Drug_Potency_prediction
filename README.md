# Project Overview

## Directory Structure

This repository contains three main directories:

```
BreadcrumbsTRPM8-bootcamp-project/
├─1_preprocess/
│   ├─ preprocess.ipynb
│   ├─ TRPM8-homosapien-compounds-activities.csv
│   └─ TRPM8-homosapien-compounds-activities-processed.csv
├─2_descriptors/
│   ├─ 3D-descriptors.ipynb
│   ├─ physicochemical-descriptors.ipynb
│   ├─ quantum-descriptors.ipynb
│   └─ topological-descriptors.ipynb
└─3_train_test_split/
    ├─ descriptors_all.csv
    └─ train_test_stratify.ipynb
```

---

## Directory Details

### 1_preprocess
This directory contains files and scripts used for initial data preprocessing.

- **preprocess.ipynb**:
  - **Description**: This Jupyter Notebook processes activity data for Homo sapiens TRPM8 sourced from Chembl. It includes steps for data cleaning and standardization.
  - **Data Source**: [Chembl Activity Data]([https://www.ebi.ac.uk/chembl/web_components/explore/activities/STATE_ID:QcxDH17OZ95LqHehOkjCEg%3D%3D](https://www.ebi.ac.uk/chembl/web_components/explore/target/CHEMBL1075319))
  - **Processing Steps**:
    - Remove empty activity values, salt ions, and small fragments
    - Ensure correct SMILES representation
    - Remove duplicates
    - Standardize IC50 data

- **TRPM8-homosapien-compounds-activities.csv**:
  - Input activity data file from Chembl.

- **TRPM8-homosapien-compounds-activities-processed.csv**:
  - Output data file after processing in `preprocess.ipynb`.

---

### 2_descriptors
This directory contains scripts for extracting various types of molecular descriptors.

- **3D-descriptors.ipynb**:
  - Extracts 3D descriptors from the processed data.

- **physicochemical-descriptors.ipynb**:
  - Extracts physicochemical descriptors.

- **quantum-descriptors.ipynb**:
  - Extracts quantum descriptors.

- **topological-descriptors.ipynb**:
  - Extracts 2D topological descriptors.

- **Outputs**:
  - Files named `[descriptor_name]-descriptors.csv` for raw descriptor data.
  - Files named `[descriptor_name]-descriptors-standardized.csv` for standardized descriptor data.

---

### 3_train_test_split
This directory contains files related to the train-test split process.

- **descriptors_all.csv**:
  - Combined descriptor data from the `2_descriptors` directory.

- **train_test_stratify.ipynb**:
  - Splits the combined descriptor data into training and testing datasets with the following proportions:
    - **Training Set**: 85% of the data
    - **Test Set**: 15% of the data
    - **Train/Validation Split**: 90%/10% for 5-fold cross-validation
  - **Outputs**:
    - `train_set.csv` and `test_set.csv`
    - `train_fold_k.csv` and `val_fold_k.csv` for each fold (k) in cross-validation

---
