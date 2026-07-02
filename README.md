# Disease Prediction from Medical Data

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-2.0%2B-brightgreen?logo=xgboost" alt="XGBoost">
</p>

A machine learning pipeline that predicts the possibility of **Heart Disease**, **Diabetes**, and **Breast Cancer** using patient clinical data. Four classification algorithms are trained, tuned, and compared across all three datasets.

---

## Datasets

| Dataset | Samples | Features | Source |
|---------|---------|----------|--------|
| Heart Disease (Cleveland) | 303 | 13 | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease) |
| Diabetes (Pima Indians) | 768 | 8 | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| Breast Cancer (Wisconsin) | 569 | 30 | [scikit-learn](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) |

## Algorithms

| Algorithm | Type | Key Strength |
|-----------|------|--------------|
| Logistic Regression | Linear | Fast, interpretable baseline |
| SVM (RBF kernel) | Kernel-based | Effective in high-dimensional spaces |
| Random Forest | Ensemble (Bagging) | Handles non-linearity, built-in feature importance |
| XGBoost | Ensemble (Boosting) | State-of-the-art on tabular data |

## Project Structure

```
CodeAlpha_Disease/
├── data/
│   └── diabetes.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── visualization.py
├── outputs/
│   ├── figures/
│   └── reports/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Krutarth-Talaviya-0712/CodeAlpha_Disease_Prediction.git
cd CodeAlpha_Disease

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

## Pipeline Overview

```
Load Datasets
     │
     ▼
Exploratory Data Analysis ──────── Class distributions, correlations, box plots
     │
     ▼
Preprocessing ──────────────────── Missing value imputation, stratified split
     │
     ▼
Hyperparameter Tuning ──────────── RandomizedSearchCV (F1-optimised, 5-fold CV)
     │
     ▼
Model Training & Evaluation ────── Accuracy, Precision, Recall, F1, ROC-AUC
     │
     ▼
Feature Importance ─────────────── Top predictors for tree-based models
     │
     ▼
Comparison & Reporting ─────────── Summary CSV, charts, classification reports
```

## Key Design Decisions

- **No data leakage** — `StandardScaler` is bundled inside `sklearn.Pipeline`, so it refits on every CV fold and tuning iteration.
- **Class imbalance handling** — `class_weight='balanced'` for LR/SVM/RF; `scale_pos_weight` for XGBoost, computed dynamically per dataset.
- **Stratified splits** — Both train/test split and cross-validation preserve class ratios.
- **Reproducibility** — All models and splits use `random_state=42`.

## Results

| Dataset | Best Model | F1-Score | ROC-AUC |
|---------|-----------|----------|---------|
| Heart Disease | Random Forest | 0.9000 | 0.9621 |
| Diabetes | Random Forest | 0.6885 | 0.8257 |
| Breast Cancer | SVM | 0.9790 | 0.9967 |

<details>
<summary><strong>Full results table</strong></summary>

| Dataset | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------|-------|----------|-----------|--------|----------|---------|
| Heart Disease | Logistic Regression | 0.8689 | 0.8125 | 0.9286 | 0.8667 | 0.9502 |
| Heart Disease | SVM | 0.8361 | 0.7500 | 0.9643 | 0.8438 | 0.9416 |
| Heart Disease | **Random Forest** | **0.9016** | **0.8438** | **0.9643** | **0.9000** | **0.9621** |
| Heart Disease | XGBoost | 0.8525 | 0.8276 | 0.8571 | 0.8421 | 0.9264 |
| Diabetes | Logistic Regression | 0.7208 | 0.5846 | 0.7037 | 0.6387 | 0.8106 |
| Diabetes | SVM | 0.7273 | 0.5909 | 0.7222 | 0.6500 | 0.8139 |
| Diabetes | **Random Forest** | **0.7532** | **0.6176** | **0.7778** | **0.6885** | **0.8257** |
| Diabetes | XGBoost | 0.7532 | 0.6212 | 0.7593 | 0.6833 | 0.8086 |
| Breast Cancer | Logistic Regression | 0.9561 | 0.9855 | 0.9444 | 0.9645 | 0.9954 |
| Breast Cancer | **SVM** | **0.9737** | **0.9859** | **0.9722** | **0.9790** | **0.9967** |
| Breast Cancer | Random Forest | 0.9474 | 0.9714 | 0.9444 | 0.9577 | 0.9944 |
| Breast Cancer | XGBoost | 0.9474 | 0.9459 | 0.9722 | 0.9589 | 0.9950 |

</details>

## Outputs

After running `python main.py`, the pipeline generates:

- **`outputs/figures/`** — EDA plots, confusion matrices, ROC curves, model comparison charts, feature importance bar plots
- **`outputs/reports/`** — Per-model classification reports (`.txt`) and a summary CSV

## Evaluation Metrics

| Metric | Why It Matters |
|--------|---------------|
| Accuracy | Overall correctness |
| Precision | Minimise false positives |
| Recall | Minimise missed disease cases (critical in healthcare) |
| F1-Score | Harmonic mean of Precision and Recall |
| ROC-AUC | Model's ability to discriminate between classes |

## Requirements

- Python 3.8+
- See [requirements.txt](requirements.txt) for full dependency list

## License

This project is licensed under the MIT License.

## Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
