import os
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_heart_disease():
    """Load the UCI Cleveland Heart Disease dataset (13 features, binary target)."""
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.cleveland.data"
    )
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
    ]
    df = pd.read_csv(url, header=None, names=columns, na_values="?")
    df["target"] = (df["target"] > 0).astype(int)

    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y, list(X.columns), "Heart Disease"


def load_diabetes():
    """Load the Pima Indians Diabetes dataset from bundled CSV (8 features, binary target)."""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "diabetes.csv")
    df = pd.read_csv(os.path.abspath(csv_path))

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y, list(X.columns), "Diabetes"


def load_breast_cancer_data():
    """Load the Wisconsin Breast Cancer dataset via sklearn (30 features, binary target)."""
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    return X, y, list(X.columns), "Breast Cancer"


def load_all_datasets():
    """Return a list of (X, y, feature_names, dataset_name) tuples."""
    return [
        load_heart_disease(),
        load_diabetes(),
        load_breast_cancer_data(),
    ]
