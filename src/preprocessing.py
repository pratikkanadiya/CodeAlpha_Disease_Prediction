import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def handle_missing_values(X, dataset_name):
    """Replace invalid zeros (Diabetes) and NaNs with median values."""
    X = X.copy()

    if dataset_name == "Diabetes":
        zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in zero_invalid:
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)

    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)


def preprocess_dataset(X, y, dataset_name, test_size=0.2, random_state=42):
    """
    Clean the data and perform a stratified train/test split.

    Scaling is handled inside sklearn Pipelines to prevent
    data leakage during cross-validation and tuning.

    Returns: X_train, X_test, y_train, y_test, feature_names, X_clean
    """
    X_clean = handle_missing_values(X, dataset_name)
    feature_names = list(X_clean.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train.values, y_test.values, feature_names, X_clean
