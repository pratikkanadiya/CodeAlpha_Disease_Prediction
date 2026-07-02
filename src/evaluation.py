import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV


def evaluate_model(model, X_test, y_test):
    """Compute test-set metrics and return (metrics_dict, y_pred, y_prob)."""
    y_pred = model.predict(X_test)

    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)

    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
    }
    return metrics, y_pred, y_prob


def get_confusion_matrix(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)


def get_classification_report(y_test, y_pred, output_dict=False):
    return classification_report(y_test, y_pred, output_dict=output_dict)


def cross_validate_model(model, X, y, cv=5, scoring="accuracy"):
    """Stratified k-fold cross-validation. Accepts raw data + Pipeline."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    return cross_val_score(model, X, y, cv=skf, scoring=scoring)


def tune_model(pipeline, param_grid, X_train, y_train,
               cv=5, n_iter=20, scoring="f1"):
    """Run RandomizedSearchCV and return (best_pipeline, best_params, best_score)."""
    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=min(n_iter, n_combos),
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scoring,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def build_results_dataframe(all_results):
    """Convert {dataset: {model: metrics}} into a tidy DataFrame."""
    rows = []
    for dataset_name, model_results in all_results.items():
        for model_name, metrics in model_results.items():
            row = {"Dataset": dataset_name, "Model": model_name}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)
