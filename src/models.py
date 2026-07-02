from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def get_models(scale_pos_weight=1.0):
    """Return a dict of configured classifiers with class-weight balancing."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42,
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, class_weight="balanced", random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, use_label_encoder=False, eval_metric="logloss",
            scale_pos_weight=scale_pos_weight, random_state=42,
        ),
    }


def build_pipeline(model):
    """Wrap a model in a Pipeline with StandardScaler (prevents CV data leakage)."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def get_param_grids():
    """Hyperparameter search spaces for RandomizedSearchCV (Pipeline-prefixed)."""
    return {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__solver": ["lbfgs", "liblinear"],
        },
        "SVM": {
            "model__C": [0.1, 1, 10, 50],
            "model__gamma": ["scale", "auto", 0.01, 0.1],
        },
        "Random Forest": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
        },
        "XGBoost": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.1, 0.3],
        },
    }
