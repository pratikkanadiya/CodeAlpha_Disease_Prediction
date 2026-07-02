import os
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.data_loader import load_all_datasets
from src.preprocessing import preprocess_dataset
from src.models import get_models, build_pipeline, get_param_grids
from src.evaluation import (
    evaluate_model, get_confusion_matrix, get_classification_report,
    cross_validate_model, tune_model, build_results_dataframe,
)
from src.visualization import (
    plot_class_distribution, plot_correlation_heatmap, plot_feature_distributions,
    plot_confusion_matrices, plot_roc_curves, plot_model_comparison,
    plot_cv_comparison, plot_feature_importance,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPORTS_DIR = os.path.join("outputs", "reports")
FIGURES_DIR = os.path.join("outputs", "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def banner(text, width=70):
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def main():
    banner("DISEASE PREDICTION FROM MEDICAL DATA")
    print("Loading datasets ...")

    datasets = load_all_datasets()
    param_grids = get_param_grids()

    all_results = {}
    cv_results = {}

    for X, y, feature_names, dataset_name in datasets:
        banner(f"Dataset: {dataset_name}")

        print(f"  Samples  : {X.shape[0]}")
        print(f"  Features : {X.shape[1]}")
        for u, c in zip(*np.unique(y, return_counts=True)):
            print(f"  Class {u} : {c}  ({c / len(y) * 100:.1f}%)")

        # EDA
        print("\n  Generating EDA plots ...")
        plot_class_distribution(y, dataset_name)
        plot_correlation_heatmap(X, dataset_name)
        plot_feature_distributions(X, y, dataset_name)

        # Preprocess
        print("\n  Preprocessing ...")
        X_train, X_test, y_train, y_test, feat_names, X_clean = preprocess_dataset(X, y, dataset_name)
        print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

        # Class-imbalance weight for XGBoost
        n_neg, n_pos = np.sum(y_train == 0), np.sum(y_train == 1)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        models = get_models(scale_pos_weight=scale_pos_weight)
        dataset_results, cm_dict, roc_data, dataset_cv = {}, {}, {}, {}

        for model_name, model in models.items():
            print(f"\n  [{model_name}]")

            pipeline = build_pipeline(model)

            # Hyperparameter tuning
            grid = param_grids.get(model_name, {})
            if grid:
                print("    Tuning ...")
                best_pipeline, best_params, best_cv_score = tune_model(
                    pipeline, grid, X_train, y_train, cv=5, n_iter=20, scoring="f1",
                )
                clean_params = {k.replace("model__", ""): v for k, v in best_params.items()}
                print(f"    Best params : {clean_params}")
                print(f"    Best CV F1  : {best_cv_score:.4f}")
            else:
                best_pipeline = pipeline
                best_pipeline.fit(X_train, y_train)

            # Evaluate
            metrics, y_pred, y_prob = evaluate_model(best_pipeline, X_test, y_test)
            dataset_results[model_name] = metrics
            for k, v in metrics.items():
                print(f"    {k:>10s}: {v:.4f}")

            cm_dict[model_name] = get_confusion_matrix(y_test, y_pred)
            roc_data[model_name] = (y_test, y_prob)

            # Save classification report
            report_path = os.path.join(
                REPORTS_DIR, f"{dataset_name.lower().replace(' ', '_')}_{model_name.lower().replace(' ', '_')}_report.txt",
            )
            with open(report_path, "w") as f:
                f.write(f"Classification Report - {dataset_name} / {model_name}\n")
                f.write("=" * 60 + "\n\n")
                f.write(get_classification_report(y_test, y_pred))

            # Leakage-free cross-validation
            fresh_pipeline = build_pipeline(clone(model))
            cv_scores = cross_validate_model(fresh_pipeline, X_clean.values, y.values, cv=5)
            dataset_cv[model_name] = cv_scores
            print(f"    CV Accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

            # Feature importance (tree-based models only)
            inner_model = best_pipeline.named_steps["model"]
            if hasattr(inner_model, "feature_importances_"):
                plot_feature_importance(inner_model.feature_importances_, feat_names, model_name, dataset_name)

        all_results[dataset_name] = dataset_results
        cv_results[dataset_name] = dataset_cv

        print("\n  Generating model plots ...")
        plot_confusion_matrices(cm_dict, dataset_name)
        plot_roc_curves(roc_data, dataset_name)

    # Overall comparison
    banner("OVERALL COMPARISON")
    results_df = build_results_dataframe(all_results)
    print("\n", results_df.to_string(index=False, float_format="%.4f"))

    csv_path = os.path.join(REPORTS_DIR, "summary_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Summary CSV -> {csv_path}")

    plot_model_comparison(results_df)
    plot_cv_comparison(cv_results)

    # Best model per dataset
    banner("BEST MODELS")
    for ds in results_df["Dataset"].unique():
        subset = results_df[results_df["Dataset"] == ds]
        best = subset.loc[subset["F1-Score"].idxmax()]
        print(f"  {ds:>15s}  ->  {best['Model']}  (F1={best['F1-Score']:.4f}, AUC={best['ROC-AUC']:.4f})")

    banner("DONE")
    print("  Figures -> outputs/figures/")
    print("  Reports -> outputs/reports/")


if __name__ == "__main__":
    main()
