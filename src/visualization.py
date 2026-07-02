import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")


def _ensure_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _safe(name):
    return name.lower().replace(" ", "_")


# --- EDA ---

def plot_class_distribution(y, dataset_name):
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(6, 4))
    unique, counts = np.unique(y, return_counts=True)
    ax.bar(unique.astype(str), counts, color=sns.color_palette("viridis", len(unique)), edgecolor="black")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(f"{dataset_name} - Class Distribution")
    for i, (u, c) in enumerate(zip(unique, counts)):
        ax.text(i, c + max(counts) * 0.01, str(c), ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{_safe(dataset_name)}_class_dist.png"), dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(X, dataset_name, max_features=15):
    _ensure_dir()
    X_plot = X.iloc[:, :max_features] if X.shape[1] > max_features else X
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = X_plot.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title(f"{dataset_name} - Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{_safe(dataset_name)}_corr_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_feature_distributions(X, y, dataset_name, max_features=8):
    _ensure_dir()
    cols = list(X.columns[:max_features])
    df = X[cols].copy()
    df["Target"] = y.values if hasattr(y, "values") else y

    ncols = 4
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        sns.boxplot(data=df, x="Target", y=col, ax=axes[i], palette="Set2")
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{dataset_name} - Feature Distributions by Class", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{_safe(dataset_name)}_feat_dist.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- Confusion Matrices ---

def plot_confusion_matrices(cm_dict, dataset_name):
    _ensure_dir()
    n = len(cm_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (model_name, cm) in zip(axes, cm_dict.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
                    ax=ax, cbar=False)
        ax.set_title(model_name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle(f"{dataset_name} - Confusion Matrices", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGURES_DIR, f"{_safe(dataset_name)}_confusion.png"), dpi=150)
    plt.close(fig)


# --- ROC Curves ---

def plot_roc_curves(roc_data, dataset_name):
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = sns.color_palette("husl", len(roc_data))

    for (model_name, (y_test, y_prob)), color in zip(roc_data.items(), colors):
        if y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{dataset_name} - ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{_safe(dataset_name)}_roc.png"), dpi=150)
    plt.close(fig)


# --- Model Comparison ---

def plot_model_comparison(results_df):
    _ensure_dir()
    metrics = ["Accuracy", "F1-Score", "ROC-AUC"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 6))
    for ax, metric in zip(axes, metrics):
        pivot = results_df.pivot(index="Model", columns="Dataset", values=metric)
        pivot.plot(kind="bar", ax=ax, edgecolor="black", width=0.7)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_xlabel("")
        ax.set_ylim(0, 1.05)
        ax.legend(title="Dataset", fontsize=9)
        ax.tick_params(axis="x", rotation=25)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    fig.suptitle("Model Performance Comparison Across Datasets", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)


def plot_cv_comparison(cv_results):
    _ensure_dir()
    datasets = list(cv_results.keys())
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        scores = cv_results[ds]
        bp = ax.boxplot(list(scores.values()), labels=list(scores.keys()), patch_artist=True)
        for patch, color in zip(bp["boxes"], sns.color_palette("pastel", len(scores))):
            patch.set_facecolor(color)
        ax.set_title(ds)
        ax.set_ylabel("Accuracy")
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("5-Fold Cross-Validation Accuracy", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGURES_DIR, "cv_comparison.png"), dpi=150)
    plt.close(fig)


# --- Feature Importance ---

def plot_feature_importance(importances, feature_names, model_name, dataset_name, top_n=10):
    _ensure_dir()
    top_n = min(top_n, len(feature_names))
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(top_n), top_importances[::-1], color=sns.color_palette("viridis", top_n))
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"{dataset_name} - {model_name}\nFeature Importance (Top {top_n})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{_safe(dataset_name)}_{_safe(model_name)}_feat_imp.png"), dpi=150)
    plt.close(fig)
