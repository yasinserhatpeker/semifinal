"""
evaluate.py — Metrikler, panel bazlı analiz ve grafikler.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
)
import matplotlib
matplotlib.use("Agg")

FIGURES_DIR = Path(__file__).parent.parent / "outputs" / "figures"
METRICS_DIR = Path(__file__).parent.parent / "outputs" / "metrics"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pos_key = str(1) if str(1) in report else 1
    return {
        "roc_auc":   roc_auc_score(y_true, y_prob),
        "avg_prec":  average_precision_score(y_true, y_prob),
        "f1":        report[pos_key]["f1-score"],
        "precision": report[pos_key]["precision"],
        "recall":    report[pos_key]["recall"],
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, tag: str = ""):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred-NEG", "Pred-POS"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Gercek-NEG", "Gercek-POS"])
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gercek")
    ax.set_title(f"Confusion Matrix {tag}")
    out = FIGURES_DIR / f"confusion_matrix{'_' + tag if tag else ''}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix kaydedildi: {out}")


def evaluate_by_panel(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    y_prob_col: str,
    panel_col: str = "panel",
):
    rows = []
    for panel, grp in df.groupby(panel_col):
        m = compute_metrics(grp[y_true_col].values, grp[y_pred_col].values, grp[y_prob_col].values)
        m["panel"] = panel
        m["n"] = len(grp)
        rows.append(m)
        print(f"Panel={panel}  n={m['n']}  AUC={m['roc_auc']:.3f}  F1={m['f1']:.3f}")

    panel_df = pd.DataFrame(rows).set_index("panel")
    panel_df.to_csv(METRICS_DIR / "panel_metrics.csv")
    return panel_df


def evaluate(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    y_prob_col: str,
    tag: str = "",
):
    y_true = df[y_true_col].values
    y_pred = df[y_pred_col].values
    y_prob = df[y_prob_col].values

    metrics = compute_metrics(y_true, y_pred, y_prob)
    print(f"\n[{tag}] Genel Metrikler:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    pd.DataFrame([metrics]).to_csv(METRICS_DIR / f"metrics{'_' + tag if tag else ''}.csv", index=False)
    plot_confusion_matrix(y_true, y_pred, tag)

    if "panel" in df.columns:
        evaluate_by_panel(df, y_true_col, y_pred_col, y_prob_col)

    return metrics
