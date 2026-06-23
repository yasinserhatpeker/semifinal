"""
calibrate.py — OOF tahminleri üzerinde F1-optimum eşik belirleme.
"""
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve, f1_score,
    precision_score, recall_score,
)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUTS_DIR   = Path(__file__).parent.parent / "outputs"
FIGURES_DIR   = OUTPUTS_DIR / "figures"
METRICS_DIR   = OUTPUTS_DIR / "metrics"
TARGET        = "Label"
DROP_COLS     = ["panel"]
N_SPLITS      = 5
RANDOM_STATE  = 42


def oof_probabilities(model, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    for tr, val in skf.split(X, y):
        m = pickle.loads(pickle.dumps(model))
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[val] = m.predict_proba(X.iloc[val])[:, 1]
    return oof


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1[:-1])
    return float(thresholds[best_idx])


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, threshold: float):
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1[:-1])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, lw=2, label="PR egri")
    ax.scatter(rec[best_idx], prec[best_idx], s=120, color="red", zorder=5,
               label=f"Optimum esik={threshold:.3f}  F1={f1[best_idx]:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Egrisi (OOF)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = FIGURES_DIR / "pr_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PR egrisi kaydedildi: {out}")


def calibrate():
    df = pd.read_csv(PROCESSED_DIR / "features.csv", low_memory=False)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    X = df.drop(columns=[TARGET] + [c for c in DROP_COLS if c in df.columns])
    y = df[TARGET]

    with open(OUTPUTS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)

    print("OOF tahminleri hesaplaniyor...")
    oof_prob = oof_probabilities(model, X, y)

    threshold = find_best_threshold(y.values, oof_prob)

    y_pred = (oof_prob >= threshold).astype(int)
    f1  = f1_score(y, y_pred)
    p   = precision_score(y, y_pred)
    r   = recall_score(y, y_pred)

    # 0.5 ile karsilastir
    y_pred_50 = (oof_prob >= 0.5).astype(int)
    f1_50 = f1_score(y, y_pred_50)

    print(f"\n  Optimum esik : {threshold:.4f}")
    print(f"  F1           : {f1:.4f}   (0.50 esiginde: {f1_50:.4f})")
    print(f"  Precision    : {p:.4f}")
    print(f"  Recall       : {r:.4f}")

    plot_pr_curve(y.values, oof_prob, threshold)

    result = {"threshold": threshold, "f1": f1, "precision": p, "recall": r}
    with open(METRICS_DIR / "threshold.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Esik kaydedildi: {METRICS_DIR / 'threshold.json'}")

    return threshold


if __name__ == "__main__":
    calibrate()
