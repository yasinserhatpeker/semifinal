"""
train.py — Stratified CV egitimi + terminal metrikleri.
Optimum esik calibrate.py tarafindan outputs/metrics/threshold.json'a kaydedilir.
"""
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedKFold

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUTS_DIR   = Path(__file__).parent.parent / "outputs"
TARGET        = "Label"
RANDOM_STATE  = 42
N_SPLITS      = 5
DROP_COLS     = ["panel"]   # panel dummy'leri features.py'de ekleniyor ama ham haliyle kalırsa çıkar


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",   # %74/%26 dengesizliğini dengeler
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


def print_separator(title: str = ""):
    line = "=" * 60
    print(f"\n{line}")
    if title:
        print(f"  {title}")
        print(line)


def train_and_evaluate(df: pd.DataFrame):
    X = df.drop(columns=[TARGET] + [c for c in DROP_COLS if c in df.columns])
    y = df[TARGET]

    # bool sütunları int'e çevir (get_dummies çıktısı)
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    print_separator("VERI OZETI")
    print(f"  Satir: {len(df)}  |  Ozellik: {X.shape[1]}")
    print(f"  Pozitif oran: {y.mean():.3f}  ({y.sum()} / {len(y)})")

    # Kayitli esigi yukle, yoksa 0.5 kullan
    thresh_path = OUTPUTS_DIR / "metrics" / "threshold.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            threshold = json.load(f)["threshold"]
        print(f"  Esik (calibrate.py): {threshold:.4f}")
    else:
        threshold = 0.5
        print(f"  Esik (varsayilan): {threshold:.4f}  -- once calibrate.py calistirin")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics = []
    print_separator(f"CAPRAZ DOGRULAMA ({N_SPLITS} FOLD)  [esik={threshold:.4f}]")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        pipe = build_pipeline()
        pipe.fit(X_tr, y_tr)

        y_prob = pipe.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)   # kalibre esik

        m = {
            "fold":      fold,
            "auc":       roc_auc_score(y_val, y_prob),
            "avg_prec":  average_precision_score(y_val, y_prob),
            "f1":        f1_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall":    recall_score(y_val, y_pred),
        }
        fold_metrics.append(m)
        print(f"  Fold {fold}  |  AUC={m['auc']:.4f}  "
              f"AP={m['avg_prec']:.4f}  F1={m['f1']:.4f}  "
              f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}")

    fm = pd.DataFrame(fold_metrics).drop(columns="fold")
    print_separator("CV ORTALAMA +/- STD")
    for col in fm.columns:
        print(f"  {col:10s}: {fm[col].mean():.4f}  (+/- {fm[col].std():.4f})")

    # Final modeli tum veriyle egit
    print_separator("FINAL MODEL (tum veri)")
    final_pipe = build_pipeline()
    final_pipe.fit(X, y)

    y_prob_all = final_pipe.predict_proba(X)[:, 1]
    y_pred_all = (y_prob_all >= threshold).astype(int)   # kalibre esik

    print(f"\n  Classification Report  [esik={threshold:.4f}]:")
    print(classification_report(y, y_pred_all, target_names=["Negatif", "Pozitif"]))

    cm = confusion_matrix(y, y_pred_all)
    print("  Confusion Matrix:")
    print(f"    {'':10s}  Pred-NEG  Pred-POS")
    print(f"    Gercek-NEG   {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"    Gercek-POS   {cm[1,0]:6d}    {cm[1,1]:6d}")

    # Ozellik onemleri
    rf = final_pipe.named_steps["clf"]
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print_separator("EN ONEMLI 15 OZELLIK")
    print(feat_imp.head(15).to_string())

    # Kaydet
    OUTPUTS_DIR.mkdir(exist_ok=True)
    model_path = OUTPUTS_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_pipe, f)
    feat_imp.to_csv(OUTPUTS_DIR / "metrics" / "feature_importance.csv", header=["importance"])

    pd.DataFrame(fold_metrics).to_csv(OUTPUTS_DIR / "metrics" / "cv_metrics.csv", index=False)
    print(f"\n  Model kaydedildi: {model_path}")
    print("=" * 60)

    return final_pipe


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "features.csv", low_memory=False)

    # bool sutunlari var ise int'e cevir
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    train_and_evaluate(df)
