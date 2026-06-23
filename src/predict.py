"""
predict.py — Notebook tekniklerini taklit eder:
  1. Majority class downsample (notebook: UA 832K → 70K)
  2. SMOTE sadece TRAIN fold icinde (data leakage yok)
  3. XGBoost + reg_lambda (notebook: reg_lambda=10)
  4. 5-fold stratified CV
  5. Macro F1 esik kalibrasyonu (her iki sinifi dengeler)
"""
import pickle, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUTS_DIR   = Path(__file__).parent.parent / "outputs"
METRICS_DIR   = OUTPUTS_DIR / "metrics"
TARGET        = "Label"
DROP_COLS     = ["panel"]
N_OUTER       = 5
N_INNER       = 4
RANDOM_STATE  = 42

# ── Sınıf dengeleme parametreleri (notebook mantığı) ─────────
# Pozitif (1) sınıfı 2817 → 1800'e downsample
# Negatif (0) sınıfı SMOTE ile 985 → 1800'e upsample
MAJ_DOWNSAMPLE = 1800   # pozitif sinif tavan
MIN_UPSAMPLE   = 1800   # negatif sinif SMOTE hedefi


def build_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        reg_lambda=10,        # notebook'tan: aşırı öğrenmeyi baskıla
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def balance_train(X_tr: pd.DataFrame, y_tr: pd.Series):
    """Notebook adımları: majority downsample → SMOTE minority upsample."""
    pos_idx = y_tr[y_tr == 1].index
    neg_idx = y_tr[y_tr == 0].index

    # 1. Pozitif sınıfı downsample (notebook: UA 832K → 70K)
    if len(pos_idx) > MAJ_DOWNSAMPLE:
        pos_idx = pd.Index(
            np.random.default_rng(RANDOM_STATE).choice(pos_idx, MAJ_DOWNSAMPLE, replace=False)
        )

    X_bal = pd.concat([X_tr.loc[pos_idx], X_tr.loc[neg_idx]])
    y_bal = pd.concat([y_tr.loc[pos_idx], y_tr.loc[neg_idx]])

    # 2. SMOTE ile negatif sinifi upsample (notebook: custom sampling_strategy)
    n_neg_now = (y_bal == 0).sum()
    n_pos_now = (y_bal == 1).sum()

    smote_strategy = {
        0: max(MIN_UPSAMPLE, n_neg_now),   # negatifi artir
        1: n_pos_now,                       # pozitif oldugu gibi kalsin
    }

    smote = SMOTE(sampling_strategy=smote_strategy, random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_bal, y_bal)
    return X_res, y_res


def find_macro_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.20, 0.80, 121):
        score = f1_score(y_true, (y_prob >= t).astype(int), average="macro")
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t)


def sep(title=""):
    print(f"\n{'=' * 60}")
    if title:
        print(f"  {title}")
        print("=" * 60)


# ── Veri yukle ────────────────────────────────────────────────
df = pd.read_csv(PROCESSED_DIR / "features.csv", low_memory=False)
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

X = df.drop(columns=[TARGET] + [c for c in DROP_COLS if c in df.columns])
y = df[TARGET]

sep("VERI OZETI")
print(f"  Toplam : {len(df)} satir  |  Ozellik: {X.shape[1]}")
print(f"  NEG(0) : {(y==0).sum()}  |  POS(1): {(y==1).sum()}  |  Oran: {y.mean():.3f}")
print(f"\n  [Dengeleme] POS downsample → {MAJ_DOWNSAMPLE}  |  SMOTE NEG → {MIN_UPSAMPLE}")

# ── 5-fold dis CV ─────────────────────────────────────────────
sep("5-FOLD STRATIFIED CV  [XGBoost + SMOTE]")

outer_skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)
inner_skf = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)

all_true = np.zeros(len(y), dtype=int)
all_prob = np.zeros(len(y))
all_pred = np.zeros(len(y), dtype=int)
fold_rows = []

for fold, (tr_idx, te_idx) in enumerate(outer_skf.split(X, y), 1):
    X_tr_raw, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr_raw, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    # IC OOF: esik kalibrasyonu — SMOTE sadece ic train parcasinda
    inner_oof = np.zeros(len(X_tr_raw))
    for itr, ival in inner_skf.split(X_tr_raw, y_tr_raw):
        X_itr, y_itr = balance_train(
            X_tr_raw.iloc[itr], y_tr_raw.iloc[itr]
        )
        m = build_model()
        m.fit(X_itr, y_itr)
        inner_oof[ival] = m.predict_proba(X_tr_raw.iloc[ival])[:, 1]

    threshold = find_macro_f1_threshold(y_tr_raw.values, inner_oof)

    # Dis fold: tum train parcasini dengele ve egit
    X_bal, y_bal = balance_train(X_tr_raw, y_tr_raw)
    model = build_model()
    model.fit(X_bal, y_bal)

    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob >= threshold).astype(int)

    mf1 = f1_score(y_te, pred, average="macro")
    f1p = f1_score(y_te, pred, pos_label=1)
    f1n = f1_score(y_te, pred, pos_label=0)
    auc = roc_auc_score(y_te, prob)

    n_bal_neg = (y_bal == 0).sum()
    n_bal_pos = (y_bal == 1).sum()
    print(f"  Fold {fold}  esik={threshold:.3f}  "
          f"AUC={auc:.4f}  MacroF1={mf1:.4f}  "
          f"F1-NEG={f1n:.4f}  F1-POS={f1p:.4f}  "
          f"[train: NEG={n_bal_neg} POS={n_bal_pos}]")

    fold_rows.append({"fold": fold, "threshold": threshold,
                      "auc": auc, "macro_f1": mf1, "f1_neg": f1n, "f1_pos": f1p})
    all_true[te_idx] = y_te.values
    all_prob[te_idx] = prob
    all_pred[te_idx] = pred

# ── Fold ozeti ────────────────────────────────────────────────
fold_df = pd.DataFrame(fold_rows).set_index("fold")
sep("FOLD ORTALAMA +/- STD")
for col in fold_df.columns:
    print(f"  {col:12s}: {fold_df[col].mean():.4f}  (+/- {fold_df[col].std():.4f})")

# ── Toplu OOF metrikleri ──────────────────────────────────────
sep("TOPLU OOF TEST METRIKLERI")
print(f"  AUC-ROC      : {roc_auc_score(all_true, all_prob):.4f}")
print(f"  Avg Precision: {average_precision_score(all_true, all_prob):.4f}")
print(f"  Macro F1     : {f1_score(all_true, all_pred, average='macro'):.4f}   <-- hedef")
print(f"  F1 Negatif   : {f1_score(all_true, all_pred, pos_label=0):.4f}")
print(f"  F1 Pozitif   : {f1_score(all_true, all_pred, pos_label=1):.4f}")
print(f"  Precision    : {precision_score(all_true, all_pred):.4f}")
print(f"  Recall       : {recall_score(all_true, all_pred):.4f}")

print(f"\n  Classification Report:")
print(classification_report(all_true, all_pred, target_names=["Negatif", "Pozitif"]))

cm = confusion_matrix(all_true, all_pred)
print("  Confusion Matrix:")
print(f"    {'':10s}  Pred-NEG  Pred-POS")
print(f"    Gercek-NEG   {cm[0,0]:6d}    {cm[0,1]:6d}")
print(f"    Gercek-POS   {cm[1,0]:6d}    {cm[1,1]:6d}")

# ── Kaydet ────────────────────────────────────────────────────
pd.DataFrame({"y_true": all_true, "y_prob": all_prob, "y_pred": all_pred}).to_csv(
    METRICS_DIR / "test_predictions.csv", index=False
)
fold_df.to_csv(METRICS_DIR / "cv_fold_metrics.csv")

avg_threshold = fold_df["threshold"].mean()
with open(METRICS_DIR / "threshold.json", "w") as f:
    json.dump({"threshold": round(avg_threshold, 4), "criterion": "macro_f1"}, f, indent=2)

# Full model (tum veri dengeli) kaydet
sep("FULL MODEL  (tum veri, kayit icin)")
X_full, y_full = balance_train(X, y)
full_model = build_model()
full_model.fit(X_full, y_full)
with open(OUTPUTS_DIR / "model.pkl", "wb") as f:
    pickle.dump({"model": full_model, "threshold": avg_threshold}, f)

print(f"  Tahminler  : {METRICS_DIR / 'test_predictions.csv'}")
print(f"  Esik (ort) : {avg_threshold:.4f}  -> threshold.json")
print(f"  Full model : {OUTPUTS_DIR / 'model.pkl'}")
print("=" * 60)
