"""
train_panel.py — claude.md §4 tam uygulama
  1. AL_ blok istatistikleri        (bio_features.py'de)
  2. Optuna Bayesian optimizasyon    (CatBoost hiperparametre arama)
  3. Ensemble soft voting            (CatBoost + LightGBM + XGBoost)
  4. Sabit esik 0.5533

Kullanim:
    python src/train_panel.py CFTR
    python src/train_panel.py CFTR 0.5533 50   # panel esik n_trials
"""
import sys, json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

ROOT          = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR       = ROOT / "data" / "raw"
OUTPUTS_DIR   = ROOT / "outputs"
METRICS_DIR   = OUTPUTS_DIR / "metrics"

TARGET       = "Label"
DROP_COLS    = ["panel"]
N_FOLDS      = 5
RANDOM_STATE = 42
TOP_N_LIVE   = 5
TOP_N_FINAL  = 20

PANEL_FILES = {
    "MASTER": RAW_DIR / "YARISMA_TRAIN_MASTER.csv",
    "KANSER": RAW_DIR / "YARISMA_TRAIN_KANSER.csv",
    "PAH":    RAW_DIR / "YARISMA_TRAIN_PAH.csv",
    "CFTR":   RAW_DIR / "YARISMA_TRAIN_CFTR.csv",
}


def sep(title=""):
    print(f"\n{'=' * 60}")
    if title:
        print(f"  {title}")
        print("=" * 60)


# ── 1. Veri yükleme — ham veri, NaN korunur ──────────────────
def load_panel_features(panel: str):
    """
    features.csv yerine combined_raw.csv'den doğrudan okur.
    Yüksek-eksik sütunlar KORUNUR, CatBoost NaN'ı içsel olarak işler.
    """
    from bio_features import add_bio_features

    raw = pd.read_csv(PROCESSED_DIR / "combined_raw.csv", low_memory=False)
    mask  = raw["panel"] == panel
    raw_p = raw[mask].reset_index(drop=True)

    df = raw_p.copy()

    # Kimlik sütunlarını çıkar
    df = df.drop(columns=["Variant_ID", "panel"], errors="ignore")

    # CAT_ → label encoding (NaN → -1)
    for col in [c for c in df.columns if c.startswith("CAT_")]:
        df[col] = pd.Categorical(df[col]).codes  # eksik → -1

    # AA_ → one-hot (NaN satırları 0 alır)
    aa_cols = [c for c in df.columns if c.startswith("AA_")]
    if aa_cols:
        df = pd.get_dummies(df, columns=aa_cols, drop_first=True)
    df[df.select_dtypes("bool").columns] = df.select_dtypes("bool").astype(int)

    # Biyolojik özellikler (bio_features.py)
    # — raw_p'den NaN'lı haliyle hesaplanır
    df = add_bio_features(df, raw_p)

    return df, raw_p


# ── 2. Optuna: CatBoost hiperparametre optimizasyonu ─────────
def optuna_catboost(X: pd.DataFrame, y: pd.Series, n_trials: int) -> dict:
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "iterations":   trial.suggest_int("iterations", 100, 600),
            "depth":        trial.suggest_int("depth", 3, 6),
            "l2_leaf_reg":  trial.suggest_float("l2_leaf_reg", 1, 15),
            "learning_rate":trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "auto_class_weights": "Balanced",
            "eval_metric": "F1",
            "random_seed": RANDOM_STATE,
            "verbose": 0,
        }
        scores = []
        for tr, val in skf.split(X, y):
            m = CatBoostClassifier(**params)
            m.fit(X.iloc[tr], y.iloc[tr])
            prob = m.predict_proba(X.iloc[val])[:, 1]
            pred = (prob >= 0.5533).astype(int)
            scores.append(f1_score(y.iloc[val], pred, average="macro", zero_division=0))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ── 3. Ensemble modelleri ─────────────────────────────────────
def build_catboost(params: dict) -> CatBoostClassifier:
    p = {k: v for k, v in params.items()}
    p.update({"auto_class_weights": "Balanced", "eval_metric": "F1",
               "random_seed": RANDOM_STATE, "verbose": 0})
    return CatBoostClassifier(**p)


def build_lgbm(n_samples: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300 if n_samples < 200 else 500,
        max_depth=4,
        reg_lambda=5,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=-1,
    )


def build_xgb(n_samples: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300 if n_samples < 200 else 500,
        max_depth=4,
        reg_lambda=10,
        learning_rate=0.05,
        scale_pos_weight=(1 - 0.811) / 0.811,  # CFTR prior
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
    )


def ensemble_predict(models: list, X: pd.DataFrame) -> np.ndarray:
    """Soft voting: modellerin tahmin olasılıklarının ortalaması."""
    probs = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=1)
    return probs.mean(axis=1)


# ── 4. Özellik önemi tablosu ──────────────────────────────────
def print_importance_table(imp: pd.Series, title: str, top_n: int):
    top = imp.head(top_n).reset_index()
    top.columns = ["Ozellik", "Onem"]
    top["Onem%"] = (top["Onem"] / imp.sum() * 100).round(2)
    top["Grup"] = top["Ozellik"].apply(
        lambda c: "BIO-BLOK" if "alblk" in c else
                  "BIO"  if c.startswith("BIO_") else
                  "EK"   if c.startswith("EK_") else
                  "AL"   if c.startswith("AL_") else
                  "MISS" if c.startswith("MISSING_") else "DIGER"
    )
    top.index = range(1, len(top) + 1)
    print(f"\n  {title}")
    print(f"  {'#':>3}  {'Ozellik':<22}  {'Onem%':>6}  Grup")
    print(f"  {'-'*3}  {'-'*22}  {'-'*6}  {'-'*8}")
    for i, row in top.iterrows():
        bar = "▪" * max(1, int(row["Onem%"] / 2))
        print(f"  {i:>3}  {row['Ozellik']:<22}  {row['Onem%']:>5.1f}%  {row['Grup']:<8}  {bar}")


# ── 5. Ana eğitim fonksiyonu ──────────────────────────────────
def train_panel(panel: str, threshold: float = 0.5533, n_trials: int = 40):
    sep(f"PANEL: {panel}  [Esik={threshold}  Optuna trials={n_trials}]")

    feat_panel, _ = load_panel_features(panel)
    X = feat_panel.drop(columns=[TARGET] + [c for c in DROP_COLS if c in feat_panel.columns],
                        errors="ignore")
    y = feat_panel[TARGET]

    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    n_pos_rate = n_pos / len(y)
    print(f"  Satirlar  : {len(y)}")
    print(f"  POS(1)    : {n_pos}   NEG(0): {n_neg}   Oran: {y.mean():.3f}")
    print(f"  Ozellikler: {X.shape[1]}  (AL blok ozellikler dahil)")

    if n_neg < 5:
        print(f"  [UYARI] Cok az negatif ornek ({n_neg}).")
        return

    # ── Optuna ────────────────────────────────────────────────
    sep(f"OPTUNA — CatBoost Bayesian Optimizasyon ({n_trials} deneme)")
    print("  Arama devam ediyor", end="", flush=True)

    def _progress(study, trial):
        if trial.number % 10 == 9:
            print(f" [{trial.number+1}✓ best={study.best_value:.4f}]", end="", flush=True)

    skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "iterations":        trial.suggest_int("iterations", 100, 600),
            "depth":             trial.suggest_int("depth", 3, 6),
            "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 1, 15),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "auto_class_weights": "Balanced",
            "eval_metric": "F1", "random_seed": RANDOM_STATE, "verbose": 0,
        }
        scores = []
        for tr, val in skf3.split(X, y):
            m = CatBoostClassifier(**params)
            m.fit(X.iloc[tr], y.iloc[tr])
            prob = m.predict_proba(X.iloc[val])[:, 1]
            pred = (prob >= threshold).astype(int)
            scores.append(f1_score(y.iloc[val], pred, average="macro", zero_division=0))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, callbacks=[_progress])
    print()

    best_params = study.best_params
    print(f"\n  En iyi Macro F1 (3-fold): {study.best_value:.4f}")
    print(f"  En iyi parametreler:")
    for k, v in best_params.items():
        print(f"    {k:<22}: {v}")

    # ── 5-Fold CV — Ensemble ──────────────────────────────────
    n_splits  = min(N_FOLDS, n_neg)
    skf_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    all_prob        = np.zeros(len(y))
    all_pred        = np.zeros(len(y), dtype=int)
    fold_rows       = []
    fold_importances = []

    sep(f"5-FOLD CV — ENSEMBLE  [CatBoost + LightGBM + XGBoost | esik={threshold}]")

    for fold, (tr_idx, te_idx) in enumerate(skf_outer.split(X, y), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        # Üç model — her biri ayrı eğitilir
        cb  = build_catboost(best_params)
        lgb = build_lgbm(len(X_tr))
        xgb = build_xgb(len(X_tr))

        cb.fit(X_tr, y_tr)
        lgb.fit(X_tr, y_tr)
        xgb.fit(X_tr, y_tr)

        # Soft voting
        prob = ensemble_predict([cb, lgb, xgb], X_te)
        pred = (prob >= threshold).astype(int)

        mf1 = f1_score(y_te, pred, average="macro", zero_division=0)
        f1n = f1_score(y_te, pred, pos_label=0,     zero_division=0)
        f1p = f1_score(y_te, pred, pos_label=1,     zero_division=0)
        try:
            auc = roc_auc_score(y_te, prob)
        except ValueError:
            auc = float("nan")

        # CatBoost ozellik onemleri (en yorumlanabilir)
        imp = pd.Series(cb.feature_importances_, index=X.columns).sort_values(ascending=False)
        fold_importances.append(imp)
        top5 = ", ".join(f"{c}({v:.1f}%)"
                         for c, v in zip(imp.head(TOP_N_LIVE).index,
                                         imp.head(TOP_N_LIVE).values / imp.sum() * 100))

        print(f"\n  ── Fold {fold} ──────────────────────────────────────")
        print(f"  AUC={auc:.4f}  MacroF1={mf1:.4f}  F1-NEG={f1n:.4f}  F1-POS={f1p:.4f}")
        print(f"  Top-{TOP_N_LIVE}: {top5}")

        fold_rows.append({"fold": fold, "auc": auc,
                          "macro_f1": mf1, "f1_neg": f1n, "f1_pos": f1p})
        all_prob[te_idx] = prob
        all_pred[te_idx] = pred

    # ── Fold özet tablosu ─────────────────────────────────────
    fold_df = pd.DataFrame(fold_rows).set_index("fold")
    sep("FOLD ORTALAMA +/- STD")
    print(f"\n  {'Metrik':<12}  {'Ortalama':>8}  {'Std':>7}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*7}")
    for col in fold_df.columns:
        print(f"  {col:<12}  {fold_df[col].mean():>8.4f}  {fold_df[col].std():>7.4f}")

    # ── OOF metrikleri ────────────────────────────────────────
    sep("TOPLU OOF TEST METRIKLERI  (Ensemble)")
    try:
        oof_auc = roc_auc_score(y.values, all_prob)
    except ValueError:
        oof_auc = float("nan")
    oof_mf1 = f1_score(y.values, all_pred, average="macro",   zero_division=0)
    oof_f1n = f1_score(y.values, all_pred, pos_label=0,       zero_division=0)
    oof_f1p = f1_score(y.values, all_pred, pos_label=1,       zero_division=0)

    print(f"\n  {'Metrik':<16}  {'Deger':>7}")
    print(f"  {'-'*16}  {'-'*7}")
    print(f"  {'AUC-ROC':<16}  {oof_auc:>7.4f}")
    print(f"  {'Macro F1':<16}  {oof_mf1:>7.4f}  <-- hedef")
    print(f"  {'F1 Negatif':<16}  {oof_f1n:>7.4f}")
    print(f"  {'F1 Pozitif':<16}  {oof_f1p:>7.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(y.values, all_pred,
                                target_names=["Negatif (Benign)", "Pozitif (Patojenik)"],
                                zero_division=0))

    cm  = confusion_matrix(y.values, all_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion Matrix:")
    print(f"  {'':14s}  Pred-NEG  Pred-POS")
    print(f"  {'Gercek-NEG':<14}  {tn:>8d}  {fp:>8d}  (Benign    toplam={n_neg})")
    print(f"  {'Gercek-POS':<14}  {fn:>8d}  {tp:>8d}  (Patojenik toplam={n_pos})")

    # ── Özellik önemi tabloları ───────────────────────────────
    avg_imp = pd.concat(fold_importances, axis=1).mean(axis=1).sort_values(ascending=False)

    sep(f"OZELLIK ONEMLERI — TOP {TOP_N_FINAL}  (CatBoost fold ortalaması)")
    print_importance_table(avg_imp, f"Panel: {panel}", TOP_N_FINAL)

    sep("OZELLIK GRUP OZETI")
    grp_map = {
        "BIO-BLOK (BIO_alblk)":  [c for c in avg_imp.index if "alblk" in c],
        "Biyolojik (BIO_)":       [c for c in avg_imp.index if c.startswith("BIO_") and "alblk" not in c],
        "Evrimsel EK_":           [c for c in avg_imp.index if c.startswith("EK_")],
        "Allel AL_":              [c for c in avg_imp.index if c.startswith("AL_")],
        "Eksiklik MISSING_":      [c for c in avg_imp.index if c.startswith("MISSING_")],
        "Kategorik CAT_":         [c for c in avg_imp.index if c.startswith("CAT_")],
    }
    total_imp = avg_imp.sum()
    print(f"\n  {'Grup':<30}  {'Toplam%':>7}  Gorsel")
    print(f"  {'-'*30}  {'-'*7}  {'-'*20}")
    for label, cols in grp_map.items():
        pct = avg_imp[cols].sum() / total_imp * 100 if cols else 0.0
        bar = "█" * int(pct / 2)
        print(f"  {label:<30}  {pct:>6.2f}%  {bar}")

    # ── Kaydet ───────────────────────────────────────────────
    # Final: tüm veriyle 3 model birden eğit
    sep("FINAL MODEL KAYDI  (tum veri)")
    final_cb  = build_catboost(best_params);  final_cb.fit(X, y)
    final_lgb = build_lgbm(len(X));           final_lgb.fit(X, y)
    final_xgb = build_xgb(len(X));            final_xgb.fit(X, y)

    model_path = OUTPUTS_DIR / f"model_{panel}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "models":    [final_cb, final_lgb, final_xgb],
            "threshold": threshold,
            "features":  list(X.columns),
            "best_params": best_params,
        }, f)

    result = {
        "panel": panel, "threshold": threshold,
        "oof_auc": round(float(oof_auc), 4),
        "macro_f1": round(oof_mf1, 4),
        "f1_neg":   round(oof_f1n, 4),
        "f1_pos":   round(oof_f1p, 4),
        "best_params": best_params,
    }
    with open(METRICS_DIR / f"panel_{panel}_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    fold_df.to_csv(METRICS_DIR / f"panel_{panel}_folds.csv")
    avg_imp.to_csv(METRICS_DIR / f"panel_{panel}_importance.csv", header=["importance"])

    print(f"  Model     : {model_path}")
    print(f"  Metrikler : {METRICS_DIR / f'panel_{panel}_metrics.json'}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim: python src/train_panel.py <PANEL> [esik] [n_trials]")
        print("  Ornek : python src/train_panel.py CFTR 0.5533 40")
        sys.exit(1)

    panel     = sys.argv[1].upper()
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5533
    n_trials  = int(sys.argv[3])   if len(sys.argv) > 3 else 40

    if panel not in PANEL_FILES:
        print(f"Gecersiz panel: {panel}. Secenekler: {list(PANEL_FILES.keys())}")
        sys.exit(1)

    train_panel(panel, threshold=threshold, n_trials=n_trials)
