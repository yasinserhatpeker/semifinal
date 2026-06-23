"""
validate.py — Çapraz doğrulama ve prior-aware kurgu kontrolleri.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

N_SPLITS = 5
RANDOM_STATE = 42


def stratified_cv_splits(X: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(skf.split(X, y))


def check_prior_consistency(y: pd.Series, expected_pos_rate: float, tol: float = 0.10):
    actual = y.mean()
    if abs(actual - expected_pos_rate) > tol:
        print(f"[UYARI] Gözlemlenen pozitif oran {actual:.3f}, "
              f"beklenen {expected_pos_rate:.3f} ± {tol}")
    else:
        print(f"[OK] Prior tutarlı: {actual:.3f}")
    return actual


def validate(X: pd.DataFrame, y: pd.Series, expected_pos_rate: float = None):
    splits = stratified_cv_splits(X, y)
    print(f"CV bölümleri oluşturuldu: {len(splits)} fold")

    if expected_pos_rate is not None:
        check_prior_consistency(y, expected_pos_rate)

    return splits


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "features.csv", low_memory=False)
    TARGET = "target"  # hedef sütun adını güncelle
    if TARGET not in df.columns:
        raise ValueError(f"'{TARGET}' sütunu bulunamadı.")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    validate(X, y)
