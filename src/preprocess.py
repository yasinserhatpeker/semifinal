"""
preprocess.py — MICE imputer, CAT_/AA_ kodlama, aykırı değer işleme.
Eksik değer: notebook'taki gibi IterativeImputer (MICE) kullanılır.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

DROP_MISSING_THRESH = 0.80
ID_COLS = ["Variant_ID"]


def drop_high_missing(df: pd.DataFrame, thresh: float = DROP_MISSING_THRESH) -> pd.DataFrame:
    miss_rate = df.isnull().mean()
    drop_cols = miss_rate[miss_rate > thresh].index.tolist()
    print(f"  Cikartilan sutunlar (>%{int(thresh*100)} eksik): {len(drop_cols)}")
    return df.drop(columns=drop_cols, errors="ignore")


def drop_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore")


def mice_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Yalnizca eksik degeri olan numerik sutunlara MICE uygular."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_with_na = [c for c in num_cols if df[c].isna().any()]

    if not cols_with_na:
        return df

    print(f"  MICE imputer: {len(cols_with_na)} sutun isleniyor...")
    imputer = IterativeImputer(max_iter=10, random_state=0, min_value=0)
    df[cols_with_na] = imputer.fit_transform(df[cols_with_na])
    return df


def fill_cat_missing(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "UNKNOWN")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [c for c in df.columns if c.startswith("CAT_")]
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes

    aa_cols = [c for c in df.columns if c.startswith("AA_")]
    if aa_cols:
        df = pd.get_dummies(df, columns=aa_cols, drop_first=True)

    return df


def clip_outliers(df: pd.DataFrame, z_thresh: float = 3.5) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    exclude = {"Label"}
    for col in num_cols:
        if col in exclude:
            continue
        mean, std = df[col].mean(), df[col].std()
        if std > 0:
            df[col] = df[col].clip(mean - z_thresh * std, mean + z_thresh * std)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_id_cols(df)
    df = drop_high_missing(df)
    df = mice_impute(df)
    df = fill_cat_missing(df)
    df = encode_categoricals(df)
    df = clip_outliers(df)
    return df


if __name__ == "__main__":
    inp = PROCESSED_DIR / "combined_raw.csv"
    df = pd.read_csv(inp, low_memory=False)
    df = preprocess(df)
    out = PROCESSED_DIR / "preprocessed.csv"
    df.to_csv(out, index=False)
    print(f"On islenmi veri kaydedildi: {out}  ({df.shape})")
