"""
features.py — Missingness flag'leri ve panel dummy degiskenleri.
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
DROP_MISSING_THRESH = 0.80


def add_missingness_flags(processed: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Ham veride eksik olan ve islenmis veride hala var olan sutunlar icin flag ekler."""
    flags = {}
    for col in raw.columns:
        if col not in processed.columns:
            continue
        miss_rate = raw[col].isna().mean()
        if 0 < miss_rate <= DROP_MISSING_THRESH:
            flags[f"MISSING_{col}"] = raw[col].isna().astype(int).values
    if flags:
        flag_df = pd.DataFrame(flags, index=processed.index)
        processed = pd.concat([processed, flag_df], axis=1)
    return processed


def add_panel_dummies(df: pd.DataFrame) -> pd.DataFrame:
    if "panel" in df.columns:
        df = pd.get_dummies(df, columns=["panel"], drop_first=True)
    return df


def engineer_features(processed: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    processed = add_missingness_flags(processed, raw)
    processed = add_panel_dummies(processed)
    return processed


if __name__ == "__main__":
    raw = pd.read_csv(PROCESSED_DIR / "combined_raw.csv", low_memory=False)
    processed = pd.read_csv(PROCESSED_DIR / "preprocessed.csv", low_memory=False)
    out_df = engineer_features(processed.copy(), raw)
    out = PROCESSED_DIR / "features.csv"
    out_df.to_csv(out, index=False)
    print(f"Ozellikler kaydedildi: {out}  ({out_df.shape})")
