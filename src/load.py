"""
load.py — 4 panel CSV'yi okur ve tek bir DataFrame'de birleştirir.
Paneller: MASTER, KANSER, PAH, CFTR
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

PANEL_FILES = {
    "MASTER": RAW_DIR / "YARISMA_TRAIN_MASTER.csv",
    "KANSER": RAW_DIR / "YARISMA_TRAIN_KANSER.csv",
    "PAH":    RAW_DIR / "YARISMA_TRAIN_PAH.csv",
    "CFTR":   RAW_DIR / "YARISMA_TRAIN_CFTR.csv",
}


def load_panel(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["panel"] = name
    return df


def load_all() -> pd.DataFrame:
    frames = [load_panel(name, path) for name, path in PANEL_FILES.items()]
    combined = pd.concat(frames, ignore_index=True)
    return combined


if __name__ == "__main__":
    df = load_all()
    out = PROCESSED_DIR / "combined_raw.csv"
    df.to_csv(out, index=False)
    print(f"Birleştirilmiş veri kaydedildi → {out}  ({df.shape})")
