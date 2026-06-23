"""
bio_features.py — Biyolojik özellik mühendisliği (claude.md stratejisi)
  - BLOSUM62 ve PAM250 amino asit ikame skorları
  - Kyte-Doolittle hidrofobiklik farkı
  - Grantham mesafesi
  - Satır bazlı NaN sayısı (nadir varyant sinyali)
  - EK_ ve AL_ satır istatistikleri (mean / max / std)
"""
import numpy as np
import pandas as pd

# ── BLOSUM62 ──────────────────────────────────────────────────
BLOSUM62 = {
    ('A','A'):4,('A','R'):-1,('A','N'):-2,('A','D'):-2,('A','C'):0,
    ('A','Q'):-1,('A','E'):-1,('A','G'):0,('A','H'):-2,('A','I'):-1,
    ('A','L'):-1,('A','K'):-1,('A','M'):-1,('A','F'):-2,('A','P'):-1,
    ('A','S'):1,('A','T'):0,('A','W'):-3,('A','Y'):-2,('A','V'):0,
    ('R','R'):5,('R','N'):-1,('R','D'):-2,('R','C'):-3,('R','Q'):1,
    ('R','E'):0,('R','G'):-2,('R','H'):0,('R','I'):-3,('R','L'):-2,
    ('R','K'):2,('R','M'):-1,('R','F'):-3,('R','P'):-2,('R','S'):-1,
    ('R','T'):-1,('R','W'):-3,('R','Y'):-2,('R','V'):-3,
    ('N','N'):6,('N','D'):1,('N','C'):-3,('N','Q'):0,('N','E'):0,
    ('N','G'):0,('N','H'):1,('N','I'):-3,('N','L'):-3,('N','K'):0,
    ('N','M'):-2,('N','F'):-3,('N','P'):-2,('N','S'):1,('N','T'):0,
    ('N','W'):-4,('N','Y'):-2,('N','V'):-3,
    ('D','D'):6,('D','C'):-3,('D','Q'):0,('D','E'):2,('D','G'):-1,
    ('D','H'):-1,('D','I'):-3,('D','L'):-4,('D','K'):-1,('D','M'):-3,
    ('D','F'):-3,('D','P'):-1,('D','S'):0,('D','T'):-1,('D','W'):-4,
    ('D','Y'):-3,('D','V'):-3,
    ('C','C'):9,('C','Q'):-3,('C','E'):-4,('C','G'):-3,('C','H'):-3,
    ('C','I'):-1,('C','L'):-1,('C','K'):-3,('C','M'):-1,('C','F'):-2,
    ('C','P'):-3,('C','S'):-1,('C','T'):-1,('C','W'):-2,('C','Y'):-2,
    ('C','V'):-1,
    ('Q','Q'):5,('Q','E'):2,('Q','G'):-2,('Q','H'):0,('Q','I'):-3,
    ('Q','L'):-2,('Q','K'):1,('Q','M'):0,('Q','F'):-3,('Q','P'):-1,
    ('Q','S'):0,('Q','T'):-1,('Q','W'):-2,('Q','Y'):-1,('Q','V'):-2,
    ('E','E'):5,('E','G'):-2,('E','H'):0,('E','I'):-3,('E','L'):-3,
    ('E','K'):1,('E','M'):-2,('E','F'):-3,('E','P'):-1,('E','S'):0,
    ('E','T'):-1,('E','W'):-3,('E','Y'):-2,('E','V'):-2,
    ('G','G'):6,('G','H'):-2,('G','I'):-4,('G','L'):-4,('G','K'):-2,
    ('G','M'):-3,('G','F'):-3,('G','P'):-2,('G','S'):0,('G','T'):-2,
    ('G','W'):-2,('G','Y'):-3,('G','V'):-3,
    ('H','H'):8,('H','I'):-3,('H','L'):-3,('H','K'):-1,('H','M'):-2,
    ('H','F'):-1,('H','P'):-2,('H','S'):-1,('H','T'):-2,('H','W'):-2,
    ('H','Y'):2,('H','V'):-3,
    ('I','I'):4,('I','L'):2,('I','K'):-1,('I','M'):1,('I','F'):0,
    ('I','P'):-3,('I','S'):-2,('I','T'):-1,('I','W'):-3,('I','Y'):-1,
    ('I','V'):3,
    ('L','L'):4,('L','K'):-2,('L','M'):2,('L','F'):0,('L','P'):-3,
    ('L','S'):-2,('L','T'):-1,('L','W'):-2,('L','Y'):-1,('L','V'):1,
    ('K','K'):5,('K','M'):-1,('K','F'):-3,('K','P'):-1,('K','S'):0,
    ('K','T'):-1,('K','W'):-3,('K','Y'):-2,('K','V'):-2,
    ('M','M'):5,('M','F'):0,('M','P'):-2,('M','S'):-1,('M','T'):-1,
    ('M','W'):-1,('M','Y'):-1,('M','V'):1,
    ('F','F'):6,('F','P'):-4,('F','S'):-2,('F','T'):-2,('F','W'):1,
    ('F','Y'):3,('F','V'):-1,
    ('P','P'):7,('P','S'):-1,('P','T'):-1,('P','W'):-4,('P','Y'):-3,
    ('P','V'):-2,
    ('S','S'):4,('S','T'):1,('S','W'):-3,('S','Y'):-2,('S','V'):-2,
    ('T','T'):5,('T','W'):-2,('T','Y'):-2,('T','V'):0,
    ('W','W'):11,('W','Y'):2,('W','V'):-3,
    ('Y','Y'):7,('Y','V'):-1,
    ('V','V'):4,
}
# simetrik yap
_blosum = {}
for (a, b), v in BLOSUM62.items():
    _blosum[(a, b)] = v
    _blosum[(b, a)] = v
BLOSUM62 = _blosum

# ── PAM250 (sadece diyagonal + sık çiftler) ───────────────────
PAM250 = {
    ('A','A'):2,('A','R'):-2,('A','N'):0,('A','D'):0,('A','C'):-2,
    ('A','Q'):0,('A','E'):0,('A','G'):1,('A','H'):-1,('A','I'):-1,
    ('A','L'):-2,('A','K'):-1,('A','M'):-1,('A','F'):-3,('A','P'):1,
    ('A','S'):1,('A','T'):1,('A','W'):-6,('A','Y'):-3,('A','V'):0,
    ('R','R'):6,('R','N'):0,('R','D'):-1,('R','C'):-4,('R','Q'):1,
    ('R','E'):-1,('R','G'):-3,('R','H'):2,('R','I'):-2,('R','L'):-3,
    ('R','K'):3,('R','M'):0,('R','F'):-4,('R','P'):0,('R','S'):-1,
    ('R','T'):-1,('R','W'):2,('R','Y'):-4,('R','V'):-2,
    ('N','N'):2,('N','D'):2,('N','C'):-4,('N','Q'):1,('N','E'):1,
    ('N','G'):0,('N','H'):2,('N','I'):-2,('N','L'):-3,('N','K'):1,
    ('N','M'):-2,('N','F'):-3,('N','P'):-1,('N','S'):1,('N','T'):0,
    ('N','W'):-4,('N','Y'):-2,('N','V'):-2,
    ('D','D'):4,('D','C'):-5,('D','Q'):2,('D','E'):3,('D','G'):1,
    ('D','H'):1,('D','I'):-2,('D','L'):-4,('D','K'):0,('D','M'):-3,
    ('D','F'):-6,('D','P'):-1,('D','S'):0,('D','T'):0,('D','W'):-7,
    ('D','Y'):-4,('D','V'):-2,
    ('C','C'):12,('C','Q'):-5,('C','E'):-5,('C','G'):-3,('C','H'):-3,
    ('C','I'):-2,('C','L'):-6,('C','K'):-5,('C','M'):-5,('C','F'):-4,
    ('C','P'):-3,('C','S'):0,('C','T'):-2,('C','W'):-8,('C','Y'):0,
    ('C','V'):-2,
    ('Q','Q'):4,('Q','E'):2,('Q','G'):-1,('Q','H'):3,('Q','I'):-2,
    ('Q','L'):-2,('Q','K'):1,('Q','M'):1,('Q','F'):-5,('Q','P'):0,
    ('Q','S'):-1,('Q','T'):-1,('Q','W'):-5,('Q','Y'):-4,('Q','V'):-2,
    ('E','E'):4,('E','G'):0,('E','H'):1,('E','I'):-2,('E','L'):-3,
    ('E','K'):0,('E','M'):-2,('E','F'):-5,('E','P'):-1,('E','S'):0,
    ('E','T'):0,('E','W'):-7,('E','Y'):-4,('E','V'):-2,
    ('G','G'):5,('G','H'):-2,('G','I'):-3,('G','L'):-4,('G','K'):-2,
    ('G','M'):-3,('G','F'):-5,('G','P'):0,('G','S'):1,('G','T'):0,
    ('G','W'):-7,('G','Y'):-5,('G','V'):-1,
    ('H','H'):6,('H','I'):-2,('H','L'):-2,('H','K'):0,('H','M'):-2,
    ('H','F'):-2,('H','P'):0,('H','S'):-1,('H','T'):-1,('H','W'):-3,
    ('H','Y'):0,('H','V'):-2,
    ('I','I'):5,('I','L'):2,('I','K'):-2,('I','M'):2,('I','F'):1,
    ('I','P'):-2,('I','S'):-1,('I','T'):0,('I','W'):-5,('I','Y'):-1,
    ('I','V'):4,
    ('L','L'):6,('L','K'):-3,('L','M'):4,('L','F'):2,('L','P'):-3,
    ('L','S'):-3,('L','T'):-2,('L','W'):-2,('L','Y'):-1,('L','V'):2,
    ('K','K'):5,('K','M'):0,('K','F'):-5,('K','P'):-1,('K','S'):0,
    ('K','T'):0,('K','W'):-3,('K','Y'):-4,('K','V'):-2,
    ('M','M'):6,('M','F'):0,('M','P'):-2,('M','S'):-2,('M','T'):-1,
    ('M','W'):-4,('M','Y'):-2,('M','V'):2,
    ('F','F'):9,('F','P'):-5,('F','S'):-3,('F','T'):-3,('F','W'):0,
    ('F','Y'):7,('F','V'):-1,
    ('P','P'):6,('P','S'):1,('P','T'):0,('P','W'):-6,('P','Y'):-5,
    ('P','V'):1,
    ('S','S'):2,('S','T'):1,('S','W'):-2,('S','Y'):-3,('S','V'):-1,
    ('T','T'):3,('T','W'):-5,('T','Y'):-3,('T','V'):0,
    ('W','W'):17,('W','Y'):0,('W','V'):-6,
    ('Y','Y'):10,('Y','V'):-2,
    ('V','V'):4,
}
_pam = {}
for (a, b), v in PAM250.items():
    _pam[(a, b)] = v
    _pam[(b, a)] = v
PAM250 = _pam

# ── Kyte-Doolittle hidrofobiklik ──────────────────────────────
HYDROPHOBICITY = {
    'A': 1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C': 2.5,
    'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I': 4.5,
    'L': 3.8,'K':-3.9,'M': 1.9,'F': 2.8,'P':-1.6,
    'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V': 4.2,
}

# ── Grantham mesafesi bileşenleri (kimyasal kompozisyon, hacim, polarite) ──
GRANTHAM_C = {'A':0,'R':0.65,'N':0.33,'D':0.37,'C':0.54,'Q':0.33,'E':0.37,'G':0,'H':0.54,'I':0,'L':0,'K':0.58,'M':0,'F':0,'P':0.39,'S':0.42,'T':0.42,'W':0.25,'Y':0.76,'V':0}
GRANTHAM_V = {'A':31,'R':124,'N':56,'D':54,'C':55,'Q':85,'E':83,'G':3,'H':96,'I':111,'L':111,'K':119,'M':105,'F':132,'P':32.5,'S':32,'T':61,'W':170,'Y':136,'V':84}
GRANTHAM_P = {'A':8.1,'R':10.5,'N':11.6,'D':13.0,'C':5.5,'Q':10.5,'E':12.3,'G':9.0,'H':10.4,'I':5.2,'L':4.9,'K':11.3,'M':5.7,'F':5.2,'P':8.0,'S':9.2,'T':8.6,'W':5.4,'Y':6.2,'V':5.9}

def grantham(a: str, b: str) -> float:
    if a not in GRANTHAM_C or b not in GRANTHAM_C:
        return np.nan
    dc = (GRANTHAM_C[a] - GRANTHAM_C[b]) ** 2
    dv = (GRANTHAM_V[a] - GRANTHAM_V[b]) ** 2
    dp = (GRANTHAM_P[a] - GRANTHAM_P[b]) ** 2
    return float(np.sqrt(1.833 * dc + 0.1018 * dv + 0.000399 * dp))


def aa_score(a: str, b: str, matrix: dict) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(matrix.get((str(a).upper(), str(b).upper()), np.nan))


def add_bio_features(df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """claude.md'deki tüm biyolojik özellikler."""

    aa1 = raw_df["AA_1"] if "AA_1" in raw_df.columns else None
    aa2 = raw_df["AA_2"] if "AA_2" in raw_df.columns else None

    if aa1 is not None and aa2 is not None:
        df["BIO_blosum62"]   = [aa_score(a, b, BLOSUM62)   for a, b in zip(aa1, aa2)]
        df["BIO_pam250"]     = [aa_score(a, b, PAM250)     for a, b in zip(aa1, aa2)]
        df["BIO_hydro_diff"] = [
            HYDROPHOBICITY.get(str(a).upper(), np.nan) - HYDROPHOBICITY.get(str(b).upper(), np.nan)
            if not (pd.isna(a) or pd.isna(b)) else np.nan
            for a, b in zip(aa1, aa2)
        ]
        df["BIO_grantham"]   = [grantham(str(a).upper(), str(b).upper())
                                 if not (pd.isna(a) or pd.isna(b)) else np.nan
                                 for a, b in zip(aa1, aa2)]
        # NaN → median (nadir vakalarda protein değişimi bilinmiyor)
        for col in ["BIO_blosum62","BIO_pam250","BIO_hydro_diff","BIO_grantham"]:
            df[col] = df[col].fillna(df[col].median())

    # Nadir varyant sinyali: ham verideki satır bazlı NaN sayısı
    df["BIO_missing_count"] = raw_df.isnull().sum(axis=1).values

    # EK_ istatistikleri
    ek_cols = [c for c in df.columns if c.startswith("EK_")]
    if ek_cols:
        df["BIO_ek_mean"] = df[ek_cols].mean(axis=1)
        df["BIO_ek_max"]  = df[ek_cols].max(axis=1)
        df["BIO_ek_std"]  = df[ek_cols].std(axis=1).fillna(0)

    # AL_ global istatistikleri
    al_cols = [c for c in df.columns if c.startswith("AL_")]
    if al_cols:
        df["BIO_al_mean"] = df[al_cols].mean(axis=1)
        df["BIO_al_max"]  = df[al_cols].max(axis=1)
        df["BIO_al_std"]  = df[al_cols].std(axis=1).fillna(0)

        # claude.md §4 — AL_ sütun bloklaması (popülasyon kökenine göre 5'erli gruplar)
        # Her blok, benzer popülasyon/veritabanı kaynağından gelen sütunları temsil eder
        block_size = 5
        blocks = [al_cols[i:i + block_size] for i in range(0, len(al_cols), block_size)]
        flag_cols = {}
        for b_idx, block in enumerate(blocks):
            prefix = f"BIO_alblk{b_idx:02d}"
            flag_cols[f"{prefix}_mean"] = df[block].mean(axis=1)
            flag_cols[f"{prefix}_max"]  = df[block].max(axis=1)
            flag_cols[f"{prefix}_std"]  = df[block].std(axis=1).fillna(0)
        df = pd.concat([df, pd.DataFrame(flag_cols, index=df.index)], axis=1)

    return df
