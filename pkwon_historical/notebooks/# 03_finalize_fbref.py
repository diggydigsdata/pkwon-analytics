# 03_finalize_fbref.py
from pathlib import Path
from functools import reduce
import pandas as pd

# ---- Settings you can tweak when running in VS Code ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "cleaned_and_merged" / "big5_standard_misc_merged" / "big5_standard_misc.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "final"
CURRENT_SEASON = "2024-2025"  # excluded from historical export

# ---- Helpers: structural ----
def coalesce(*series):
    return reduce(lambda a, b: a.combine_first(b), series)

def build_canonical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Canonical identity fields
    df["nationality"]  = coalesce(df.get("nationality_standard"),  df.get("nationality_misc"))
    df["country_code"] = coalesce(df.get("country_code_standard"), df.get("country_code_misc"))
    df["position"]     = coalesce(df.get("position_standard"),     df.get("position_misc"))
    df["age"]          = coalesce(df.get("age_standard"),          df.get("age_misc"))
    df["birth_year"]   = coalesce(df.get("birth_year_standard"),   df.get("birth_year_misc"))

    return df

def drop_redundant_source_cols(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [
        "nationality_standard","country_code_standard","position_standard","age_standard","birth_year_standard",
        "nationality_misc","country_code_misc","position_misc","age_misc","birth_year_misc",
    ]
    return df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")

def dedupe_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_nnz"] = df.notna().sum(axis=1)
    df = df.sort_values("_nnz", ascending=False)
    df = df.drop_duplicates(subset=["player","team","league","season"], keep="first")
    return df.drop(columns="_nnz")

# ---- Helpers: typing / cleaning ----
def clean_strings(df: pd.DataFrame, cols) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    for c in cols:
        df[c] = (
            df[c].astype("string")
                 .str.strip()
                 .replace({"": pd.NA})
        )
    return df

def coerce_nullable_ints(df, cols):
    cols = [c for c in cols if c in df.columns]
    for c in cols:
        s = df[c]
        # sanitize common junk first
        if s.dtype == "object":
            s = (s.astype("string")
                   .str.replace(r"\u2009|\u00A0", "", regex=True)  # thin/nbsp spaces
                   .str.replace(",", "", regex=False)              # thousands sep
                   .str.replace("—", "", regex=False)              # em dash
                   .str.replace("-", "", regex=False)              # plain dash used as NA
                )
        s = pd.to_numeric(s, errors="coerce")
        try:
            df[c] = s.astype("Int64")
        except Exception as e:
            print(f"[coerce_nullable_ints] Failed on '{c}': {e}. Keeping as float.")
            df[c] = s.astype(float)
    return df

def coerce_floats(df: pd.DataFrame, cols) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df

# ---- Main flow ----
def main():
    # 1) Load
    df = pd.read_csv(INPUT_CSV)

    # 2) Build canonical, drop redundant source cols
    df = build_canonical(df)
    df = drop_redundant_source_cols(df)

    # 3) Tidy strings on canonical identifiers
    canon_str = ["player","team","league","season","nationality","country_code","position"]
    df = clean_strings(df, canon_str)

    # 4) Enforce types on kept columns (post-canonical)
    int_cols = [
        "age","birth_year","matches_played","starts","minutes",
        "goals","assists","goals_and_assists","non_penalty_goals",
        "pens_scored","pens_attempted","fouls_drawn","offsides","pkwon","progressive_carries","prgp"
    ]
    float_cols = [
        "goals_per90","assists_per90","goals_and_assists_per90","non_penalty_goals_per90",
        "xg_expected_goals","npxg_non_penalty_xg","xag_expected_assisted_goals","npxg+xag","non_penalty_goals_and_assists"  # remove if you didn’t compute it
    ]
    df = coerce_nullable_ints(df, int_cols)
    df = coerce_floats(df, float_cols)

    # 5) Dedupe
    df = dedupe_rows(df)

    # 6) Split historical vs current (optional)
    historical = df[df["season"] != CURRENT_SEASON].copy()

    # 8) Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    historical.to_csv(OUTPUT_DIR / "big5_player_season_final.csv", index=False, encoding="utf-8")
    try:
        historical.to_parquet(OUTPUT_DIR / "big5_player_season_final.parquet", index=False)
    except Exception as e:
        print("Parquet save skipped:", e)

    print("✅ Finalized historical dataset written to", OUTPUT_DIR)

if __name__ == "__main__":
    main()