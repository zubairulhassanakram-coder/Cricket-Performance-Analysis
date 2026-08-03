"""
=====================================================================
01_data_cleaning.py
Cricket Performance Analytics - PSL 2020-2021
Step 3: Data Cleaning and Preprocessing
=====================================================================
Purpose:
    Load the raw ball-by-ball dataset, inspect its structure, handle
    missing/duplicate/inconsistent values, correct data types, and
    save a clean version of the dataset for downstream analysis.
"""

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

RAW_PATH = "../data/raw/psl_cleaned_2020_2021.csv"
CLEAN_PATH = "../data/cleaned/psl_cleaned_final.csv"

# ---------------------------------------------------------------
# 3.1 Load dataset
# ---------------------------------------------------------------
df = pd.read_csv(RAW_PATH)

print("=" * 70)
print("3.1 DATASET OVERVIEW")
print("=" * 70)
print(f"Shape (rows, columns): {df.shape}")
print("\nColumn names:")
print(list(df.columns))
print("\nFirst 5 rows:")
print(df.head())

# ---------------------------------------------------------------
# 3.2 Data types
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("3.2 DATA TYPES (before conversion)")
print("=" * 70)
print(df.dtypes)

# NOTE: The source column for runs scored off the bat is named "runs".
# We rename it to "batsman_runs" to match the standard ball-by-ball
# cricket-analytics naming convention used throughout this project,
# and to avoid confusion with "total_runs" (which includes extras).
df = df.rename(columns={"runs": "batsman_runs"})

# ---------------------------------------------------------------
# 3.3 Missing values
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("3.3 MISSING VALUE CHECK")
print("=" * 70)
missing = df.isna().sum()
print(missing[missing > 0])

# player_of_match has 102 genuine missing values (likely abandoned /
# no-result matches where no Player of the Match was awarded).
# We keep these as NaN because they represent a real absence of an
# award, not a data-entry error.
print(f"\nRows with missing player_of_match: {df['player_of_match'].isna().sum()}")

# ---------------------------------------------------------------
# 3.4 Structural "missing" values explanation
# ---------------------------------------------------------------
# The columns player_dismissed, dismissal_kind, extras_type and fielder
# are NOT missing in the conventional sense. The source data already
# encodes the "no event happened on this ball" case using explicit
# placeholder strings instead of NaN:
#   - extras_type       -> "No Extra"      when no extra was bowled
#   - player_dismissed  -> "Not Dismissed" when no wicket fell
#   - dismissal_kind    -> "Not Out"       when no wicket fell
#   - fielder           -> "No Fielder"    when no fielder was involved
#                                           (e.g. bowled, lbw, or no wicket)
# These are STRUCTURAL missing values: the absence of a value is itself
# meaningful information (most balls are NOT extras and NOT wickets),
# so they must never be dropped or imputed - they are analytically valid
# categories and are used directly in feature engineering (Step 4).
print("\nStructural placeholder categories confirmed:")
for col, placeholder in [
    ("extras_type", "No Extra"),
    ("player_dismissed", "Not Dismissed"),
    ("dismissal_kind", "Not Out"),
    ("fielder", "No Fielder"),
]:
    count = (df[col] == placeholder).sum()
    print(f"  {col:20s} -> '{placeholder}': {count} rows ({count/len(df)*100:.1f}%)")

# ---------------------------------------------------------------
# 3.5 Duplicate check
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("3.5 DUPLICATE CHECK")
print("=" * 70)
dup_all = df.duplicated().sum()
dup_id = df.duplicated(subset=["id"]).sum()
print(f"Fully duplicated rows: {dup_all}")
print(f"Duplicated 'id' values: {dup_id}")
df = df.drop_duplicates()

# ---------------------------------------------------------------
# 3.6 Data type conversion
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("3.6 DATA TYPE CONVERSION")
print("=" * 70)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["season"] = df["season"].astype(int)
df["is_wicket"] = df["is_wicket"].astype(bool)

categorical_cols = [
    "venue", "batting_team", "bowling_team", "batter", "bowler",
    "non_striker", "extras_type", "player_dismissed", "dismissal_kind",
    "fielder", "winner", "match_type",
]
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip()          # 3.7 remove extra spaces
    df[col] = df[col].astype("category")

print("Data types after conversion:")
print(df.dtypes)

# ---------------------------------------------------------------
# 3.7 Remove extra spaces / inconsistent text values
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("3.7 INCONSISTENT VALUE HANDLING")
print("=" * 70)
# Bowler names contain a trailing shirt-number in parentheses, e.g.
# "Mohammad Nawaz (3)". We keep a clean name column for player-level
# aggregation while preserving the original if ever needed.
df["bowler_clean"] = df["bowler"].astype(str).str.replace(
    r"\s*\(\d+\)\s*$", "", regex=True
).str.strip()

# Standardise venue text (strip repeated spaces, fix casing)
df["venue"] = df["venue"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

print("Example bowler name cleaning:")
print(df[["bowler", "bowler_clean"]].drop_duplicates().head(8))

# ---------------------------------------------------------------
# 3.8 Sanity checks on numeric ranges
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("3.8 NUMERIC RANGE SANITY CHECKS")
print("=" * 70)
print(f"over range: {df['over'].min()} - {df['over'].max()}")
print(f"ball range: {df['ball'].min()} - {df['ball'].max()}")
print(f"batsman_runs range: {df['batsman_runs'].min()} - {df['batsman_runs'].max()}")
print(f"total_runs range: {df['total_runs'].min()} - {df['total_runs'].max()}")
print(f"seasons present: {sorted(df['season'].unique())}")

# ---------------------------------------------------------------
# 3.9 Save cleaned dataset
# ---------------------------------------------------------------
df.to_csv(CLEAN_PATH, index=False)
print("\n" + "=" * 70)
print(f"Cleaned dataset saved to: {CLEAN_PATH}")
print(f"Final shape: {df.shape}")
print("=" * 70)
