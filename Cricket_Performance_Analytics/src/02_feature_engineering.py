"""
=====================================================================
02_feature_engineering.py
Cricket Performance Analytics - PSL 2020-2021
Step 4: Feature Engineering
=====================================================================
Purpose:
    Derive analytically useful ball-level features from the cleaned
    dataset so that phase-wise, boundary, and wicket analysis can be
    performed efficiently in later steps.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("../data/cleaned/psl_cleaned_final.csv")
df["date"] = pd.to_datetime(df["date"])

# ---------------------------------------------------------------
# 4.1 dot_ball
# Why: A dot ball (0 runs off the bat, no extras) shows bowling
# pressure/control. It is the base metric for run-rate & pressure
# analysis, and helps identify economical bowlers.
# ---------------------------------------------------------------
df["dot_ball"] = ((df["batsman_runs"] == 0) & (df["total_runs"] == 0)).astype(int)

# ---------------------------------------------------------------
# 4.2 four / six / boundary
# Why: Boundaries are the clearest indicator of attacking intent and
# batting power. Separating fours and sixes lets us compare players'
# and teams' hitting styles; "boundary" combines both for overall
# boundary-percentage metrics.
# ---------------------------------------------------------------
df["four"] = (df["batsman_runs"] == 4).astype(int)
df["six"] = (df["batsman_runs"] == 6).astype(int)
df["boundary"] = ((df["four"] == 1) | (df["six"] == 1)).astype(int)

# ---------------------------------------------------------------
# 4.3 wicket
# Why: Flags balls on which a wicket fell, independent of dismissal
# type. This underpins bowler strike-rate, team collapse and phase-
# wise wicket analysis.
# ---------------------------------------------------------------
df["wicket"] = df["is_wicket"].astype(int)

# ---------------------------------------------------------------
# 4.4 ball_count
# Why: A constant helper column (=1 per delivery) that makes
# groupby(...).sum() calls for "balls faced/bowled" trivial and
# readable, and is used as the denominator for strike-rate and
# economy-rate calculations.
# ---------------------------------------------------------------
df["ball_count"] = 1

# ---------------------------------------------------------------
# 4.5 powerplay / middle_over / death_over
# Why: T20 innings are strategically divided into three phases with
# different risk/run profiles. Tagging each ball by phase is essential
# for phase-wise scoring-rate, wicket, and strategy analysis.
#   Powerplay   : overs 1-6   (field restrictions, attacking starts)
#   Middle overs: overs 7-15  (rebuilding / rotation phase)
#   Death overs : overs 16-20 (aggressive finishing overs)
# ---------------------------------------------------------------
df["powerplay"] = df["over"].between(1, 6).astype(int)
df["middle_over"] = df["over"].between(7, 15).astype(int)
df["death_over"] = df["over"].between(16, 20).astype(int)


def phase_label(over):
    if 1 <= over <= 6:
        return "Powerplay"
    elif 7 <= over <= 15:
        return "Middle Overs"
    else:
        return "Death Overs"


df["match_phase"] = df["over"].apply(phase_label)

# ---------------------------------------------------------------
# 4.6 run_type
# Why: A single categorical label for every delivery's scoring
# outcome (dot, single/double/triple, four, six, or extra) simplifies
# distribution and pie-chart style analysis of how runs are scored.
# ---------------------------------------------------------------


def classify_run(row):
    if row["is_wicket"] and row["total_runs"] == 0:
        return "Wicket"
    if row["batsman_runs"] == 6:
        return "Six"
    if row["batsman_runs"] == 4:
        return "Four"
    if row["batsman_runs"] in (1, 2, 3):
        return "Running Runs"
    if row["batsman_runs"] == 0 and row["total_runs"] > 0:
        return "Extra"
    return "Dot Ball"


df["run_type"] = df.apply(classify_run, axis=1)

# ---------------------------------------------------------------
# 4.7 Additional helper features used across the EDA section
# ---------------------------------------------------------------
df["is_extra"] = (df["extras_type"] != "No Extra").astype(int)
df["extra_runs"] = df["total_runs"] - df["batsman_runs"]
df["match_year"] = df["date"].dt.year
df["match_month"] = df["date"].dt.to_period("M").astype(str)

# Save feature-engineered dataset
OUT_PATH = "../data/cleaned/psl_features.csv"
df.to_csv(OUT_PATH, index=False)

print("Feature engineering complete.")
print(f"New shape: {df.shape}")
print("New columns added:")
new_cols = [
    "dot_ball", "four", "six", "boundary", "wicket", "ball_count",
    "powerplay", "middle_over", "death_over", "match_phase", "run_type",
    "is_extra", "extra_runs", "match_year", "match_month",
]
print(new_cols)
print(f"\nSaved to: {OUT_PATH}")
print("\nrun_type distribution:")
print(df["run_type"].value_counts())
