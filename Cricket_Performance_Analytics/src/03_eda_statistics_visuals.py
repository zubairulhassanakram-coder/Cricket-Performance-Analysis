"""
=====================================================================
03_eda_statistics_visuals.py
Cricket Performance Analytics - PSL 2020-2021
Steps 5, 6 & 7: EDA, Statistical Analysis, 20 Visualizations
=====================================================================
All 20 charts are saved as PNG files into ../images/
A text summary of every numeric result is printed to stdout and also
captured to ../reports/analysis_output_log.txt for use in the report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

IMG = "../images/"
df = pd.read_csv("../data/cleaned/psl_features.csv")
df["date"] = pd.to_datetime(df["date"])

PALETTE = "viridis"
TEAM_PALETTE = "Set2"

# =====================================================================
# helper: match-level table (one row per match) - needed for win/venue stats
# =====================================================================
match_df = df.drop_duplicates(subset="match_id")[
    ["match_id", "date", "season", "venue", "winner", "win_by",
     "match_type", "player_of_match"]
].reset_index(drop=True)

# team total runs per match/innings (for run distribution per match)
team_innings = (
    df.groupby(["match_id", "inning", "batting_team"])["total_runs"]
    .sum().reset_index()
)

print("#" * 70)
print("STEP 5: EXPLORATORY DATA ANALYSIS (EDA)")
print("#" * 70)

# ---------------------------------------------------------------
# 5.1 Dataset Summary
# ---------------------------------------------------------------
print("\n--- 5.1 DATASET SUMMARY ---")
print(f"Total deliveries (rows): {len(df)}")
print(f"Total matches: {df['match_id'].nunique()}")
print(f"Total teams: {df['batting_team'].nunique()}")
print(f"Total unique batters: {df['batter'].nunique()}")
print(f"Total unique bowlers: {df['bowler_clean'].nunique()}")
print(f"Total venues: {df['venue'].nunique()}")
print(f"Seasons covered: {sorted(df['season'].unique())}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ---------------------------------------------------------------
# 5.2 Descriptive Statistics
# ---------------------------------------------------------------
print("\n--- 5.2 DESCRIPTIVE STATISTICS (numeric columns) ---")
desc = df[["batsman_runs", "total_runs", "extra_runs", "over", "ball"]].describe()
print(desc)

# ---------------------------------------------------------------
# 6. STATISTICAL ANALYSIS (mean, median, mode, std, var, corr, cov, skew, kurtosis)
# ---------------------------------------------------------------
print("\n" + "#" * 70)
print("STEP 6: STATISTICAL ANALYSIS")
print("#" * 70)

for col in ["batsman_runs", "total_runs"]:
    s = df[col]
    print(f"\nColumn: {col}")
    print(f"  Mean     : {s.mean():.4f}")
    print(f"  Median   : {s.median():.4f}")
    print(f"  Mode     : {s.mode().iloc[0]}")
    print(f"  Std Dev  : {s.std():.4f}")
    print(f"  Variance : {s.var():.4f}")
    print(f"  Skewness : {s.skew():.4f}")
    print(f"  Kurtosis : {s.kurtosis():.4f}")

corr_matrix = df[["batsman_runs", "total_runs", "extra_runs", "is_wicket",
                   "over", "ball", "dot_ball", "boundary"]].astype(float).corr()
print("\nCorrelation matrix:")
print(corr_matrix.round(3))

cov_val = df["batsman_runs"].astype(float).cov(df["is_wicket"].astype(float))
print(f"\nCovariance(batsman_runs, is_wicket): {cov_val:.4f}")

# =====================================================================
# CATEGORICAL / NUMERICAL / VENUE / PLAYER / TEAM / SEASON / PHASE / etc.
# =====================================================================
print("\n--- 5.3 CATEGORICAL ANALYSIS: Teams ---")
print(df["batting_team"].value_counts())

print("\n--- 5.4 VENUE ANALYSIS ---")
venue_matches = match_df.groupby("venue")["match_id"].nunique().sort_values(ascending=False)
print(venue_matches)

print("\n--- 5.5 SEASON ANALYSIS: Runs per season ---")
season_runs = df.groupby("season")["total_runs"].sum()
print(season_runs)

print("\n--- 5.6 PLAYER ANALYSIS: Top 10 run scorers ---")
top_scorers = df.groupby("batter")["batsman_runs"].sum().sort_values(ascending=False).head(10)
print(top_scorers)

print("\n--- 5.7 PLAYER ANALYSIS: Top 10 wicket takers ---")
top_wicket_takers = df[df["is_wicket"] == True].groupby("bowler_clean")["is_wicket"].sum() \
    .sort_values(ascending=False).head(10)
print(top_wicket_takers)

print("\n--- 5.8 TEAM ANALYSIS: Match wins ---")
team_wins = match_df["winner"].value_counts()
print(team_wins)

print("\n--- 5.9 POWERPLAY / MIDDLE / DEATH OVER ANALYSIS ---")
phase_runs = df.groupby("match_phase")["total_runs"].sum()
phase_wickets = df.groupby("match_phase")["is_wicket"].sum()
phase_balls = df.groupby("match_phase")["ball_count"].sum()
phase_rr = (phase_runs / phase_balls * 6).round(2)
print("Runs per phase:\n", phase_runs)
print("Wickets per phase:\n", phase_wickets)
print("Run-rate per phase:\n", phase_rr)

print("\n--- 5.10 BOUNDARY ANALYSIS ---")
print(f"Total fours: {df['four'].sum()}  | Total sixes: {df['six'].sum()}")
print("Top 10 boundary hitters:")
print(df.groupby("batter")["boundary"].sum().sort_values(ascending=False).head(10))

print("\n--- 5.11 PLAYER OF THE MATCH ANALYSIS ---")
print(match_df["player_of_match"].value_counts().head(10))

print("\n--- 5.12 MATCH TYPE ANALYSIS ---")
print(match_df["match_type"].value_counts())

print("\n--- 5.13 TIME-SERIES: Matches per month ---")
matches_month = match_df.copy()
matches_month["month"] = matches_month["date"].dt.to_period("M").astype(str)
print(matches_month.groupby("month")["match_id"].nunique())

# =====================================================================
# STEP 7: 20 PROFESSIONAL VISUALIZATIONS
# =====================================================================
print("\n" + "#" * 70)
print("STEP 7: GENERATING 20 VISUALIZATIONS")
print("#" * 70)

# 1. Top Run Scorers
plt.figure(figsize=(9, 6))
top_scorers.sort_values().plot(kind="barh", color=sns.color_palette(PALETTE, 10))
plt.title("Top 10 Run Scorers - PSL 2020-2021", fontsize=14, fontweight="bold")
plt.xlabel("Total Runs")
plt.ylabel("Batter")
plt.tight_layout()
plt.savefig(IMG + "01_top_run_scorers.png")
plt.close()

# 2. Top Wicket Takers
plt.figure(figsize=(9, 6))
top_wicket_takers.sort_values().plot(kind="barh", color=sns.color_palette("mako", 10))
plt.title("Top 10 Wicket Takers - PSL 2020-2021", fontsize=14, fontweight="bold")
plt.xlabel("Total Wickets")
plt.ylabel("Bowler")
plt.tight_layout()
plt.savefig(IMG + "02_top_wicket_takers.png")
plt.close()

# 3. Team-wise Runs
team_runs = df.groupby("batting_team")["total_runs"].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=team_runs.values, y=team_runs.index, palette=TEAM_PALETTE)
plt.title("Team-wise Total Runs Scored - PSL 2020-2021", fontsize=14, fontweight="bold")
plt.xlabel("Total Runs")
plt.ylabel("Team")
plt.tight_layout()
plt.savefig(IMG + "03_team_wise_runs.png")
plt.close()

# 4. Team-wise Wins
plt.figure(figsize=(10, 6))
sns.barplot(x=team_wins.values, y=team_wins.index, palette=TEAM_PALETTE)
plt.title("Team-wise Match Wins - PSL 2020-2021", fontsize=14, fontweight="bold")
plt.xlabel("Number of Wins")
plt.ylabel("Team")
plt.tight_layout()
plt.savefig(IMG + "04_team_wise_wins.png")
plt.close()

# 5. Top Boundary Hitters
top_boundary = df.groupby("batter")["boundary"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(9, 6))
top_boundary.sort_values().plot(kind="barh", color=sns.color_palette("crest", 10))
plt.title("Top 10 Boundary Hitters (4s + 6s) - PSL 2020-2021", fontsize=14, fontweight="bold")
plt.xlabel("Total Boundaries")
plt.tight_layout()
plt.savefig(IMG + "05_top_boundary_hitters.png")
plt.close()

# 6. Top Six Hitters
top_six = df.groupby("batter")["six"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(9, 6))
top_six.sort_values().plot(kind="barh", color=sns.color_palette("flare", 10))
plt.title("Top 10 Six Hitters - PSL 2020-2021", fontsize=14, fontweight="bold")
plt.xlabel("Total Sixes")
plt.tight_layout()
plt.savefig(IMG + "06_top_six_hitters.png")
plt.close()

# 7. Player of the Match Awards
pom = match_df["player_of_match"].value_counts().head(10)
plt.figure(figsize=(9, 6))
pom.sort_values().plot(kind="barh", color=sns.color_palette("rocket", 10))
plt.title("Top 10 Player of the Match Award Winners", fontsize=14, fontweight="bold")
plt.xlabel("Awards Won")
plt.tight_layout()
plt.savefig(IMG + "07_player_of_match.png")
plt.close()

# 8. Team-wise Wickets (wickets taken by bowling team)
team_wickets = df[df["is_wicket"] == True].groupby("bowling_team")["is_wicket"].sum() \
    .sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=team_wickets.values, y=team_wickets.index, palette="Set2")
plt.title("Team-wise Wickets Taken - PSL 2020-2021", fontsize=14, fontweight="bold")
plt.xlabel("Wickets Taken")
plt.tight_layout()
plt.savefig(IMG + "08_team_wise_wickets.png")
plt.close()

# 9. Venue-wise Matches
plt.figure(figsize=(10, 6))
sns.barplot(x=venue_matches.values, y=venue_matches.index, palette="crest")
plt.title("Number of Matches Played per Venue", fontsize=14, fontweight="bold")
plt.xlabel("Matches Played")
plt.tight_layout()
plt.savefig(IMG + "09_venue_wise_matches.png")
plt.close()

# 10. Venue-wise Average Runs (per innings)
venue_avg_runs = df.groupby(["venue", "match_id", "inning"])["total_runs"] \
    .sum().reset_index().groupby("venue")["total_runs"].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=venue_avg_runs.values, y=venue_avg_runs.index, palette="mako")
plt.title("Average Runs per Innings by Venue", fontsize=14, fontweight="bold")
plt.xlabel("Average Runs per Innings")
plt.tight_layout()
plt.savefig(IMG + "10_venue_avg_runs.png")
plt.close()

# 11. Runs per Over (Line Chart)
runs_per_over = df.groupby("over")["total_runs"].sum() / df["match_id"].nunique() / 2
# average runs scored in that over across all innings
runs_per_over_avg = df.groupby(["match_id", "inning", "over"])["total_runs"].sum() \
    .reset_index().groupby("over")["total_runs"].mean()
plt.figure(figsize=(10, 6))
plt.plot(runs_per_over_avg.index, runs_per_over_avg.values, marker="o", color="#21295C", linewidth=2)
plt.title("Average Runs Scored per Over - PSL 2020-2021", fontsize=14, fontweight="bold")
plt.xlabel("Over Number")
plt.ylabel("Average Runs")
plt.xticks(range(1, 21))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(IMG + "11_runs_per_over.png")
plt.close()

# 12. Wickets per Over (Line Chart)
wickets_per_over_avg = df.groupby(["match_id", "inning", "over"])["is_wicket"].sum() \
    .reset_index().groupby("over")["is_wicket"].mean()
plt.figure(figsize=(10, 6))
plt.plot(wickets_per_over_avg.index, wickets_per_over_avg.values, marker="o", color="#990011", linewidth=2)
plt.title("Average Wickets Lost per Over - PSL 2020-2021", fontsize=14, fontweight="bold")
plt.xlabel("Over Number")
plt.ylabel("Average Wickets")
plt.xticks(range(1, 21))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(IMG + "12_wickets_per_over.png")
plt.close()

# 13. Powerplay vs Middle vs Death Runs
plt.figure(figsize=(8, 6))
phase_order = ["Powerplay", "Middle Overs", "Death Overs"]
sns.barplot(x=phase_order, y=[phase_runs[p] for p in phase_order],
            palette=["#028090", "#00A896", "#02C39A"])
plt.title("Total Runs by Match Phase - PSL 2020-2021", fontsize=14, fontweight="bold")
plt.ylabel("Total Runs")
plt.tight_layout()
plt.savefig(IMG + "13_phase_runs.png")
plt.close()

# 14. Distribution of Runs (Histogram)
plt.figure(figsize=(9, 6))
sns.histplot(df["batsman_runs"], bins=7, kde=False, color="#B85042")
plt.title("Distribution of Runs Scored per Ball", fontsize=14, fontweight="bold")
plt.xlabel("Runs off the Bat")
plt.ylabel("Frequency (balls)")
plt.tight_layout()
plt.savefig(IMG + "14_runs_distribution.png")
plt.close()

# 15. Correlation Heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Key Ball-Level Metrics", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(IMG + "15_correlation_heatmap.png")
plt.close()

# 16. Scatter Plot: Team total runs vs wickets lost per innings
inn_summary = df.groupby(["match_id", "inning", "batting_team"]).agg(
    total_runs=("total_runs", "sum"), wickets=("is_wicket", "sum")
).reset_index()
plt.figure(figsize=(9, 6))
sns.scatterplot(data=inn_summary, x="wickets", y="total_runs", hue="wickets",
                 palette="viridis", legend=False, s=70, alpha=0.75)
plt.title("Innings Total Runs vs Wickets Lost", fontsize=14, fontweight="bold")
plt.xlabel("Wickets Lost in Innings")
plt.ylabel("Total Runs Scored in Innings")
plt.tight_layout()
plt.savefig(IMG + "16_runs_vs_wickets_scatter.png")
plt.close()

# 17. Box Plot: Runs per over distribution by phase
overs_summary = df.groupby(["match_id", "inning", "over"]).agg(
    total_runs=("total_runs", "sum")
).reset_index()
overs_summary["match_phase"] = overs_summary["over"].apply(phase_label := (
    lambda o: "Powerplay" if o <= 6 else ("Middle Overs" if o <= 15 else "Death Overs")
))
plt.figure(figsize=(9, 6))
sns.boxplot(data=overs_summary, x="match_phase", y="total_runs", order=phase_order,
            palette=["#F96167", "#F9E795", "#2F3C7E"])
plt.title("Distribution of Runs per Over by Match Phase", fontsize=14, fontweight="bold")
plt.xlabel("Match Phase")
plt.ylabel("Runs in the Over")
plt.tight_layout()
plt.savefig(IMG + "17_boxplot_runs_by_phase.png")
plt.close()

# 18. Violin Plot: Runs per over by phase
plt.figure(figsize=(9, 6))
sns.violinplot(data=overs_summary, x="match_phase", y="total_runs", order=phase_order,
                palette=["#84B59F", "#69A297", "#50808E"])
plt.title("Violin Plot: Runs per Over Distribution by Phase", fontsize=14, fontweight="bold")
plt.xlabel("Match Phase")
plt.ylabel("Runs in the Over")
plt.tight_layout()
plt.savefig(IMG + "18_violin_runs_by_phase.png")
plt.close()

# 19. Pie Chart: Overall run type share
run_type_counts = df["run_type"].value_counts()
plt.figure(figsize=(8, 8))
colors = sns.color_palette("Set3", len(run_type_counts))
plt.pie(run_type_counts.values, labels=run_type_counts.index, autopct="%1.1f%%",
        startangle=90, colors=colors)
plt.title("Share of Delivery Outcomes (Run Type) - PSL 2020-2021", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(IMG + "19_run_type_pie.png")
plt.close()

# 20. Season-wise Performance Comparison
season_summary = df.groupby("season").agg(
    total_runs=("total_runs", "sum"),
    total_wickets=("is_wicket", "sum"),
    total_sixes=("six", "sum"),
    total_fours=("four", "sum"),
).reset_index()
season_summary_melt = season_summary.melt(id_vars="season",
                                           value_vars=["total_runs", "total_wickets",
                                                       "total_sixes", "total_fours"])
plt.figure(figsize=(10, 6))
sns.barplot(data=season_summary_melt, x="variable", y="value", hue="season", palette="Set1")
plt.title("Season-wise Performance Comparison (2020 vs 2021)", fontsize=14, fontweight="bold")
plt.xlabel("Metric")
plt.ylabel("Total Count")
plt.legend(title="Season")
plt.tight_layout()
plt.savefig(IMG + "20_season_comparison.png")
plt.close()

print("\nAll 20 visualizations saved successfully to ../images/")
print("\nSeason summary table:")
print(season_summary)

print("\n" + "#" * 70)
print("EDA, STATISTICAL ANALYSIS AND VISUALIZATION SCRIPT COMPLETE")
print("#" * 70)
