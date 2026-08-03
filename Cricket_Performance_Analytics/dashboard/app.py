"""
=====================================================================
Cricket Performance Analytics Dashboard
PSL 2020-2021 Ball-by-Ball Data
=====================================================================
Run with:
    streamlit run app.py
"""
%pip install matplotlib, pandas, seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path
import os

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------
st.set_page_config(
    page_title="PSL 2020-2021 Analytics Dashboard",
    page_icon="🏏",
    layout="wide",
)

# ---------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------
# Build the data path relative to THIS file's location, not the process's
# current working directory. This makes the app work identically whether
# run locally (streamlit run app.py from inside dashboard/) or deployed
# on Streamlit Cloud (which may launch from the repo root).
APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR.parent / "data" / "cleaned" / "psl_features.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


try:
    df_raw = load_data()
except FileNotFoundError:
    st.error(
        f"Could not find the data file at: `{DATA_PATH}`\n\n"
        "Make sure the `data/cleaned/psl_features.csv` file is committed to "
        "your GitHub repo in the same relative location as `dashboard/app.py`."
    )
    st.stop()

st.title("🏏 Cricket Performance Analytics Dashboard")
st.markdown("### Pakistan Super League (PSL) — 2020 & 2021 Seasons | Ball-by-Ball Analysis")

# ---------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------
st.sidebar.header("Filters")

seasons = st.sidebar.multiselect(
    "Season", options=sorted(df_raw["season"].unique()),
    default=sorted(df_raw["season"].unique())
)

teams = st.sidebar.multiselect(
    "Team (Batting or Bowling)",
    options=sorted(df_raw["batting_team"].unique()),
    default=[]
)

batters = st.sidebar.multiselect(
    "Batter", options=sorted(df_raw["batter"].unique()), default=[]
)

bowlers = st.sidebar.multiselect(
    "Bowler", options=sorted(df_raw["bowler_clean"].unique()), default=[]
)

venues = st.sidebar.multiselect(
    "Venue", options=sorted(df_raw["venue"].unique()), default=[]
)

# ---------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------
df = df_raw.copy()
if seasons:
    df = df[df["season"].isin(seasons)]
if teams:
    df = df[(df["batting_team"].isin(teams)) | (df["bowling_team"].isin(teams))]
if batters:
    df = df[df["batter"].isin(batters)]
if bowlers:
    df = df[df["bowler_clean"].isin(bowlers)]
if venues:
    df = df[df["venue"].isin(venues)]

if df.empty:
    st.warning("No data matches the selected filters. Please adjust your filter selection.")
    st.stop()

# ---------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Matches", df["match_id"].nunique())
col2.metric("Total Runs", int(df["total_runs"].sum()))
col3.metric("Total Wickets", int(df["is_wicket"].sum()))
col4.metric("Total Sixes", int(df["six"].sum()))
col5.metric("Total Fours", int(df["four"].sum()))

col6, col7, col8, col9, col10 = st.columns(5)
balls = df["ball_count"].sum()
run_rate = df["total_runs"].sum() / balls * 6 if balls else 0
dot_pct = df["dot_ball"].mean() * 100
boundary_pct = df["boundary"].mean() * 100

col6.metric("Overall Run Rate", f"{run_rate:.2f}")
col7.metric("Dot Ball %", f"{dot_pct:.1f}%")
col8.metric("Boundary %", f"{boundary_pct:.1f}%")
col9.metric("Unique Batters", df["batter"].nunique())
col10.metric("Unique Bowlers", df["bowler_clean"].nunique())

st.markdown("---")

# ---------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🏏 Batting", "🎯 Bowling", "🏟️ Venue & Phase", "📋 Data Table"]
)

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Team-wise Total Runs")
        team_runs = df.groupby("batting_team")["total_runs"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=team_runs.values, y=team_runs.index, hue=team_runs.index,
                    palette="Set2", legend=False, ax=ax)
        ax.set_xlabel("Total Runs")
        st.pyplot(fig)
    with c2:
        st.subheader("Run Type Share")
        run_type_counts = df["run_type"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(run_type_counts.values, labels=run_type_counts.index, autopct="%1.1f%%",
               colors=sns.color_palette("Set3", len(run_type_counts)))
        st.pyplot(fig)

    st.subheader("Season-wise Comparison")
    season_summary = df.groupby("season").agg(
        total_runs=("total_runs", "sum"),
        total_wickets=("is_wicket", "sum"),
        total_sixes=("six", "sum"),
        total_fours=("four", "sum"),
    ).reset_index()
    st.dataframe(season_summary, use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top 10 Run Scorers")
        top_scorers = df.groupby("batter")["batsman_runs"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=top_scorers.values, y=top_scorers.index, hue=top_scorers.index,
                    palette="viridis", legend=False, ax=ax)
        st.pyplot(fig)
    with c2:
        st.subheader("Top 10 Boundary Hitters")
        top_boundary = df.groupby("batter")["boundary"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=top_boundary.values, y=top_boundary.index, hue=top_boundary.index,
                    palette="crest", legend=False, ax=ax)
        st.pyplot(fig)

    st.subheader("Runs Distribution per Ball")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df["batsman_runs"], bins=7, color="#B85042", ax=ax)
    st.pyplot(fig)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top 10 Wicket Takers")
        top_wickets = df[df["is_wicket"] == True].groupby("bowler_clean")["is_wicket"] \
            .sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=top_wickets.values, y=top_wickets.index, hue=top_wickets.index,
                    palette="mako", legend=False, ax=ax)
        st.pyplot(fig)
    with c2:
        st.subheader("Wickets per Over")
        wpo = df.groupby(["match_id", "inning", "over"])["is_wicket"].sum() \
            .reset_index().groupby("over")["is_wicket"].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(wpo.index, wpo.values, marker="o", color="#990011")
        ax.set_xlabel("Over")
        ax.set_ylabel("Avg Wickets")
        st.pyplot(fig)

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Matches per Venue")
        venue_matches = df.drop_duplicates("match_id")["venue"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=venue_matches.values, y=venue_matches.index, hue=venue_matches.index,
                    palette="crest", legend=False, ax=ax)
        st.pyplot(fig)
    with c2:
        st.subheader("Runs by Match Phase")
        phase_runs = df.groupby("match_phase")["total_runs"].sum()
        order = ["Powerplay", "Middle Overs", "Death Overs"]
        phase_runs = phase_runs.reindex(order)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=phase_runs.index, y=phase_runs.values,
                    hue=phase_runs.index, palette=["#028090", "#00A896", "#02C39A"],
                    legend=False, ax=ax)
        st.pyplot(fig)

with tab5:
    st.subheader("Filtered Ball-by-Ball Data")
    st.dataframe(
        df[["date", "season", "venue", "batting_team", "bowling_team", "over", "ball",
            "batter", "bowler_clean", "batsman_runs", "total_runs", "is_wicket",
            "dismissal_kind"]],
        use_container_width=True,
        height=500,
    )
    st.download_button(
        "Download Filtered Data as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_psl_data.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Cricket Performance Analytics | PSL 2020-2021 | Built with Streamlit, Pandas, Matplotlib & Seaborn")