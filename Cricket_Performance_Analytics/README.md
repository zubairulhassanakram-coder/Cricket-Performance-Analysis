# 🏏 Cricket Performance Analytics using Pakistan Super League (PSL) 2020–2021 Ball-by-Ball Dataset

A complete university Data Analytics capstone project analyzing 15,164 ball-by-ball deliveries from the PSL 2020 and 2021 seasons to uncover batting, bowling, team, venue, and match-phase performance patterns.

---

## 📌 Project Overview

This project applies a full data analytics pipeline — data cleaning, feature engineering, exploratory data analysis, statistical analysis, visualization, and dashboarding — to PSL ball-by-ball data in order to generate actionable cricket performance insights for teams, analysts, broadcasters, and fans.

## 🎯 Objectives

- Clean and structure raw ball-by-ball data into an analysis-ready format.
- Engineer match-phase and outcome-based features (powerplay/middle/death overs, boundaries, dot balls, wickets).
- Perform detailed exploratory and statistical analysis across players, teams, venues, and seasons.
- Produce 20 professional visualizations answering specific business questions.
- Build an interactive Streamlit dashboard for self-service exploration.
- Deliver a full academic report and presentation summarizing findings.

## 🗂️ Repository Structure

```
Cricket_Performance_Analytics/
│
├── data/
│   ├── raw/                 # Original PSL 2020-2021 ball-by-ball CSV
│   └── cleaned/              # Cleaned & feature-engineered datasets
│
├── notebooks/
│   └── Cricket_Performance_Analytics.ipynb   # End-to-end analysis notebook
│
├── src/
│   ├── 01_data_cleaning.py
│   ├── 02_feature_engineering.py
│   └── 03_eda_statistics_visuals.py
│
├── dashboard/
│   └── app.py                # Streamlit interactive dashboard
│
├── reports/
│   ├── Cricket_Performance_Analytics_Report.docx
│   └── analysis_output_log.txt
│
├── images/
│   └── 01_top_run_scorers.png ... 20_season_comparison.png
│
├── presentation/
│   └── Cricket_Performance_Analytics_Presentation.pptx
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## 📊 Dataset

- **Source:** Ball-by-ball delivery records from the Pakistan Super League, filtered to the **2020 and 2021 seasons only**.
- **Size:** 15,164 rows × 26 original columns (27+ after cleaning, 42 after feature engineering).
- **Grain:** One row = one ball delivered in a PSL match.
- **Key fields:** match_id, date, season, venue, inning, batting_team, bowling_team, over, ball, batter, bowler, batsman_runs, total_runs, extras_type, is_wicket, dismissal_kind, winner, player_of_match, etc.

## ⚙️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the analysis pipeline
```bash
cd src
python 01_data_cleaning.py
python 02_feature_engineering.py
python 03_eda_statistics_visuals.py
```
This regenerates the cleaned datasets in `data/cleaned/` and all 20 charts in `images/`.

### 3. Launch the interactive dashboard
```bash
cd dashboard
streamlit run app.py
```

### 4. Explore the notebook
Open `notebooks/Cricket_Performance_Analytics.ipynb` in Jupyter to walk through the full pipeline interactively.

## 🔑 Key Features Engineered

| Feature | Description |
|---|---|
| `dot_ball` | 1 if no run and no extra scored off the delivery |
| `four` / `six` / `boundary` | Flags for boundary events |
| `wicket` | 1 if a wicket fell on that ball |
| `powerplay` / `middle_over` / `death_over` | Match-phase flags based on over number |
| `run_type` | Categorical outcome of each ball (Dot, Running Runs, Four, Six, Extra, Wicket) |

## 📈 Highlights of Findings

- **Babar Azam** was the leading run-scorer across the 2020–2021 seasons.
- **Shaheen Shah Afridi** led all bowlers in wickets taken.
- **Multan Sultans** recorded the most match wins across the two seasons.
- Death overs (17–20) produced the highest run rate (~9.8 runs/over) but also the highest wicket concentration, confirming the boom-or-bust nature of the finishing phase.
- Roughly 1 in 3 deliveries (≈34%) was a dot ball, highlighting the value bowlers place on building pressure.

See the full report in `reports/` for all 20+ insights and recommendations.

## 🛠️ Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · SciPy · Streamlit

## 👤 Author

University Data Analytics Capstone Project — 2026

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
