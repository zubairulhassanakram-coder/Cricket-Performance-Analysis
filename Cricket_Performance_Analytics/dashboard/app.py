"""
=====================================================================
Cricket Performance Analytics Dashboard
PSL 2020-2021 Ball-by-Ball Data
=====================================================================
Run with:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
# Project paths
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "psl_features.csv"

# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"Dataset not found!\n\nExpected location:\n{DATA_PATH}")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

df_raw = load_data()