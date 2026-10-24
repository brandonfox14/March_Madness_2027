from pathlib import Path

import pandas as pd
import streamlit as st

st.title("Rankings")

# APP/Pages/2_Rankings.py -> March_Madness_2027
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "Data" / "27_March_Madness_Data" / "All_Stats-THE_TABLE.csv"

if not DATA_FILE.is_file():
    st.error(f"File not found: {DATA_FILE}")
    st.stop()

@st.cache_data
def load_stats(file_path: str, last_modified: float) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="latin1")
    df.columns = df.columns.astype(str).str.strip()
    return df

stats = load_stats(str(DATA_FILE), DATA_FILE.stat().st_mtime)

# Last year's stats used the column "Teams".
team_column = next(
    (col for col in stats.columns if col.lower() in {"teams", "team", "team_name"}),
    None,
)

if team_column is None:
    st.error("No team-name column found. Expected 'Teams', 'Team', or 'Team_Name'.")
    st.write("Available columns:", stats.columns.tolist())
    st.stop()

teams = (
    stats[team_column]
    .dropna()
    .astype(str)
    .str.strip()
)
teams = sorted((name for name in teams.unique() if name), key=str.casefold)

for team in teams:
    st.write(team)
