from pathlib import Path
import re

import pandas as pd
import streamlit as st

# -------------------------------------------------------------
# FILE PATHS
# -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data" / "27_March_Madness_Data"

CURRENT_FILE = DATA_DIR / "Current_Day-Table 1.csv"
AP_FILE = DATA_DIR / "AP_Poll-Table 1.csv"
STATS_FILE = DATA_DIR / "All_Stats-THE_TABLE.csv"


# -------------------------------------------------------------
# LOAD CSV
# -------------------------------------------------------------

@st.cache_data
def load_csv(path, modified):
    df = pd.read_csv(path, encoding="latin1")
    df.columns = df.columns.astype(str).str.strip()
    return df


# -------------------------------------------------------------
# GET CURRENT DATE
# -------------------------------------------------------------

def get_current_date():
    df = load_csv(
        str(CURRENT_FILE),
        CURRENT_FILE.stat().st_mtime_ns
    )

    df.columns = df.columns.str.upper()

    mm = int(float(df.loc[0, "MM"]))
    dd = int(float(df.loc[0, "DD"]))
    yyyy = int(float(df.loc[0, "YYYY"]))

    return pd.Timestamp(year=yyyy, month=mm, day=dd)


# -------------------------------------------------------------
# MARCH METRICS TOP 25 (ALPHABETICAL FOR NOW)
# -------------------------------------------------------------

def get_march_top25():
    df = load_csv(
        str(STATS_FILE),
        STATS_FILE.stat().st_mtime_ns
    )

    team_col = next(
        (c for c in df.columns
         if c.lower() in ["teams", "team", "team_name"]),
        None
    )

    if team_col is None:
        raise ValueError("Teams column not found in All Stats.")

    teams = (
        df[team_col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    teams = sorted(
        [t for t in teams if t],
        key=str.casefold
    )

    return teams[:25]


# -------------------------------------------------------------
# AP POLL
# -------------------------------------------------------------

@st.cache_data
def get_ap_poll(path, modified):
    # The AP export has WEEK1, WEEK2, etc. across row 1
    # and ranking information down column A.

    df = pd.read_csv(
        path,
        header=None,
        dtype=str,
        encoding="latin1",
        keep_default_na=False
    )

    df = df.fillna("")

    headers = df.iloc[0].astype(str).str.strip().str.upper()

    week_columns = [
        (col, label)
        for col, label in enumerate(headers)
        if re.fullmatch(r"WEEK\s*\d+", label)
    ]

    labels = df.iloc[:, 0].astype(str).str.strip().str.upper()

    start_rows = labels[labels == "WEEK_START"].index.tolist()
    end_rows = labels[labels == "WEEK_END"].index.tolist()

    if not start_rows or not end_rows:
        raise ValueError("WEEK_START or WEEK_END missing in AP Poll.")

    def read_week_date(start_row, column):
        values = {}

        for row in range(start_row + 1, min(start_row + 5, len(df))):
            label = labels.iloc[row]

            if label in ("MM", "DD", "YYYY"):
                value = df.iat[row, column].strip()
                values[label] = int(float(value)) if value else None

        if any(values.get(k) is None for k in ("MM", "DD", "YYYY")):
            return None

        return pd.Timestamp(
            year=values["YYYY"],
            month=values["MM"],
            day=values["DD"]
        )

    ranking_rows = {}

    for row, label in enumerate(labels):
        if label.isdigit():
            rank = int(label)

            if 1 <= rank <= 25:
                ranking_rows[rank] = row

    weeks = []

    for col, name in week_columns:
        start = read_week_date(start_rows[0], col)
        end = read_week_date(end_rows[0], col)

        if start is None or end is None:
            continue

        teams = [
            df.iat[ranking_rows[rank], col].strip()
            if rank in ranking_rows else ""
            for rank in range(1, 26)
        ]

        weeks.append({
            "week": name,
            "start": start,
            "end": end,
            "teams": teams
        })

    return weeks


# -------------------------------------------------------------
# MAIN PAGE
# -------------------------------------------------------------

st.title("March Metrics")
st.caption("2026–27 College Basketball Analytics")

try:
    current_date = get_current_date()
    march_top25 = get_march_top25()
    ap_weeks = get_ap_poll(
        str(AP_FILE),
        AP_FILE.stat().st_mtime_ns
    )

except (FileNotFoundError, ValueError, KeyError, IndexError) as e:
    st.error(f"Error loading data: {e}")
    st.stop()


st.write(f"**Current Date: {current_date:%m/%d/%Y}**")


# -------------------------------------------------------------
# FIND CORRECT AP WEEK
# -------------------------------------------------------------

selected_week = None

for week in sorted(
    ap_weeks,
    key=lambda x: x["start"],
    reverse=True
):
    if week["start"] <= current_date <= week["end"]:
        selected_week = week
        break


# -------------------------------------------------------------
# DISPLAY TWO TOP 25 TABLES
# -------------------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("March Metrics Top 25")
    st.caption("Alphabetical until ratings are developed")

    march_df = pd.DataFrame({
        "Rank": range(1, len(march_top25) + 1),
        "Team": march_top25
    })

    st.dataframe(
        march_df,
        hide_index=True,
        use_container_width=True,
        height=920
    )


with right:
    st.subheader("AP Top 25")

    if selected_week is None:
        st.info("No AP Poll week matches the current date.")

    else:
        st.caption(
            f"{selected_week['week']} "
            f"({selected_week['start']:%m/%d/%Y} - "
            f"{selected_week['end']:%m/%d/%Y})"
        )

        ap_df = pd.DataFrame({
            "Rank": range(1, 26),
            "Team": selected_week["teams"]
        })

        st.dataframe(
            ap_df,
            hide_index=True,
            use_container_width=True,
            height=920
        )
