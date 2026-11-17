"""March Metrics home page. Save to APP/Pages/1_Main.py.

APP/App.py already calls st.set_page_config and manages navigation.
"""
from pathlib import Path
import re
import sys

import pandas as pd
import streamlit as st
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from _v3_rankings import rankings_for_day

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data" / "27_March_Madness_Data"
CURRENT_FILE = DATA_DIR / "Current_Day-Table 1.csv"
AP_FILE = DATA_DIR / "AP_Poll-Table 1.csv"
STATS_FILE = DATA_DIR / "All_Stats-THE_TABLE.csv"


def read_csv(path, **kwargs):
    if not path.is_file():
        st.error(f"File not found: {path}")
        st.stop()
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **kwargs)


def number(value):
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def current_date(path, changed):
    df = read_csv(Path(path), dtype=str)
    df.columns = df.columns.astype(str).str.strip().str.upper()
    if df.empty or not {"MM", "DD", "YYYY"}.issubset(df.columns):
        raise ValueError("Current_Day must contain headers MM, DD, YYYY and a date in row 2.")
    row = df.iloc[0]
    return pd.Timestamp(year=number(row["YYYY"]), month=number(row["MM"]), day=number(row["DD"]))


@st.cache_data(show_spinner=False)
def alphabetical_teams(path, changed):
    df = read_csv(Path(path), dtype=str)
    df.columns = df.columns.astype(str).str.strip()
    col = next((c for c in df.columns if c.lower() in {"teams", "team", "team_name"}), None)
    if col is None:
        raise ValueError("All_Stats-THE_TABLE.csv needs a Teams column.")
    names = df[col].dropna().astype(str).str.strip()
    return sorted(set(n for n in names if n), key=str.casefold)


@st.cache_data(show_spinner=False)
def ap_weeks(path, changed):
    # Apple Numbers export: first row WEEK1..WEEK20; first column contains row labels.
    raw = read_csv(Path(path), header=None, dtype=str, keep_default_na=False).fillna("")
    headers = raw.iloc[0].astype(str).str.strip().str.upper()
    week_cols = [(col, title) for col, title in enumerate(headers) if re.fullmatch(r"WEEK\s*\d+", title)]
    if not week_cols:
        raise ValueError("AP poll needs WEEK1, WEEK2, ... in its first row.")

    labels = raw.iloc[:, 0].astype(str).str.strip().str.upper()
    starts = labels[labels == "WEEK_START"].index.to_list()
    ends = labels[labels == "WEEK_END"].index.to_list()
    if not starts or not ends or ends[0] <= starts[0]:
        raise ValueError("AP poll needs WEEK_START followed by WEEK_END in column A.")

    def section_date(section, next_section, col):
        fields = {}
        for r in range(section + 1, next_section):
            label = labels.iloc[r]
            if label in {"MM", "DD", "YYYY"}:
                fields[label] = number(raw.iat[r, col])
        if not all(fields.get(key) is not None for key in ("MM", "DD", "YYYY")):
            return None
        try:
            return pd.Timestamp(year=fields["YYYY"], month=fields["MM"], day=fields["DD"])
        except ValueError:
            return None

    first_rank_row = ends[0] + 1
    ranks = {}
    for r in range(first_rank_row, len(raw)):
        value = number(labels.iloc[r])
        if value is not None and 1 <= value <= 25 and value not in ranks:
            ranks[value] = r

    weeks = []
    for col, label in week_cols:
        start = section_date(starts[0], ends[0], col)
        end = section_date(ends[0], min(first_rank_row + 3, len(raw)), col)
        if start is None or end is None:
            continue
        names = [str(raw.iat[ranks[rank], col]).strip() if rank in ranks else "" for rank in range(1, 26)]
        weeks.append({"label": label, "start": start, "end": end, "teams": names})
    return sorted(weeks, key=lambda w: w["start"])


def week_for_date(weeks, date):
    # End dates are inclusive unless that date also starts another week.
    candidates = [w for w in weeks if w["start"] <= date <= w["end"]]
    return max(candidates, key=lambda w: w["start"]) if candidates else None


st.title("March Metrics")
st.caption("2026–27 College Basketball")

try:
    day = current_date(str(CURRENT_FILE), CURRENT_FILE.stat().st_mtime_ns)
    stats_df = read_csv(STATS_FILE)
    stats_df.columns = stats_df.columns.astype(str).str.strip()
    weeks = ap_weeks(str(AP_FILE), AP_FILE.stat().st_mtime_ns)
except FileNotFoundError as exc:
    st.error(f"Missing required CSV file: {exc.filename}")
    st.stop()
except (ValueError, TypeError, IndexError, KeyError, pd.errors.ParserError) as exc:
    st.error(f"Could not read rankings or poll data: {exc}")
    st.stop()

# Rankings and Main share one neutral-simulation function and the same cache.
# The catalog is used solely for the fallback alphabetical list before V3 activates.
from _v3_shared import team_catalog
try:
    with st.spinner("Calculating March Metrics Top 25..."):
        ranking_table, ranking_note = rankings_for_day(day, team_catalog(stats_df))
except Exception as exc:
    st.error(f"Unable to calculate the V3 Top 25: {exc}")
    st.stop()

st.write(f"**Databook date:** {day:%m/%d/%Y}")
selected_week = week_for_date(weeks, day)
left, right = st.columns(2, gap="medium")

with left:
    st.subheader("March Metrics Top 25")
    if ranking_note:
        st.caption(ranking_note)
    else:
        st.caption("V3 neutral-court round robin · projected wins against all teams")
    st.dataframe(
        ranking_table[["Rank", "Team"]].head(25),
        hide_index=True, use_container_width=True, height=920,
    )

with right:
    st.subheader("AP Top 25")
    if selected_week is None:
        st.info("No AP poll week contains the current databook date.")
    else:
        st.caption(f"{selected_week['label']} · {selected_week['start']:%m/%d/%Y}–{selected_week['end']:%m/%d/%Y}")
        st.dataframe(
            pd.DataFrame({"Rank": range(1, 26), "Team": [t or "—" for t in selected_week["teams"]]}),
            hide_index=True, use_container_width=True, height=920,
        )
