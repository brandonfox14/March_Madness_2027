def read_ap_weeks(path, modified):
    """Parse Numbers AP export: WEEK1..WEEK20 across columns, labels down column A.

    Expects WEEK_START / MM / DD / YYYY / WEEK_END / MM / DD / YYYY,
    followed by ranking rows with 1..25 in column A.
    """
    raw = read_csv(Path(path), header=None, dtype=str, keep_default_na=False)
    raw = raw.fillna("")
    names = raw.iloc[0].astype(str).str.strip().str.upper().tolist()
    week_columns = [(idx, name) for idx, name in enumerate(names) if re.fullmatch(r"WEEK\s*\d+", name)]
    if not week_columns:
        raise ValueError("AP poll file needs WEEK1, WEEK2, ... column headers.")

    labels = raw.iloc[:, 0].astype(str).str.strip().str.upper()
    start_idx = labels[labels == "WEEK_START"].index.tolist()
    end_idx = labels[labels == "WEEK_END"].index.tolist()
    if not start_idx or not end_idx:
        raise ValueError("AP poll file needs WEEK_START and WEEK_END in column A.")
    start_row, end_row = start_idx[0], end_idx[0]

    def date_below(section_row, col):
        vals = {}
        # Look only between this section marker and the next marker, or at most 5 lines.
        limit = min(section_row + 6, len(raw))
        for row in range(section_row + 1, limit):
            tag = labels.iloc[row]
            if tag in {"WEEK_START", "WEEK_END"}:
                break
            if tag in ("MM", "DD", "YYYY"):
                vals[tag] = value_to_int(raw.iat[row, col])
        if any(vals.get(k) is None for k in ("MM", "DD", "YYYY")):
            return None
        try:
            return pd.Timestamp(year=vals["YYYY"], month=vals["MM"], day=vals["DD"])
        except ValueError:
            return None

    rank_rows = {}
    for row, label in enumerate(labels):
        rank = value_to_int(label)
        if rank is not None and 1 <= rank <= 25:
            rank_rows[rank] = row

    weeks = []
    for col, week_name in week_columns:
        begin = date_below(start_row, col)
        finish = date_below(end_row, col)
        teams = []
        for rank in range(1, 26):
            row = rank_rows.get(rank)
            name = str(raw.iat[row, col]).strip() if row is not None else ""
            teams.append(name)
        if begin is not None and finish is not None and any(teams):
            weeks.append({"week": week_name, "start": begin, "end": finish, "teams": teams})
    return sorted(weeks, key=lambda w: w["start"])


def choose_ap_week(weeks, day):
    # Treat WEEK_END as the transition into the next week, e.g. Nov 8.
    matches = [w for w in weeks if w["start"] <= day < w["end"]]
    if matches:
        return matches[-1]
    # Last end date can be included when no later week exists.
    matches = [w for w in weeks if w["start"] <= day <= w["end"]]
    if matches:
        return matches[-1]
    return None


st.title("March Metrics")
st.caption("2026–27 College Basketball")

try:
    date = get_current_day(str(CURRENT_DAY_FILE), CURRENT_DAY_FILE.stat().st_mtime_ns)
    teams = get_all_teams(str(STATS_FILE), STATS_FILE.stat().st_mtime_ns)
    weeks = read_ap_weeks(str(AP_FILE), AP_FILE.stat().st_mtime_ns)
except FileNotFoundError as e:
    st.error(f"Missing data file: {e.filename}")
    st.stop()
except (ValueError, KeyError, IndexError, pd.errors.ParserError) as e:
    st.error(f"Could not read home page data: {e}")
    st.stop()

active_week = choose_ap_week(weeks, date)
st.write(f"**Databook date:** {date.strftime('%m/%d/%Y')}")
left, right = st.columns(2, gap="medium")

with left:
    st.subheader("March Metrics Top 25")
    st.caption("TEMPORARY: first 25 teams alphabetically; NOT a rating ranking.")
    march_df = pd.DataFrame({
        "Position": range(1, min(25, len(teams)) + 1),
        "Team": teams[:25],
    })
    st.dataframe(march_df, hide_index=True, use_container_width=True, height=920)

with right:
    st.subheader("AP Top 25")
    if active_week is None:
        st.info("No AP poll week matches the current databook date.")
    else:
        st.caption(f"{active_week['week']} · {active_week['start']:%m/%d/%Y} – {active_week['end']:%m/%d/%Y}")
        ap_df = pd.DataFrame({
            "Rank": range(1, 26),
            "Team": [name if name else "—" for name in active_week["teams"]],
        })
        st.dataframe(ap_df, hide_index=True, use_container_width=True, height=920)
