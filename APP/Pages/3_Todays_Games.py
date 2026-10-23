
def require_columns(frame: pd.DataFrame, required: list[str], filename: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        st.error(f"{filename} is missing columns: {', '.join(missing)}")
        st.stop()


def numeric_date_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for col in ("MM", "DD", "YYYY"):
        frame[col] = pd.to_numeric(frame[col].astype("string").str.strip(), errors="coerce")
    return frame


# -------------------------------------------------------------
# LOAD THE CURRENT DATE FROM APPLE NUMBERS EXPORT
# -------------------------------------------------------------
try:
    current_day = read_csv(str(CURRENT_DAY_FILE))
    schedule = read_csv(str(SCHEDULE_FILE))
except FileNotFoundError as exc:
    st.error(f"Could not find the CSV: {exc.filename}")
    st.stop()
except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
    st.error(f"There is a problem reading the CSV files: {exc}")
    st.stop()

require_columns(current_day, ["MM", "DD", "YYYY"], CURRENT_DAY_FILE.name)
require_columns(
    schedule,
    ["MM", "DD", "YYYY", "GameID", "Team1", "Team2"],
    SCHEDULE_FILE.name,
)

current_day = numeric_date_columns(current_day)
schedule = numeric_date_columns(schedule)
valid_current_days = current_day.dropna(subset=["MM", "DD", "YYYY"])
if valid_current_days.empty:
    st.error("Current_Day CSV has no usable MM / DD / YYYY row.")
    st.stop()

# Use the first valid date row (CURRENT may be blank in the export).
date_row = valid_current_days.iloc[0]
try:
    selected_date = date(int(date_row["YYYY"]), int(date_row["MM"]), int(date_row["DD"]))
except ValueError as exc:
    st.error(f"Current_Day CSV contains an invalid date: {exc}")
    st.stop()

# -------------------------------------------------------------
# FIND ALL GAMES WHOSE MONTH, DAY, AND YEAR MATCH
# -------------------------------------------------------------
daily_games = schedule.loc[
    (schedule["MM"] == selected_date.month)
    & (schedule["DD"] == selected_date.day)
    & (schedule["YYYY"] == selected_date.year)
].copy()

# A schedule that lists each game for both teams can contain repeated GameIDs.
# Only remove exact GameID duplicates; don't discard unique matchups.
daily_games = daily_games.drop_duplicates(subset=["GameID"], keep="first")

# -------------------------------------------------------------
# ### FUTURE: ORDER TODAY'S GAMES BY COMBINED TEAM RATINGS ###
# -------------------------------------------------------------
# ### Load all-team ratings when the new rating system is finalized.
# ### Match Team1Acronym and Team2Acronym to their team ratings.
# ### daily_games["Combined_Rating"] = (
# ###     daily_games["Team1_Rating"] + daily_games["Team2_Rating"]
# ### )
# ### daily_games = daily_games.sort_values(
# ###     "Combined_Rating", ascending=False
# ### )
# ### Until then, keep the original CSV ordering.

# -------------------------------------------------------------
# DISPLAY
# -------------------------------------------------------------
st.title("Today's Games")
st.caption(f"Schedule date from Current_Day: {selected_date:%m/%d/%Y}")
st.metric("Games scheduled", len(daily_games))

if daily_games.empty:
    st.info("No games match the MM, DD, and YYYY values in Current_Day.")
    st.stop()

# Include available columns, in a consistent order.
columns_to_show = [
    "GameID", "Team1", "Team1Acronym", "Team2", "Team2Acronym",
    "Location", "Home_Team", "ML1", "ML2",
]
columns_to_show = [column for column in columns_to_show if column in daily_games.columns]

display_games = daily_games[columns_to_show].copy().fillna("")
st.dataframe(display_games, hide_index=True, use_container_width=True)

# -------------------------------------------------------------
# OPTIONAL: MATCHUP DETAILS
# -------------------------------------------------------------
st.subheader("Matchups")
for _, game in daily_games.iterrows():
    team1 = str(game.get("Team1", ""))
    team2 = str(game.get("Team2", ""))
    code1 = str(game.get("Team1Acronym", "")) if pd.notna(game.get("Team1Acronym")) else ""
    code2 = str(game.get("Team2Acronym", "")) if pd.notna(game.get("Team2Acronym")) else ""
    with st.expander(f"{team1} ({code1}) vs. {team2} ({code2})"):
        st.write(f"**Game ID:** {game.get('GameID', '')}")
        st.write(f"**Location:** {game.get('Location', '')}")
        st.write(f"**Home Team:** {game.get('Home_Team', '')}")
        ml1 = game.get("ML1", "")
        ml2 = game.get("ML2", "")
        st.write(f"**Moneylines:** {team1}: {ml1 if pd.notna(ml1) and str(ml1).strip() else 'Not available'} | {team2}: {ml2 if pd.notna(ml2) and str(ml2).strip() else 'Not available'}")
