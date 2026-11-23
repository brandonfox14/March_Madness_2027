"""March Metrics — Core 8 Players.

Place in APP/Pages/6_Core_Players.py.
Loads Players_COMP-Table 1.csv and optionally All_Stats-THE_TABLE.csv
from Data/27_March_Madness_Data. Designed for Apple Numbers CSV exports.
"""
from pathlib import Path
from io import BytesIO
import pandas as pd
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'Data' / '27_March_Madness_Data'
PLAYER_FILE = DATA / 'Players_COMP-Table 1.csv'
ALL_STATS_FILE = DATA / 'All_Stats-THE_TABLE.csv'

RAW_STATS = [
    'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'REB',
    'AST', 'TO', 'STL', 'BLKS', 'OREB', 'DREB', 'PF',
]
CORE_STATS = [
    'CORE_PTS', 'CORE_FGM', 'CORE_FGA', 'CORE_FG3M', 'CORE_FG3A',
    'CORE_FTM', 'CORE_FTA', 'CORE_REB', 'CORE_AST', 'CORE_TO',
    'CORE_STL', 'CORE_BLKS', 'CORE_OREB', 'CORE_DREB',
    'CORE_PF', 'CORE_FOUL_TROUBLE',
]
MIN_COLS = [f'MIN{i}' for i in range(1, 9)]


def read_csv(path_or_stream):
    for encoding in ('utf-8-sig', 'latin1'):
        try:
            if hasattr(path_or_stream, 'seek'):
                path_or_stream.seek(0)
            frame = pd.read_csv(path_or_stream, encoding=encoding, low_memory=False)
            frame.columns = frame.columns.astype(str).str.strip()
            return frame
        except UnicodeDecodeError:
            continue
    raise ValueError('CSV encoding could not be read.')


@st.cache_data(show_spinner=False)
def load_disk(path, modified_ns):
    return read_csv(Path(path))


def normalize_name(value):
    if pd.isna(value):
        return ''
    return ' '.join(str(value).strip().casefold().split())


def build_core(raw):
    required = ['GameID', 'TEAM', 'Player_ID', 'MIN'] + RAW_STATS
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError('Players_COMP missing columns: ' + ', '.join(missing))

    # Numbers' group-header/summary rows must not be counted as player games.
    games = raw.loc[
        raw['GameID'].notna() & raw['TEAM'].notna() &
        raw['Player_ID'].notna()
    , required + (['PLAYER'] if 'PLAYER' in raw.columns else [])].copy()
    for name in ['GameID', 'TEAM', 'Player_ID']:
        games[name] = games[name].astype(str).str.strip()
    games = games.loc[
        games['GameID'].ne('') & games['TEAM'].ne('') &
        games['Player_ID'].ne('')
    ].copy()

    for col in ['MIN'] + RAW_STATS:
        games[col] = pd.to_numeric(games[col], errors='coerce')
    games = games.loc[games['MIN'].notna() & games['MIN'].ge(0)].copy()

    # Some Numbers exports contain repeated full game blocks. Count each
    # player once per team/game, not once per repeated export block.
    unique_key = ['TEAM', 'GameID', 'Player_ID']
    duplicate_count = int(games.duplicated(unique_key).sum())
    conflicting = int(games.loc[games.duplicated(unique_key, keep=False)]
                      .groupby(unique_key)[['MIN'] + RAW_STATS]
                      .nunique(dropna=False).gt(1).any(axis=1).sum())
    games = games.drop_duplicates(unique_key, keep='first')
    for col in RAW_STATS:
        games[col] = games[col].fillna(0)

    # One row per unique team/player; minutes accumulated across ALL completed
    # games. Deterministic ranking handles ties by Player_ID.
    player_totals = (
        games.groupby(['TEAM', 'Player_ID'], as_index=False, sort=False)
        .agg({'MIN': 'sum', **{col: 'sum' for col in RAW_STATS}})
    )
    player_totals = player_totals.sort_values(
        ['TEAM', 'MIN', 'Player_ID'], ascending=[True, False, True], kind='stable'
    )
    player_totals['Core Rank'] = player_totals.groupby('TEAM').cumcount() + 1
    top = player_totals.loc[player_totals['Core Rank'].le(8)].copy()

    # Count games with 4+ personal fouls ONLY from the selected players.
    chosen_keys = top[['TEAM', 'Player_ID']]
    selected_games = games.merge(chosen_keys, on=['TEAM', 'Player_ID'], how='inner')
    selected_games['_FOUL_TROUBLE'] = selected_games['PF'].ge(4).astype(int)

    summarized = top.groupby('TEAM', sort=True)[RAW_STATS].sum()
    summarized.columns = ['CORE_' + c for c in summarized.columns]
    foul = selected_games.groupby('TEAM')['_FOUL_TROUBLE'].sum()
    summarized['CORE_FOUL_TROUBLE'] = foul
    minutes = top.pivot(index='TEAM', columns='Core Rank', values='MIN')
    minutes = minutes.reindex(columns=range(1, 9))
    minutes.columns = MIN_COLS

    final = minutes.join(summarized).reset_index()
    final = final[['TEAM'] + MIN_COLS + CORE_STATS]
    for col in MIN_COLS + CORE_STATS:
        final[col] = final[col].fillna(0).round(6)
        if np.all(np.isclose(final[col], np.round(final[col]))):
            final[col] = final[col].round(0).astype('int64')

    detail = top[['TEAM', 'Core Rank', 'Player_ID', 'MIN'] + RAW_STATS].copy()
    return final, detail, duplicate_count, conflicting, len(games)


def all_stats_order(core, stats):
    if 'Teams' not in stats.columns:
        raise ValueError('All_Stats must contain a Teams column.')
    order = stats[['Teams']].copy()
    order['join_key'] = order['Teams'].map(normalize_name)
    lookup = core.copy()
    lookup['join_key'] = lookup['TEAM'].map(normalize_name)
    lookup = lookup.drop_duplicates('join_key')
    aligned = order.merge(
        lookup[['join_key'] + MIN_COLS + CORE_STATS],
        on='join_key', how='left', sort=False, validate='many_to_one'
    )
    missing = aligned.loc[aligned['CORE_PTS'].isna(), 'Teams'].tolist()
    # Blanks for unmatched teams are deliberate: do NOT silently overwrite
    # missing rosters with zero statistics.
    return aligned, missing


st.title('March Metrics — Core 8 Players')
st.caption('Ranks players by total minutes, then sums all current-season game stats for each team\'s eight most-used players.')

upload = st.file_uploader('Upload Players_COMP CSV (optional if saved in Data folder)', type='csv')
if upload is not None:
    data = read_csv(upload)
elif PLAYER_FILE.is_file():
    data = load_disk(str(PLAYER_FILE), PLAYER_FILE.stat().st_mtime_ns)
else:
    st.error(f'Place {PLAYER_FILE.name} in {DATA}, or upload it above.')
    st.stop()

try:
    core, roster, duplicates, conflicts, n_games = build_core(data)
except (ValueError, TypeError, KeyError) as exc:
    st.error(f'Unable to calculate Core 8: {exc}')
    st.stop()

if core.empty:
    st.warning('No valid player-game rows with GameID, TEAM, Player_ID and minutes.')
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric('Teams with player stats', len(core))
c2.metric('Unique player-game rows', n_games)
c3.metric('Repeated rows ignored', duplicates)
if conflicts:
    st.warning(f'{conflicts} repeated player/game keys contain conflicting values. Their first occurrence was used; inspect the original CSV.')

st.subheader('Full Core 8 summary')
st.caption('TEAM, MIN1–MIN8, and the 16 CORE fields in your requested order. MIN columns are cumulative minutes for each ranked player.')
st.dataframe(core, hide_index=True, use_container_width=True)
st.download_button(
    'Download Core 8 — full summary CSV', core.to_csv(index=False).encode('utf-8-sig'),
    file_name='March_Metrics_Core_8_Full.csv', mime='text/csv'
)

st.subheader('Paste-ready exports')
st.caption('The following 16 columns match your requested CORE field order exactly.')
core_only = core[CORE_STATS]
st.dataframe(core_only, hide_index=True, use_container_width=True)
st.download_button(
    'Download CORE columns only — source team order',
    core_only.to_csv(index=False).encode('utf-8-sig'),
    file_name='March_Metrics_Core_Only_Source_Order.csv', mime='text/csv',
    help='Use only if your destination sheet is sorted to the same TEAM order as the full export.'
)

if ALL_STATS_FILE.is_file():
    try:
        stats = load_disk(str(ALL_STATS_FILE), ALL_STATS_FILE.stat().st_mtime_ns)
        aligned, missing_teams = all_stats_order(core, stats)
        st.markdown('**All_Stats row order — safest for pasting into your existing sheet**')
        st.caption('Same row order as All_Stats-THE_TABLE.csv. Blanks are left for missing teams.')
        if missing_teams:
            st.warning(f'{len(missing_teams)} All_Stats teams did not match player data. They remain blank; check their spellings before pasting.')
            with st.expander('Unmatched All_Stats teams'):
                st.write(missing_teams)
        st.dataframe(aligned[['Teams'] + CORE_STATS], hide_index=True, use_container_width=True)
        st.download_button(
            'Download CORE columns — aligned to All_Stats row order',
            aligned[CORE_STATS].to_csv(index=False).encode('utf-8-sig'),
            file_name='March_Metrics_Core_Only_All_Stats_Order.csv', mime='text/csv'
        )
        st.download_button(
            'Download TEAM + CORE columns — aligned to All_Stats',
            aligned[['Teams'] + CORE_STATS].to_csv(index=False).encode('utf-8-sig'),
            file_name='March_Metrics_Core_With_Teams_All_Stats_Order.csv', mime='text/csv'
        )
    except (ValueError, TypeError, KeyError) as exc:
        st.warning(f'All_Stats alignment unavailable: {exc}')
else:
    st.info('Add All_Stats-THE_TABLE.csv to the Data folder to enable exports in the exact All_Stats row order.')

st.subheader('Inspect which players count')
chosen = st.selectbox('Choose a team', core['TEAM'].tolist())
st.dataframe(roster.loc[roster['TEAM'].eq(chosen)], hide_index=True, use_container_width=True)
st.caption('CORE_FOUL_TROUBLE counts selected-player game appearances with PF >= 4, consistent with your existing sheet.')
