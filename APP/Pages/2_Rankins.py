"""March Metrics 2027 rankings: neutral-court round robin, ML after Dec 1.
Save as APP/Pages/2_Rankings.py. APP/App.py manages st.set_page_config.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'Data' / '27_March_Madness_Data'
STATS_FILE = DATA / 'All_Stats-THE_TABLE.csv'
SEASON_FILE = DATA / 'COMP_STAT-THE_TABLE.csv'
DATE_FILE = DATA / 'Current_Day-Table 1.csv'
START = pd.Timestamp('2026-12-02')  # later than December 1
MIN_TRAIN_ROWS = 40

# Maps pregame game-log metric suffix to current All_Stats metric.
# The All_Stats counting columns are season totals and are converted to per-game.
METRICS = {
    'GP': None,
    'WIN_PCT': 'Win_Perc',
    'PTS_FOR': 'PTS', 'PTS_ALLOW': 'OPP_PTS',
    '1H_FOR': '1H', '1H_ALLOW': 'OPP_1H',
    '2H_FOR': '2H', '2H_ALLOW': 'OPP_2H',
    'FG_PCT_FOR': 'FGPerc', 'FG_PCT_ALLOW': 'OPP_FGPerc',
    '2P_PCT_FOR': 'FG2Perc', '2P_PCT_ALLOW': 'OPP_FG2Perc',
    '3P_PCT_FOR': 'FG3Perc', '3P_PCT_ALLOW': 'OPP_FG3Perc',
    'FT_PCT_FOR': 'FTPerc',
    'EFG_FOR': 'EFG_PCT', 'EFG_ALLOW': 'OPP_EFG_PCT',
    'TS_FOR': 'TS_PCT', 'TS_ALLOW': 'OPP_TS_PCT',
    'OFF_EFF': 'OFF_EFF', 'DEF_EFF': 'DEF_EFF', 'NET_EFF': 'NET_EFF',
    'OREB_RATE': 'OREB_RATE', 'DREB_RATE': 'DREB_RATE',
    'REB_SHARE': 'REB_SHARE', 'TURN_RATE': 'TURN_RATE',
    'FORCED_TURN_RATE': 'FORCED_TURN_RATE',
    'EXTRA_CREATED': 'EXTRA_OPP_CREATED',
    'EXTRA_ALLOWED': 'EXTRA_OPP_ALLOWED',
    'NET_EXTRA_OPP': 'NET_EXTRA_OPP',
    'NET_EXTRA_RATE': 'NET_EXTRA_OPP_RATE',
    'AST_TO': 'AST_TO_RATIO', 'AST_TO_ALLOW': 'OPP_AST_TO_RATIO',
    'AST_FGM_RATE': 'AST_FGM_RATE',
    'AST_FGM_RATE_ALLOW': 'OPP_AST_FGM_RATE',
}
PER_GAME = {'PTS_FOR', 'PTS_ALLOW', '1H_FOR', '1H_ALLOW', '2H_FOR', '2H_ALLOW'}
FEATURES = ['TEAM_PRE_' + name for name in METRICS] + ['OPP_PRE_' + name for name in METRICS] + ['HAN_NEUTRAL']


def signature(path):
    return (str(path), path.stat().st_mtime_ns if path.is_file() else None)


@st.cache_data(show_spinner=False)
def read_data(path, stamp):
    if stamp is None:
        return None
    try:
        df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin1', low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    return df


def as_number(series):
    return pd.to_numeric(series, errors='coerce')


def numeric_column(df, name):
    return as_number(df[name]) if name in df.columns else pd.Series(np.nan, index=df.index)


def game_dates(df):
    if {'MM', 'DD', 'YYYY'}.issubset(df.columns):
        return pd.to_datetime(pd.DataFrame({
            'year': numeric_column(df, 'YYYY'),
            'month': numeric_column(df, 'MM'),
            'day': numeric_column(df, 'DD'),
        }), errors='coerce')
    if 'GAME_DATE' in df.columns:
        return pd.to_datetime(df['GAME_DATE'], errors='coerce')
    return pd.Series(pd.NaT, index=df.index)


def build_profiles(stats):
    required = {'Teams', 'Acronym'}
    if not required.issubset(stats.columns):
        raise ValueError('All_Stats must contain Teams and Acronym columns.')
    df = stats.copy()
    df['Teams'] = df['Teams'].fillna('').astype(str).str.strip()
    df['Acronym'] = df['Acronym'].fillna('').astype(str).str.strip().str.upper()
    df = df[(df['Teams'] != '') & (df['Acronym'] != '')].drop_duplicates('Acronym').reset_index(drop=True)
    if df.empty:
        raise ValueError('No valid teams found in All_Stats.')
    gp = numeric_column(df, 'Win').fillna(0) + numeric_column(df, 'Loss').fillna(0)
    profiles = pd.DataFrame(index=df.index)
    for metric, col in METRICS.items():
        if metric == 'GP':
            value = gp.astype(float)
        else:
            value = numeric_column(df, col)
            if metric in PER_GAME:
                value = value.div(gp.replace(0, np.nan))
        profiles[metric] = value.astype(float)
    # Avoid treating teams with no games as having genuine 0.0% shooting/efficiency.
    profiles.loc[gp.eq(0), :] = np.nan
    profiles.loc[gp.eq(0), 'GP'] = 0.0
    return df, profiles


def build_training(season, day):
    required = {'PTS', 'OPP_PTS', 'TEAM_PRE_GP', 'OPP_PRE_GP'}
    if not required.issubset(season.columns):
        raise ValueError('COMP_STAT missing: ' + ', '.join(sorted(required - set(season.columns))))
    df = season.loc[game_dates(season).lt(day)].copy()
    df = df.loc[numeric_column(df, 'PTS').notna() & numeric_column(df, 'OPP_PTS').notna()]
    if df.empty:
        return None, None, 0
    x = pd.DataFrame(index=df.index)
    for feature in FEATURES:
        if feature == 'HAN_NEUTRAL':
            loc = df['HAN'].fillna('').astype(str).str.strip().str.upper() if 'HAN' in df else pd.Series('', index=df.index)
            x[feature] = loc.map({'HOME': 1., 'AWAY': -1., 'NEUTRAL': 0.})
        else:
            x[feature] = numeric_column(df, feature)
    x = x.replace([np.inf, -np.inf], np.nan)
    y = pd.DataFrame({'PTS': numeric_column(df, 'PTS'), 'OPP_PTS': numeric_column(df, 'OPP_PTS')}, index=df.index)
    return x, y, len(df)


@st.cache_resource(show_spinner=False)
def fit_models(date_key, season_signature):
    # Training must be cached by both the selected date and updated file signature.
    season = read_data(*season_signature)
    x, y, n = build_training(season, pd.Timestamp(date_key))
    if x is None or n < MIN_TRAIN_ROWS:
        return None, f'Need at least {MIN_TRAIN_ROWS} completed team-game rows (found {n}).'
    usable = [c for c in x.columns if x[c].notna().sum() >= 15 and x[c].nunique(dropna=True) > 1]
    if len(usable) < 4:
        return None, 'Not enough usable pregame metrics in COMP_STAT yet.'
    x = x[usable]
    factories = {
        'Ridge': lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=25)),
        'Extra Trees': lambda: make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(
            n_estimators=35, max_depth=10, min_samples_leaf=4, n_jobs=-1, random_state=42)),
        'Histogram Boosting': lambda: make_pipeline(SimpleImputer(strategy='median'), HistGradientBoostingRegressor(
            max_iter=45, max_leaf_nodes=12, min_samples_leaf=15, learning_rate=0.07, random_state=42)),
    }
    fitted = {}
    for label, factory in factories.items():
        fitted[label] = {}
        for outcome in ['PTS', 'OPP_PTS']:
            model = factory()
            model.fit(x, y[outcome])
            fitted[label][outcome] = model
    return {'models': fitted, 'features': usable, 'rows': n}, None


def predict_round_robin(teams, profiles, engine):
    count = len(teams)
    # Every directed matchup, including self-matches (which are discarded).
    team_idx = np.repeat(np.arange(count), count)
    opp_idx = np.tile(np.arange(count), count)
    base = {}
    for metric in METRICS:
        values = profiles[metric].to_numpy(dtype=float)
        base['TEAM_PRE_' + metric] = values[team_idx]
        base['OPP_PRE_' + metric] = values[opp_idx]
    base['HAN_NEUTRAL'] = np.zeros(count * count, dtype=float)
    x = pd.DataFrame({k: base[k] for k in engine['features']})
    # Predict each team's points and points conceded. Then reconcile the two
    # independent team perspectives so every matchup has one shared score.
    scored, allowed = [], []
    for bundle in engine['models'].values():
        scored.append(bundle['PTS'].predict(x).reshape(count, count))
        allowed.append(bundle['OPP_PTS'].predict(x).reshape(count, count))
    scored = np.mean(scored, axis=0)
    allowed = np.mean(allowed, axis=0)
    predicted = np.clip((scored + allowed.T) / 2, 0, None)
    np.fill_diagonal(predicted, np.nan)
    margin = predicted - predicted.T
    wins = (margin > 1e-8).sum(axis=1).astype(float)
    ties = (np.abs(margin) <= 1e-8) & ~np.eye(count, dtype=bool)
    wins += 0.5 * ties.sum(axis=1)
    allowed_avg = np.nanmean(predicted.T, axis=1)
    return predicted, wins, allowed_avg, margin


def order_by_tiebreaks(names, wins, points, margin):
    # Stage 1: predicted neutral-court wins against the full field.
    # Stage 2: tied teams' head-to-head projected results.
    # Stage 3: for circular ties, fewest expected points conceded within tie group.
    # Final deterministic fallback: team name A–Z.
    group_ids = {}
    for idx, wins_value in enumerate(wins):
        group_ids.setdefault(round(float(wins_value), 6), []).append(idx)
    ordered = []
    for score in sorted(group_ids, reverse=True):
        tied = group_ids[score]
        if len(tied) == 1:
            ordered += tied
            continue
        sub = margin[np.ix_(tied, tied)]
        h2h = (sub > 1e-8).sum(axis=1) + 0.5 * ((np.abs(sub) <= 1e-8).sum(axis=1) - 1)
        # Head-to-head tiebreak first, then lowest points allowed in tied games.
        allowed_in_tie = np.nanmean(points[np.ix_(tied, tied)].T, axis=1)
        ordered += sorted(tied, key=lambda i: (
            -float(h2h[tied.index(i)]), float(allowed_in_tie[tied.index(i)]), names[i].casefold()
        ))
    return ordered


# ============================================================
# PAGE
# ============================================================
st.title('Rankings')

for required in (STATS_FILE, DATE_FILE):
    if not required.is_file():
        st.error(f'File not found: {required}')
        st.stop()

try:
    day_table = read_data(*signature(DATE_FILE))
    if day_table.empty or not {'MM', 'DD', 'YYYY'}.issubset(day_table.columns):
        raise ValueError('Current_Day needs MM, DD, YYYY in row 1 and values in row 2.')
    today = pd.Timestamp(year=int(day_table.iloc[0]['YYYY']),
                         month=int(day_table.iloc[0]['MM']),
                         day=int(day_table.iloc[0]['DD']))
    stats = read_data(*signature(STATS_FILE))
    teams, profiles = build_profiles(stats)
except (ValueError, KeyError, IndexError, TypeError) as exc:
    st.error(f'Data issue: {exc}')
    st.stop()

st.caption(f'Databook date: {today:%m/%d/%Y} · {len(teams)} teams')
rank_table = None
if today >= START:
    if not SEASON_FILE.is_file():
        st.warning(f'ML not available: {SEASON_FILE.name} is missing. Alphabetical list shown.')
    else:
        try:
            with st.spinner('Training 3 fast regressors and forecasting neutral-site matchups...'):
                engine, reason = fit_models(today.isoformat(), signature(SEASON_FILE))
                if engine is not None:
                    points, wins, allowed, margin = predict_round_robin(teams, profiles, engine)
                    order = order_by_tiebreaks(teams['Teams'].tolist(), wins, points, margin)
                    rank_table = pd.DataFrame({
                        'Rank': np.arange(1, len(order) + 1),
                        'Team': teams.loc[order, 'Teams'].to_numpy(),
                        'Projected Wins': wins[order],
                        'Projected Losses': (len(teams) - 1) - wins[order],
                        'Points Allowed / Game': allowed[order],
                    })
                    st.success(f'ML rankings active · {engine["rows"]} current-season team-game training rows · neutral-site round robin')
                else:
                    st.info('ML not active yet: ' + reason)
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f'Unable to run rankings model: {exc}')
            st.stop()
else:
    st.info('Before December 2, 2026: alphabetical ordering only. No simulated ranking is claimed.')

if rank_table is None:
    names = sorted(teams['Teams'].tolist(), key=str.casefold)
    rank_table = pd.DataFrame({'Rank': range(1, len(names) + 1), 'Team': names})

st.subheader('Top 25')
# Row-major 5 x 5 (1–5 first row, 6–10 second row, etc.)
for start in range(0, min(25, len(rank_table)), 5):
    cols = st.columns(5, gap='small')
    for offset, col in enumerate(cols):
        pos = start + offset
        if pos >= len(rank_table):
            continue
        r = rank_table.iloc[pos]
        with col:
            with st.container(border=True):
                st.caption(f'#{int(r["Rank"])}')
                st.markdown(f'**{r["Team"]}**')
                if 'Projected Wins' in rank_table:
                    st.caption(f'{r["Projected Wins"]:.1f} projected wins')

st.subheader('All Teams')
formatting = {'Projected Wins': '{:.1f}', 'Projected Losses': '{:.1f}', 'Points Allowed / Game': '{:.1f}'}
st.dataframe(rank_table.style.format({k: v for k, v in formatting.items() if k in rank_table}),
             hide_index=True, use_container_width=True)
if 'Projected Wins' in rank_table:
    st.caption('These are projections, not historical results. Each opponent is played once on a neutral court. '
               'Win totals use projected score margins; exact score ties count as half a win. '
               'Tiebreak: head-to-head among tied teams, then lowest projected points allowed within that group.')
