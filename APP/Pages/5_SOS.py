"""March Metrics | 5_SOS.py

Drop this file in APP/Pages/. Reads existing COMP_STAT and All_Stats.
No changes to V3 regression, rankings, or success helpers are required.
Scores are retrospective descriptions, not pregame forecasts.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'Data' / '27_March_Madness_Data'
GAME_PATH = DATA / 'COMP_STAT-THE_TABLE.csv'
TEAM_PATH = DATA / 'All_Stats-THE_TABLE.csv'
DAY_PATH = DATA / 'Current_Day-Table 1.csv'

# Ten-professor score design: maximum per-game score is +1 and minimum is -1.
# A win over a strong opponent is rewarded more, a loss to a weak one penalized more.
# Score margin is adjusted for relative team strength and venue; ATS is optional.
RESULT_WEIGHT = 0.40
PERFORMANCE_WEIGHT = 0.32
ATS_WEIGHT = 0.18
VENUE_WEIGHT = 0.10


def csv_read(path):
    if not path.exists():
        raise FileNotFoundError(str(path))
    for enc in ('utf-8-sig', 'latin1'):
        try:
            frame = pd.read_csv(path, encoding=enc, low_memory=False)
            frame.columns = frame.columns.astype(str).str.strip()
            return frame
        except UnicodeDecodeError:
            pass
    raise ValueError(f'Unable to decode {path.name}')


def num(s):
    return pd.to_numeric(s, errors='coerce')


def upper(s):
    return s.fillna('').astype(str).str.strip().str.upper()


def dates(df):
    if {'MM', 'DD', 'YYYY'}.issubset(df.columns):
        out = pd.to_datetime(pd.DataFrame({
            'year': num(df['YYYY']), 'month': num(df['MM']),
            'day': num(df['DD'])}, index=df.index), errors='coerce')
    else:
        out = pd.Series(pd.NaT, index=df.index)
    for name in ('GAME_DATE', 'Date'):
        if name in df:
            parsed = pd.to_datetime(df[name], errors='coerce', format='%m/%d/%Y')
            parsed = parsed.fillna(pd.to_datetime(df[name], errors='coerce', format='%m/%d/%y'))
            out = out.fillna(parsed)
            break
    return out


def venue_code(raw, home_indicator=np.nan, away_indicator=np.nan, neutral_indicator=np.nan):
    val = str(raw).strip().upper()
    if val in ('HOME', 'H', '1'):
        return 'HOME'
    if val in ('AWAY', 'A', '2'):
        return 'AWAY'
    if val in ('NEUTRAL', 'N', '0'):
        return 'NEUTRAL'
    if pd.notna(home_indicator) and home_indicator == 1:
        return 'HOME'
    if pd.notna(away_indicator) and away_indicator == 1:
        return 'AWAY'
    if pd.notna(neutral_indicator) and neutral_indicator == 1:
        return 'NEUTRAL'
    return 'UNKNOWN'


def prepare(log, teams, day):
    required = {'TEAM_ACR', 'OPP_ACR', 'PTS', 'OPP_PTS'}
    absent = required - set(log.columns)
    if absent:
        raise ValueError(f'COMP_STAT missing columns: {", ".join(sorted(absent))}')
    required_teams = {'Teams', 'Acronym'}
    if not required_teams.issubset(teams.columns):
        raise ValueError('All_Stats needs Teams and Acronym columns.')

    catalog = teams[['Acronym','Teams']].copy()
    catalog['Acronym'] = upper(catalog['Acronym'])
    catalog = catalog.loc[catalog['Acronym'].ne('')].drop_duplicates('Acronym')
    names = dict(zip(catalog['Acronym'], catalog['Teams']))
    games = log.copy()
    games['_DATE'] = dates(games)
    games['TEAM_ACR'] = upper(games['TEAM_ACR'])
    games['OPP_ACR'] = upper(games['OPP_ACR'])
    games['PTS'] = num(games['PTS'])
    games['OPP_PTS'] = num(games['OPP_PTS'])
    games = games.loc[
        games['_DATE'].notna() & games['_DATE'].lt(day) &
        games['PTS'].notna() & games['OPP_PTS'].notna() &
        games['TEAM_ACR'].ne('') & games['OPP_ACR'].ne('') &
        games['TEAM_ACR'].ne(games['OPP_ACR']) &
        games['TEAM_ACR'].isin(names)
    ].copy()
    # The file is one row per team-game. Keep both perspectives but no duplicates.
    if 'GAME_ID' in games:
        ids = upper(games['GAME_ID'])
        blank = ids.eq('') | ids.eq('NAN')
        ids = ids.where(~blank, games['TEAM_ACR'] + '-' + games['OPP_ACR'] + '-' + games['_DATE'].dt.strftime('%Y%m%d'))
        games['_GAME_ID'] = ids
    else:
        games['_GAME_ID'] = games['TEAM_ACR'] + '-' + games['OPP_ACR'] + '-' + games['_DATE'].dt.strftime('%Y%m%d')
    games = games.sort_values('_DATE').drop_duplicates(['_GAME_ID', 'TEAM_ACR'], keep='last').copy()
    games['_VENUE'] = [venue_code(
        row.get('HAN',''), pd.to_numeric(row.get('HOME_IND'),errors='coerce'),
        pd.to_numeric(row.get('AWAY_IND'),errors='coerce'),
        pd.to_numeric(row.get('NEUTRAL_IND'),errors='coerce'))
        for _,row in games.iterrows()]
    games['_MARGIN'] = games['PTS'] - games['OPP_PTS']
    games['_WIN'] = (games['_MARGIN'] > 0).astype(float)
    games['_SPREAD'] = num(games['SPREAD']) if 'SPREAD' in games else np.nan
    games['_ATS_MARGIN'] = games['_MARGIN'] + games['_SPREAD']
    games['_OPPONENT_NAME'] = games['OPP_ACR'].map(names).fillna(
        games['OPP'].astype(str) if 'OPP' in games else games['OPP_ACR'])
    games['_TEAM_NAME'] = games['TEAM_ACR'].map(names)
    return games, catalog


def strength_estimates(games, catalog):
    """An iterative, schedule-adjusted estimate using ONLY completed games.

    Returns opponent quality between 0 and 1. Unknown teams are neutral (0.5).
    Shrink small samples toward .5 and bootstrap using opponents' qualities.
    """
    codes = catalog['Acronym'].tolist()
    nteams = len(codes)
    lookup = {code:i for i,code in enumerate(codes)}
    strength = np.full(nteams, .5)
    if games.empty:
        return pd.Series(strength,index=codes)
    i = games['TEAM_ACR'].map(lookup).to_numpy(dtype=int)
    j = games['OPP_ACR'].map(lookup).fillna(-1).to_numpy(dtype=int)
    margin = games['_MARGIN'].to_numpy(dtype=float)
    wins = games['_WIN'].to_numpy(dtype=float)
    played = np.bincount(i, minlength=nteams)
    # Initial estimates from wins + capped margins, smoothed for sparse schedules.
    win_sum = np.bincount(i, weights=wins, minlength=nteams)
    margin_sum = np.bincount(i, weights=np.clip(margin,-30,30), minlength=nteams)
    wp = (win_sum+2)/(played+4)
    avmargin = margin_sum/np.maximum(1,played)
    base = .72*wp + .28*(.5+.5*np.tanh(avmargin/17))
    valid_opp = j>=0
    for _ in range(15):
        opp_sum=np.bincount(i[valid_opp],weights=strength[j[valid_opp]],minlength=nteams)
        opp_count=np.bincount(i[valid_opp],minlength=nteams)
        average_opp=np.divide(opp_sum,opp_count,out=np.full(nteams,.5),where=opp_count>0)
        next_strength=np.clip(base+.47*(average_opp-.5),.03,.97)
        strength=.5*strength+.5*next_strength
    # Convert to a bounded team quality percentile, avoiding 0/1 certainty and
    # avoiding very early-season records looking like known national extremes.
    raw=pd.Series(strength,index=codes)
    percentiles=raw.rank(pct=True).to_numpy()
    reliability=played/(played+5.0)
    quality=.5+reliability*(percentiles-.5)
    return pd.Series(np.clip(quality,.05,.95),index=codes)


def build_scores(games, catalog):
    if games.empty:
        blank = catalog.rename(columns={'Teams':'Team'}).copy()
        for c in ('Games','Wins','Losses','SOS Score','Average/Game','Schedule Difficulty','Opponent Quality','ATS Win %'):
            blank[c]=0 if c in ('Games','Wins','Losses') else np.nan
        blank.insert(0,'Rank',np.arange(1,len(blank)+1))
        blank['Difficulty Rank']=np.arange(1,len(blank)+1)
        return blank, games
    quality=strength_estimates(games,catalog)
    a=games['TEAM_ACR'].map(quality).fillna(.5).to_numpy(dtype=float)
    b=games['OPP_ACR'].map(quality).fillna(.5).to_numpy(dtype=float)
    margin=games['_MARGIN'].to_numpy(dtype=float)
    win=margin>0
    venue=games['_VENUE']
    road=venue.eq('AWAY').to_numpy()
    home=venue.eq('HOME').to_numpy()
    venue_pts=np.where(home,3.0,np.where(road,-3.0,0.0))
    # Strength-adjusted expected margin is used for retrospective grading, not
    # treated as a calibrated prediction. Bad home losses are punished heavily.
    expected=22.0*(a-b)+venue_pts
    results=np.where(win,RESULT_WEIGHT*(.5+.5*b),-RESULT_WEIGHT*(1-.5*b))
    performance=PERFORMANCE_WEIGHT*np.tanh((margin-expected)/14.0)
    ats_margin=games['_ATS_MARGIN'].to_numpy(dtype=float)
    ats=np.where(np.isfinite(ats_margin),ATS_WEIGHT*np.tanh(np.nan_to_num(ats_margin)/9.0),0.0)
    venue_bonus=np.where(road & win,VENUE_WEIGHT,np.where(home & ~win,-VENUE_WEIGHT,0.0))
    total=np.clip(results+performance+ats+venue_bonus,-1.,1.)
    g=games.copy()
    g['Opponent Strength']=b
    g['Team Strength']=a
    g['Schedule Difficulty Game']=np.clip(b+np.where(road,.1,np.where(home,-.1,0.)),0,1)
    g['Result Contribution']=results
    g['Performance Contribution']=performance
    g['ATS Contribution']=ats
    g['Venue Contribution']=venue_bonus
    g['Game SOS Score']=total
    g['Expected Margin']=expected
    g['ATS Result']=np.where(~np.isfinite(ats_margin),'No line',np.where(ats_margin>0,'Cover',np.where(ats_margin<0,'No cover','Push')))

    agg=g.groupby('TEAM_ACR').agg(
        Games=('Game SOS Score','size'),
        Wins=('_WIN','sum'),
        **{'SOS Score':('Game SOS Score','sum'),
           'Average/Game':('Game SOS Score','mean'),
           'Schedule Difficulty':('Schedule Difficulty Game','mean'),
           'Opponent Quality':('Opponent Strength','mean')}
    )
    g['_ATS_WIN']=np.where(np.isfinite(ats_margin)&(ats_margin!=0),(ats_margin>0).astype(float),np.nan)
    ats_pct=g.groupby('TEAM_ACR')['_ATS_WIN'].mean()
    agg['ATS Win %']=ats_pct
    agg['Losses']=agg['Games']-agg['Wins']
    result=catalog.rename(columns={'Teams':'Team'}).set_index('Acronym').join(agg,how='left')
    for col in ('Games','Wins','Losses'):
        result[col]=result[col].fillna(0).astype(int)
    # Sum is the requested primary metric; average shows performance per game.
    result=result.reset_index().sort_values(['SOS Score','Average/Game','Schedule Difficulty'],ascending=[False,False,False],na_position='last').reset_index(drop=True)
    result.insert(0,'Rank',np.arange(1,len(result)+1))
    result['Difficulty Rank']=result['Schedule Difficulty'].rank(ascending=False,method='min',na_option='bottom').astype(int)
    return result,g


@st.cache_data(show_spinner=False)
def calculate(game_mtime,team_mtime,day_stamp):
    games, catalog=prepare(csv_read(GAME_PATH),csv_read(TEAM_PATH),pd.Timestamp(day_stamp))
    return build_scores(games,catalog)


st.title('March Metrics — Strength of Schedule')
st.caption('Every completed game contributes from −1 to +1. Rankings show the sum of those contributions.')
try:
    today=csv_read(DAY_PATH)
    if today.empty:
        raise ValueError('Current_Day has no date row.')
    row=today.iloc[0]
    day=pd.Timestamp(year=int(num(row['YYYY'])),month=int(num(row['MM'])),day=int(num(row['DD'])))
    leaderboard, game_details=calculate(GAME_PATH.stat().st_mtime_ns,TEAM_PATH.stat().st_mtime_ns,day.isoformat())
except (FileNotFoundError,ValueError,KeyError,TypeError,pd.errors.ParserError) as exc:
    st.error(f'Unable to load SOS: {exc}')
    st.stop()

st.caption(f'Databook date: {day:%m/%d/%Y} · completed COMP_STAT games before this date · D1 opponents in All_Stats receive an estimated strength rating.')

st.markdown('### Team standings')
col_a,col_b,col_c=st.columns(3)
col_a.metric('Teams',len(leaderboard))
col_b.metric('Completed team-games',len(game_details))
col_c.metric('Games graded ATS',int(game_details['_ATS_WIN'].notna().sum()) if not game_details.empty else 0)

st.dataframe(leaderboard[['Rank','Team','Games','Wins','Losses','SOS Score','Average/Game','Difficulty Rank','Schedule Difficulty','ATS Win %']].style.format({
    'SOS Score':'{:+.3f}', 'Average/Game':'{:+.3f}',
    'Schedule Difficulty':'{:.3f}', 'ATS Win %':'{:.1%}'},na_rep='—'),hide_index=True,use_container_width=True)

st.markdown('### Explore a team')
ordered=leaderboard['Team'].astype(str).tolist()
chosen=st.selectbox('Choose a team to expand its game-by-game SOS breakdown',ordered,index=0)
selected=leaderboard.loc[leaderboard['Team'].eq(chosen)].iloc[0]
acronym=selected['Acronym']
with st.expander(f"{chosen} — game-by-game contributions",expanded=True):
    if selected['Games']==0:
        st.info('No qualifying completed games before the selected databook date.')
    else:
        sub=game_details.loc[game_details['TEAM_ACR'].eq(acronym)].sort_values(['_DATE','_GAME_ID']).copy()
        sub['Running SOS']=sub['Game SOS Score'].cumsum()
        sub=sub.sort_values(['_DATE','_GAME_ID'],ascending=False)
        m1,m2,m3=st.columns(3)
        m1.metric('Cumulative SOS',f"{selected['SOS Score']:+.3f}")
        m2.metric('Average per game',f"{selected['Average/Game']:+.3f}")
        m3.metric('Schedule difficulty',f"{selected['Schedule Difficulty']:.3f}")
        sub['Date']=sub['_DATE'].dt.strftime('%m/%d/%Y')
        sub['Score']=sub['PTS'].astype(int).astype(str)+'–'+sub['OPP_PTS'].astype(int).astype(str)
        sub['Venue']=sub['_VENUE']
        sub['Opponent']=sub['_OPPONENT_NAME']
        sub['Spread']=sub['_SPREAD']
        sub['Cover Margin']=sub['_ATS_MARGIN']
        display=sub[['Date','Opponent','Venue','Score','Spread','ATS Result','Opponent Strength','Schedule Difficulty Game','Result Contribution','Performance Contribution','ATS Contribution','Venue Contribution','Game SOS Score','Running SOS']]
        def shade_game(x):
            if pd.isna(x):return ''
            return 'background-color: #c6efce; color: #14532d' if x>0 else ('background-color: #ffc7ce; color: #7f1d1d' if x<0 else '')
        st.dataframe(display.style.format({
            'Spread':'{:+.1f}', 'Cover Margin':'{:+.1f}',
            'Opponent Strength':'{:.3f}','Schedule Difficulty Game':'{:.3f}',
            'Result Contribution':'{:+.3f}','Performance Contribution':'{:+.3f}',
            'ATS Contribution':'{:+.3f}','Venue Contribution':'{:+.3f}',
            'Game SOS Score':'{:+.3f}','Running SOS':'{:+.3f}'},na_rep='—').map(shade_game,subset=['Game SOS Score']),hide_index=True,use_container_width=True)
        st.download_button('Download this team’s game breakdown',sub.to_csv(index=False).encode('utf-8-sig'),file_name=f'March_Metrics_SOS_{acronym}.csv',mime='text/csv')

st.download_button('Download complete SOS leaderboard',leaderboard.to_csv(index=False).encode('utf-8-sig'),file_name=f'March_Metrics_SOS_{day:%Y-%m-%d}.csv',mime='text/csv')
with st.expander('How SOS is calculated — ten-professor methodology'):
    st.markdown('''**Each game contributes between −1 and +1** through four components:

- **40% game result and opponent quality:** wins against stronger opponents earn more; losses to weaker opponents hurt more.
- **32% strength- and venue-adjusted scoring margin:** rewards beating expectations, penalizes lopsided defeats.
- **18% against-the-spread margin:** positive cover margins raise the score; negative margins reduce it. If no line exists, this portion contributes zero.
- **10% venue bonus/penalty:** an away win earns an additional bonus; a home loss receives an additional penalty.

**Opponent quality** is estimated from this season's completed-game network, starting from records and scoring margins, iteratively incorporating opponents' strength and shrinking small samples toward neutral. Unknown opponents are given a neutral 0.5 rather than invented strength. **Schedule Difficulty** averages opponent quality with a 0.1 bonus for away games and a 0.1 discount for home games.

**SOS Score** is the sum of game contributions. **Average/Game** prevents teams with more games from automatically looking superior. **Schedule Difficulty** measures opponents faced independent of whether the team won.

This is a new March Metrics retrospective performance index, not an NCAA NET rating or a calibrated probability. Strength estimates use completed games available before the selected databook date, including games after earlier games, so the index should not be presented as a pregame prediction made at the time.''')
