"""Shared, date-safe daily projection snapshots and grading for March Metrics V3."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
PAGES=Path(__file__).resolve().parent
if str(PAGES) not in sys.path: sys.path.insert(0,str(PAGES))
import _v3_shared as v

SNAP_COLUMNS=['GAME_ID','Date','Team1','Team1Acronym','Team2','Team2Acronym','Location','Home_Team',
              'ML1','ML2','Favorite','Spread','OverUnder','Projected_PTS1','Projected_PTS2',
              'Model_ML1','Model_ML2','Model_Cover1','Model_Cover2','Model_Over','Model_Under',
              'Model_Version','Snapshot_Date']

def clean_id(value):
    if pd.isna(value): return ''
    return str(value).strip().upper()

def load_data():
    signatures=tuple((key,name,(v.DATA/name).stat().st_mtime_ns if (v.DATA/name).is_file() else None)
                     for key,name in v.FILES.items())
    return v.read_sources(signatures)

def prep_for_day(sources, day):
    catalog=v.v3_attach_coaches(v.team_catalog(sources['teams']), sources.get('coaches'))
    past=v.prepare_games(sources.get('historical'),'historical',day,catalog)
    current=v.prepare_games(sources.get('season'),'season',day,catalog)
    combined=pd.concat([x for x in (past,current) if not x.empty],ignore_index=True) if not (past.empty and current.empty) else pd.DataFrame()
    if not combined.empty:
        combined=combined.sort_values('_SOURCE',key=lambda c:c.eq('season'),ascending=False).drop_duplicates('_KEY')
    engine=None
    if day>=v.ML_START_DATE and not current.empty:
        path=v.DATA/v.FILES['season']
        engine,_=v.v3_train_engine((path.stat().st_mtime_ns,day.isoformat()),current)
    return catalog,combined,current,engine

def snapshot_for_date(sources,day):
    """Only pregame information, never recover a snapshot from final scores."""
    schedule=sources.get('schedule')
    if schedule is None or schedule.empty: return pd.DataFrame(columns=SNAP_COLUMNS)
    sd=v.find_date(schedule)
    fixtures=schedule.loc[sd.eq(day)].copy()
    if fixtures.empty: return pd.DataFrame(columns=SNAP_COLUMNS)
    catalog,games,current,engine=prep_for_day(sources,day)
    cache={}; rows=[]
    for _,f in fixtures.iterrows():
        a,b=clean_id(f.get('Team1Acronym')),clean_id(f.get('Team2Acronym'))
        if a not in catalog.index or b not in catalog.index or games.empty: continue
        try:
            ctx1=v.make_context(a,b,f.get('Home_Team',''),f.get('Location',''),catalog)
            ctx2=v.make_context(b,a,f.get('Home_Team',''),f.get('Location',''),catalog)
            def get_side(ctx):
                key=(ctx['TEAM'],ctx['Coach'],ctx['OPP_CONF'],ctx['HAN'])
                if key not in cache: cache[key]=v.forecast_side(games,ctx)
                return cache[key]
            one,two=get_side(ctx1),get_side(ctx2)
            if one is None or two is None: continue
            his=v.model_matchup(one,two)
            historic=np.array([his['p1'],his['p2']],float)
            modeled=None
            if engine is not None:
                try: modeled=v.v3_predict_regressions(engine,catalog,a,b,ctx1,ctx2,f)
                except (ValueError,KeyError,TypeError): pass
            odds,_=v.v3_odds_estimate(current,f,ctx1,ctx2)
            tendency=v.v3_tendency_adjustment(games,ctx1,ctx2,f)
            scores,shares=v.v3_blend_forecasts(historic,modeled,odds,tendency)
            if not np.isfinite(scores).all(): continue
            gid=clean_id(f.get('GameID'))
            sim=v.v3_simulate(scores[0],scores[1],one,two,engine,games,gid or (a+b+str(day)))
            fav=v.key_text(f.get('Favorite',''))
            line=v.n(f.get('Spread',np.nan))
            signed=np.nan
            if pd.notna(line):
                if fav in (ctx1['TEAM'],a.lower()): signed=-abs(float(line))
                elif fav in (ctx2['TEAM'],b.lower()): signed=abs(float(line))
            total=v.n(f.get('OverUnder',np.nan))
            probs=v.v3_event_probabilities(sim,signed,float(total) if pd.notna(total) else np.nan)
            rows.append({'GAME_ID':gid,'Date':day.strftime('%m/%d/%Y'),'Team1':f.get('Team1',''),
                         'Team1Acronym':a,'Team2':f.get('Team2',''),'Team2Acronym':b,
                         'Location':f.get('Location',''),'Home_Team':f.get('Home_Team',''),
                         'ML1':f.get('ML1',np.nan),'ML2':f.get('ML2',np.nan),
                         'Favorite':f.get('Favorite',''),'Spread':f.get('Spread',np.nan),
                         'OverUnder':f.get('OverUnder',np.nan),'Projected_PTS1':round(float(scores[0]),2),
                         'Projected_PTS2':round(float(scores[1]),2),'Model_ML1':probs['ml1'],
                         'Model_ML2':probs['ml2'],'Model_Cover1':probs.get('cover1',np.nan),
                         'Model_Cover2':probs.get('cover2',np.nan),'Model_Over':probs.get('over',np.nan),
                         'Model_Under':probs.get('under',np.nan),'Model_Version':'V3',
                         'Snapshot_Date':day.strftime('%m/%d/%Y')})
        except (ValueError,KeyError,TypeError,IndexError): continue
    return pd.DataFrame(rows,columns=SNAP_COLUMNS)

def existing_snapshot(df):
    if df is None or df.empty: return pd.DataFrame(columns=SNAP_COLUMNS)
    df=df.copy();df.columns=df.columns.astype(str).str.strip()
    # If exported Numbers file has a leading unnamed index column, ignore it.
    df=df.loc[:,~df.columns.str.startswith('Unnamed:')]
    aliases={'GameID':'GAME_ID','Game_ID':'GAME_ID','GAMEID':'GAME_ID','DATE':'Date'}
    df=df.rename(columns={k:val for k,val in aliases.items() if k in df and val not in df})
    if 'GAME_ID' not in df: raise ValueError('Correct_Rate needs GAME_ID column matching the daily projection export.')
    for c in SNAP_COLUMNS:
        if c not in df: df[c]=np.nan
    df['GAME_ID']=df['GAME_ID'].map(clean_id)
    return df.loc[df.GAME_ID.ne(''),SNAP_COLUMNS].drop_duplicates('GAME_ID',keep='first')

def merge_snapshots(old,new):
    # Never overwrite a previously saved forecast with a hindsight forecast.
    return pd.concat([existing_snapshot(old),existing_snapshot(new)],ignore_index=True).drop_duplicates('GAME_ID',keep='first')

def grade_snapshots(snapshots,season,day):
    result=snapshots.copy()
    for c in ['Actual_PTS1','Actual_PTS2','ML_Correct','Spread_Correct','OU_Correct','Score_MAE','Result_Status']:
        result[c]=np.nan if c!='Result_Status' else 'Pending'
    if season is None or season.empty or result.empty: return result
    completed=season.copy()
    completed['_DATE']=v.find_date(completed)
    completed=completed.loc[completed['_DATE'].lt(day)].copy()
    if 'GAME_ID' not in completed: return result
    completed['GAME_ID']=completed['GAME_ID'].map(clean_id)
    completed['TEAM_ACR']=completed.get('TEAM_ACR',pd.Series('',index=completed.index)).fillna('').astype(str).str.upper().str.strip()
    completed['PTS']=pd.to_numeric(completed.get('PTS'),errors='coerce')
    completed['OPP_PTS']=pd.to_numeric(completed.get('OPP_PTS'),errors='coerce')
    completed=completed.loc[(completed.PTS.notna())&(completed.OPP_PTS.notna())]
    by_id={k:g for k,g in completed.groupby('GAME_ID',sort=False)}
    for i,row in result.iterrows():
        gid=clean_id(row.get('GAME_ID')); entries=by_id.get(gid)
        if entries is None:continue
        pick=entries.loc[entries.TEAM_ACR.eq(clean_id(row.get('Team1Acronym')))]
        if pick.empty:
            # Never assume Team1 orientation if only other team's row exists.
            other=entries.loc[entries.TEAM_ACR.eq(clean_id(row.get('Team2Acronym')))]
            if other.empty:continue
            p1=float(other.iloc[0].OPP_PTS);p2=float(other.iloc[0].PTS)
        else:
            p1=float(pick.iloc[0].PTS);p2=float(pick.iloc[0].OPP_PTS)
        result.at[i,'Actual_PTS1']=p1;result.at[i,'Actual_PTS2']=p2
        result.at[i,'Result_Status']='Final'
        predicted1=pd.to_numeric(row.get('Projected_PTS1'),errors='coerce')
        predicted2=pd.to_numeric(row.get('Projected_PTS2'),errors='coerce')
        if pd.notna(predicted1) and pd.notna(predicted2):
            result.at[i,'Score_MAE']=(abs(predicted1-p1)+abs(predicted2-p2))/2
            if p1!=p2 and predicted1!=predicted2:
                result.at[i,'ML_Correct']=int((predicted1>predicted2)==(p1>p2))
        fav=v.key_text(row.get('Favorite',''))
        s=pd.to_numeric(row.get('Spread'),errors='coerce')
        if pd.notna(s) and pd.notna(predicted1) and pd.notna(predicted2):
            a=clean_id(row.get('Team1Acronym')).lower();b=clean_id(row.get('Team2Acronym')).lower()
            n1=v.key_text(row.get('Team1',''));n2=v.key_text(row.get('Team2',''))
            signed=-abs(s) if fav in (a,n1) else abs(s) if fav in (b,n2) else np.nan
            if pd.notna(signed):
                predicted_margin=(predicted1-predicted2)+signed
                actual_margin=(p1-p2)+signed
                if predicted_margin!=0 and actual_margin!=0:
                    result.at[i,'Spread_Correct']=int((predicted_margin>0)==(actual_margin>0))
        ou=pd.to_numeric(row.get('OverUnder'),errors='coerce')
        if pd.notna(ou) and pd.notna(predicted1) and pd.notna(predicted2):
            if (predicted1+predicted2)!=ou and (p1+p2)!=ou:
                result.at[i,'OU_Correct']=int((predicted1+predicted2>ou)==(p1+p2>ou))
    return result
