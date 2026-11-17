"""March Metrics Rankings V3 — same ten-regression engine as Today's Games.
Install alongside APP/Pages/_v3_shared.py and _march_workflow.py.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import _v3_shared as v
from _march_workflow import load_data, prep_for_day

st.title('March Metrics Rankings')
try:
    sources=load_data()
    dates=sources.get('date');stats=sources.get('teams')
    if stats is None or dates is None or dates.empty:
        st.error('Need All_Stats-THE_TABLE.csv and Current_Day-Table 1.csv.');st.stop()
    day=v.date_from_row(dates.iloc[0]); catalog=v.team_catalog(stats)
except (KeyError,ValueError,TypeError) as exc:
    st.error(str(exc));st.stop()

names=catalog['Teams'].astype(str)
acronyms=catalog.index.to_numpy()
N=len(catalog)

@st.cache_data(show_spinner=False)
def neutral_rankings(day_stamp, files_sig):
    # Uses same eligible COMP_STAT rows, score-regression objectives, meta learner,
    # context history and tendency shifts as Today's Games. Market-specific 10%
    # is unavailable in a neutral matchup without actual sportsbook lines.
    sources=load_data()
    catalog,games,current,engine=prep_for_day(sources,pd.Timestamp(day_stamp))
    if engine is None:
        return None,'V3 needs qualifying COMP_STAT rows and date >= November 20.'
    codes=catalog.index.to_numpy(); nteams=len(codes)
    stats=catalog
    gps=(pd.to_numeric(stats.get('Win'),errors='coerce').fillna(0)
         +pd.to_numeric(stats.get('Loss'),errors='coerce').fillna(0)).to_numpy(float)
    gp=np.where(gps>0,gps,np.nan)
    colnames=engine['features']
    # Assemble one feature vector per team for use on BOTH sides of a matchup.
    first={}; second={}
    for feature in colnames:
        if feature.startswith('TEAM_PRE_'):
            suffix=feature[9:]
            first[feature]=np.array([v.v3_profile_value(stats.iloc[i],suffix,gp[i]) for i in range(nteams)],float)
        elif feature.startswith('OPP_PRE_CLUTCH_'):
            src='CLUTCH_'+feature[len('OPP_PRE_CLUTCH_'):]
            second[feature]=pd.to_numeric(stats.get(src,pd.Series(np.nan,index=stats.index)),errors='coerce').to_numpy(float)
        elif feature.startswith('OPP_PRE_'):
            suffix=feature[8:]
            second[feature]=np.array([v.v3_profile_value(stats.iloc[i],suffix,gp[i]) for i in range(nteams)],float)
        elif feature.startswith('OPP_CORE_'):
            src='CORE_'+feature[len('OPP_CORE_'):]
            second[feature]=pd.to_numeric(stats.get(src,pd.Series(np.nan,index=stats.index)),errors='coerce').to_numpy(float)
        elif feature.startswith('CORE_'):
            first[feature]=pd.to_numeric(stats.get(feature,pd.Series(np.nan,index=stats.index)),errors='coerce').to_numpy(float)
        elif feature.startswith('TEAM_CLUTCH_'):
            src='CLUTCH_'+feature[len('TEAM_CLUTCH_'):]
            first[feature]=pd.to_numeric(stats.get(src,pd.Series(np.nan,index=stats.index)),errors='coerce').to_numpy(float)
    i=np.repeat(np.arange(nteams),nteams)
    j=np.tile(np.arange(nteams),nteams)
    cl=pd.to_numeric(stats.get('CLUTCH_FGM',pd.Series(np.nan,index=stats.index)),errors='coerce').to_numpy(float)
    eligible=(cl[i]>0)&(cl[j]>0)
    data={}
    for feat in engine['matrix_columns']:
        if feat in first: val=first[feat][i]
        elif feat in second: val=second[feat][j]
        elif feat=='VENUE_NEUTRAL':val=np.ones(len(i))
        elif feat in ('VENUE_HOME','VENUE_AWAY'):val=np.zeros(len(i))
        else:val=np.full(len(i),np.nan)
        if 'CLUTCH' in feat: val=np.where(eligible,val,np.nan)
        data[feat]=val
    frame=pd.DataFrame(data).replace([np.inf,-np.inf],np.nan)
    X=engine['scaler'].transform(engine['imputer'].transform(frame))
    X=np.column_stack([np.ones(len(X)),np.clip(X,-8,8)])
    raw=np.zeros((len(X),2))
    for label,_,weight in v.V3_OBJECTIVES:
        raw+=(weight/100)*(X@engine['heads'][label]+70)
    correction=np.column_stack([m.predict(X[:,1:]) for m in engine['meta']])
    scored=raw[:,0].reshape(nteams,nteams)
    allowed=raw[:,1].reshape(nteams,nteams)
    meta_scored=(raw[:,0]+np.clip(correction[:,0],-8,8)).reshape(nteams,nteams)
    meta_allowed=(raw[:,1]+np.clip(correction[:,1],-8,8)).reshape(nteams,nteams)
    # 75% offense + 25% opposing defensive estimate, consistent with V3.
    regression=.75*scored+.25*allowed.T
    supervised=.75*meta_scored+.25*meta_allowed.T
    # Context forecast calculated once per team (rather than 130,000 times).
    # Opponent-specific conference contexts cannot be included without losing
    # the fast vectorization advantage; this is disclosed in UI.
    off=np.full(nteams,np.nan);defn=np.full(nteams,np.nan);tendency=np.zeros(nteams)
    for k,acr in enumerate(codes):
        row=catalog.loc[acr]
        ctx={'TEAM':v.key_text(row['Teams']),'Coach':v.key_text(row.get('Coach Name','')),
             'CONF':v.key_text(row.get('Conference','')),'OPP_CONF':'','HAN':'NEUTRAL'}
        side=v.forecast_side(games,ctx) if not games.empty else None
        if side:
            off[k]=side['mean'].get('PTS',np.nan)
            defn[k]=side['mean'].get('OPP_PTS',np.nan)
        subset=games.loc[games['TEAM'].eq(ctx['TEAM'])] if not games.empty else pd.DataFrame()
        if not subset.empty:
            margins=pd.to_numeric(subset.get('COVER_MARGIN_CALC'),errors='coerce').to_numpy(float)
            ww=subset['_WEIGHT'].to_numpy(float)
            good=np.isfinite(margins)&np.isfinite(ww)
            if good.any():
                eff=ww[good].sum()**2/np.sum(ww[good]**2)
                tendency[k]=float(np.clip((eff/(eff+25))*np.average(np.clip(margins[good],-12,12),weights=ww[good])*.25,-4,4))
    # Fall back to national average only where a team lacks usable records.
    gmean=float(pd.to_numeric(games['PTS'],errors='coerce').mean()) if not games.empty else 70.
    off=np.where(np.isfinite(off),off,gmean)
    defn=np.where(np.isfinite(defn),defn,gmean)
    history=.75*off[:,None]+.25*defn[None,:]
    tendency_score=history+tendency[:,None]
    # Comparable betting-line component cannot exist for hypothetical games.
    shares={'Regression objectives':.42,'Team/coach history':.33,
            'Supervised correction':.10,'Betting tendencies':.05}
    scale=sum(shares.values())
    points=(shares['Regression objectives']*regression+
            shares['Team/coach history']*history+
            shares['Supervised correction']*supervised+
            shares['Betting tendencies']*tendency_score)/scale
    points=np.clip(points,0,160)
    np.fill_diagonal(points,np.nan)
    diff=points-points.T
    wins=(diff>1e-9).sum(axis=1)+.5*((np.abs(diff)<=1e-9)&~np.eye(nteams,dtype=bool)).sum(axis=1)
    conceded=np.nanmean(points.T,axis=1)
    # Tied wins: compare the tied team's direct projected results, then
    # predicted points conceded against others in that same tied group.
    groups={}
    for k,score in enumerate(wins):groups.setdefault(float(score),[]).append(k)
    ranking=[]
    for key in sorted(groups,reverse=True):
        group=groups[key]
        if len(group)==1:ranking+=group;continue
        tie_diff=diff[np.ix_(group,group)]
        h2h=(tie_diff>1e-9).sum(axis=1)
        tie_allowed=np.nanmean(points[np.ix_(group,group)].T,axis=1)
        ordering=sorted(range(len(group)),key=lambda z:(-h2h[z],float(tie_allowed[z]),str(stats.loc[codes[group[z]],'Teams']).casefold()))
        ranking += [group[z] for z in ordering]
    table=pd.DataFrame({'Rank':np.arange(1,nteams+1),
         'Team':catalog.iloc[ranking]['Teams'].to_numpy(),
         'Acronym':codes[ranking], 'Projected Wins':wins[ranking],
         'Projected Losses':(nteams-1)-wins[ranking],
         'Average Points Allowed':conceded[ranking]})
    return table,None

ranked=None
if day>=v.ML_START_DATE:
    try:
        with st.spinner('Running V3 neutral-court round robin...'):
            sig=tuple((name,(v.DATA/name).stat().st_mtime_ns if (v.DATA/name).is_file() else None)
                      for name in v.FILES.values())
            ranked,msg=neutral_rankings(day.isoformat(),sig)
        if msg:st.info(msg)
    except Exception as exc:
        st.warning(f'Neutral simulation unavailable: {exc}')
else:
    st.info('Before November 20, 2026, rankings are alphabetical until V3 activates.')
if ranked is None:
    sorted_rows=catalog.sort_values('Teams',key=lambda x:x.str.casefold())
    ranked=pd.DataFrame({'Rank':range(1,len(sorted_rows)+1),
                         'Team':sorted_rows['Teams'].to_numpy(),
                         'Acronym':sorted_rows.index.to_numpy()})
else:
    st.caption('V3 neutral simulations use the same ten regression objectives and team/coach history as Today’s Games. '
               'Unknown hypothetical betting lines mean the market-specific module is excluded; the normal historical '
               'component is approximated by neutral-site team and coach profiles for speed.')
st.subheader('Top 25')
for start in range(0,min(len(ranked),25),5):
    cols=st.columns(5,gap='small')
    for k,col in enumerate(cols):
        idx=start+k
        if idx>=len(ranked):break
        r=ranked.iloc[idx]
        with col:
            with st.container(border=True):
                st.caption(f'#{int(r.Rank)}')
                st.markdown(f'**{r.Team}**')
                if 'Projected Wins' in ranked: st.caption(f'{r["Projected Wins"]:.1f} projected wins')
st.subheader('All Teams')
st.dataframe(ranked,hide_index=True,use_container_width=True)
