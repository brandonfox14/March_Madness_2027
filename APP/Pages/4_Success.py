"""March Metrics V3 forecast archive, daily CSV export and postgame grading.
Save APP/Pages/4_success.py with _v3_shared.py and _march_workflow.py.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import _v3_shared as v
from _march_workflow import (SNAP_COLUMNS,load_data,existing_snapshot,
                             snapshot_for_date,merge_snapshots,grade_snapshots)

st.title('Prediction Success')
st.caption('Saved pregame forecasts vs completed games — no hindsight reforecasting.')
try:
    sources=load_data()
    if sources.get('date') is None or sources['date'].empty:
        st.error('Missing Current_Day-Table 1.csv');st.stop()
    day=v.date_from_row(sources['date'].iloc[0])
except (ValueError,KeyError,TypeError) as exc:
    st.error(f'Databook date unavailable: {exc}');st.stop()

CORRECT_NAMES=['Correct_Rate-Table 1.csv','Correct_Rate.csv','Correct_Rate-THE_TABLE.csv']
correct_path=next((v.DATA/name for name in CORRECT_NAMES if (v.DATA/name).is_file()),None)
if correct_path:
    try:
        saved=existing_snapshot(v.read_csv(correct_path))
        st.caption(f'Reading saved predictions from {correct_path.name}.')
    except ValueError as exc:
        saved=pd.DataFrame(columns=SNAP_COLUMNS)
        st.warning(f'Could not read Correct_Rate: {exc}. The export below uses the required columns.')
else:
    saved=pd.DataFrame(columns=SNAP_COLUMNS)
    st.info('No Correct_Rate CSV found yet. Export your first day of predictions below, then save that '
            'CSV in Data/27_March_Madness_Data as Correct_Rate-Table 1.csv.')

# No completed-game outcomes are used to create these forecasts.
# Export the SAME date shown on Today's Games (Current_Day).
# Save this snapshot before tip-off; never regenerate past predictions for grading.
export_day=day
st.subheader("Today's Games — Predictions for Correct_Rate")
st.caption(f"Databook date: {export_day:%b %d, %Y} · Save these forecasts before games begin.")
try:
    with st.spinner("Preparing today's V3 forecasts..."):
        upcoming=snapshot_for_date(sources,export_day)
except Exception as exc:
    upcoming=pd.DataFrame(columns=SNAP_COLUMNS)
    st.warning(f"Unable to forecast today's games: {exc}")

if upcoming.empty:
    st.info('No forecastable games for the current databook date. Confirm this date is present in Schedule and '
            'that the team acronyms and completed historical games exist.')
else:
    st.dataframe(upcoming,hide_index=True,use_container_width=True)
    st.download_button("Download TODAY's projections CSV",upcoming.to_csv(index=False).encode('utf-8-sig'),
        file_name=f'March_Metrics_Projections_{export_day:%Y-%m-%d}.csv',mime='text/csv')

# Combine today's saved forecasts with the archive; previously saved rows take priority.
combined=merge_snapshots(saved,upcoming)
st.download_button('Download UPDATED Correct_Rate (old + new; no duplicate GameIDs)',
        combined.to_csv(index=False).encode('utf-8-sig'),
        file_name='Correct_Rate-Table 1.csv',mime='text/csv',
        help='Replace your exported Correct_Rate CSV with this cumulative file. Existing predictions are never overwritten.')
st.caption("Using Apple Numbers? Import today's projections CSV and append its rows to your Correct_Rate "
           'table, or use the cumulative download to replace the CSV. The app cannot write directly into Numbers.')

st.divider()
st.subheader('Actual Prediction Accuracy')
if saved.empty:
    st.info('Saved game forecasts will be graded after completed results appear in COMP_STAT. '
            'A score cannot be graded until its *pregame* prediction is saved in Correct_Rate.')
else:
    graded=grade_snapshots(saved,sources.get('season'),day)
    final=graded.loc[graded.Result_Status.eq('Final')].copy()
    cols=st.columns(4)
    for idx,(key,label) in enumerate([('ML_Correct','ML accuracy'),('Spread_Correct','Spread accuracy'),
                                       ('OU_Correct','O/U accuracy')]):
        known=pd.to_numeric(final[key],errors='coerce').dropna()
        cols[idx].metric(label,f'{known.mean():.1%}' if len(known) else 'N/A',
                         help=f'{len(known)} graded bets; pushes and missing lines excluded')
        cols[idx].caption(f'{int(known.sum())}/{len(known)} correct' if len(known) else 'No graded outcomes')
    mae=pd.to_numeric(final['Score_MAE'],errors='coerce').dropna()
    cols[3].metric('Avg score error',f'{mae.mean():.1f} pts' if len(mae) else 'N/A')
    st.caption(f'{len(final)} completed saved games · {len(graded)-len(final)} awaiting COMP_STAT results')
    if not final.empty:
        by_date=final.copy()
        by_date['Date']=pd.to_datetime(by_date['Date'],errors='coerce')
        summary=by_date.groupby('Date',dropna=False).agg(
            Games=('GAME_ID','count'),ML=('ML_Correct','mean'),Spread=('Spread_Correct','mean'),
            OverUnder=('OU_Correct','mean'),Score_MAE=('Score_MAE','mean')).reset_index().sort_values('Date',ascending=False)
        st.markdown('**Accuracy by game date**')
        st.dataframe(summary.style.format({'ML':'{:.1%}','Spread':'{:.1%}',
                  'OverUnder':'{:.1%}','Score_MAE':'{:.1f}'},na_rep='N/A'),
                  hide_index=True,use_container_width=True)
    st.markdown('**All saved forecasts and actual results**')
    display=['Date','GAME_ID','Team1','Team2','Projected_PTS1','Projected_PTS2',
             'Actual_PTS1','Actual_PTS2','ML_Correct','Spread_Correct','OU_Correct',
             'Score_MAE','Result_Status']
    st.dataframe(graded[display],hide_index=True,use_container_width=True)
    st.download_button('Download graded prediction history CSV',graded.to_csv(index=False).encode('utf-8-sig'),
                       file_name='March_Metrics_Success_Report.csv',mime='text/csv')
st.caption('ML grades the predicted winner. Spread grades the predicted side against the saved market line. '
           'O/U grades predicted total versus saved total. Pushes and missing lines are omitted from percentages. '
           'These are pick-direction hit rates, not betting ROI or probability calibration.')
