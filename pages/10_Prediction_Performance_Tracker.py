"""Version 10: import immutable pregame snapshots and grade against results."""
import os
import requests
import pandas as pd
import streamlit as st
from prediction_tracker import snapshot, grade, summary, by_edge

st.title('📈 Prediction Performance Tracker — V10')
st.info('Download and keep a pregame snapshot BEFORE kickoff. Streamlit session data is not permanent. Upload the saved snapshot later to grade it. A snapshot made after kickoff is marked unverified.')
st.caption('Research tracking only. An archived prediction is not proof of a validated betting edge.')

st.subheader('1. Archive predictions before kickoff')
pred=st.file_uploader('Upload a predictions CSV exported from the 2026 Game Predictions page',type='csv',key='snapshot_source')
if pred:
    try:
        raw=pd.read_csv(pred)
        snap=snapshot(raw)
        valid=int(snap.captured_before_kickoff.sum())
        st.write(f'{len(snap)} games; {valid} captured before scheduled kickoff, {len(snap)-valid} late/unverified.')
        st.dataframe(snap[['date','home_team','away_team','home_spread','model_vs_market_points','snapshot_status']],hide_index=True)
        st.download_button('Download timestamped prediction snapshot',snap.to_csv(index=False).encode(),
            f'cfb_pregame_snapshot_{pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")}.csv','text/csv')
    except Exception as exc: st.error(f'Snapshot failed: {exc}')

st.subheader('2. Grade saved snapshots')
snap_upload=st.file_uploader('Upload one or more saved prediction snapshots',type='csv',accept_multiple_files=True,key='grade_snapshots')
source=st.radio('Final scores source',['CFBD API (completed games)','Upload final scores CSV'],horizontal=True)
results=None
if source.startswith('CFBD'):
    try: api_key=st.secrets.get('CFBD_API_KEY','')
    except Exception: api_key=''
    year=st.number_input('Season',min_value=2019,max_value=2100,value=2026,step=1)
    @st.cache_data(ttl=3600,show_spinner=False)
    def fetch_results(key,season):
        response=requests.get('https://api.collegefootballdata.com/games',params={'year':int(season),'seasonType':'regular','classification':'fbs'},headers={'Authorization':f'Bearer {key}'},timeout=45)
        response.raise_for_status()
        return pd.DataFrame([{'date':g['startDate'],'home_team':g['homeTeam'],'away_team':g['awayTeam'],
            'home_score':g['homePoints'],'away_score':g['awayPoints']} for g in response.json()
            if g.get('completed') and g.get('homePoints') is not None and g.get('awayPoints') is not None])
    if st.button('Fetch completed scores',disabled=not bool(api_key)):
        try: st.session_state['tracker_results']=fetch_results(api_key,int(year))
        except Exception as exc: st.error(f'CFBD results fetch failed: {exc}. You can upload a scores CSV instead.')
    results=st.session_state.get('tracker_results')
else:
    scores_upload=st.file_uploader('CSV: date,home_team,away_team,home_score,away_score',type='csv',key='scores')
    if scores_upload: results=pd.read_csv(scores_upload)

if snap_upload and results is not None:
    try:
        saved=pd.concat([pd.read_csv(f) for f in snap_upload],ignore_index=True)
        graded=grade(saved,results)
        stats=summary(graded)
        st.subheader('Verified pregame results')
        a,b,c=st.columns(3)
        a.metric('Graded games',stats['graded_games'])
        b.metric('ATS accuracy',f"{stats['ats_accuracy']:.1%}" if stats['ats_accuracy'] is not None else 'N/A')
        c.metric('Winner accuracy',f"{stats['winner_accuracy']:.1%}" if stats['winner_accuracy'] is not None else 'N/A')
        st.write(f"Model margin MAE: {stats['model_margin_MAE']:.2f} | Market margin MAE: {stats['market_margin_MAE']:.2f}" if stats['model_margin_MAE'] is not None else 'No verified completed games yet.')
        st.caption('Only snapshots demonstrably captured before kickoff count in headline metrics. Late/unverified imports remain visible but excluded.')
        st.dataframe(by_edge(graded),hide_index=True)
        st.dataframe(graded[['date','home_team','away_team','home_spread','projected_home_margin','home_score','away_score','grade_status','ats_result','margin_absolute_error']],hide_index=True)
        st.download_button('Download graded results CSV',graded.to_csv(index=False).encode(),'cfb_graded_predictions.csv','text/csv')
        pending=int(graded.grade_status.eq('AWAITING RESULT').sum())
        if pending: st.info(f'{pending} predictions are still awaiting completed scores.')
    except Exception as exc: st.error(f'Grading failed: {exc}')
