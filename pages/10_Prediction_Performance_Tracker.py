"""V11: portable multi-week snapshots, optional CFBD scoring, and performance trends."""
import pandas as pd
import requests
import streamlit as st
from prediction_tracker import snapshot,grade,summary,by_edge
from performance_v11 import load_snapshots,weekly_metrics,archive_zip

st.title('📈 Prediction Performance Tracker — V11')
st.info('Keep original pregame snapshots. This page does not permanently store files on the Streamlit server. Download an archive after every grading session.')
st.caption('Forward-testing research only. Snapshot timestamps are not tamper-proof evidence, and model cover probabilities are not validated.')

st.subheader('1. Create a timestamped pregame snapshot')
pred=st.file_uploader('Upload a predictions CSV (skip if you already have a saved snapshot)',type='csv',key='v11_new_snapshot')
if pred is not None:
    try:
        raw=pd.read_csv(pred)
        snap=snapshot(raw)
        st.write(f'{len(snap)} games; {int(snap.captured_before_kickoff.sum())} pre-kickoff at snapshot creation.')
        st.download_button('Download timestamped pregame snapshot',snap.to_csv(index=False).encode(),
            f'cfb_pregame_snapshot_{pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")}.csv','text/csv')
    except Exception as exc:
        st.error(f'Snapshot error: {exc}')

st.subheader('2. Import your weekly snapshot archive')
files=st.file_uploader('Upload one or more saved pregame snapshot CSV files',type='csv',accept_multiple_files=True,key='v11_archive')
if not files:
    st.info('Start by uploading your saved November 21 pregame snapshot. You can add more weeks as they happen.')
    st.stop()
try:
    saved=load_snapshots(files)
    if 'advanced_enabled' in saved.columns:
        saved['model_version']=saved.advanced_enabled.astype(str).str.lower().map({'true':'V7 advanced','false':'Score-only'}).fillna('Unspecified')
    else:
        saved['model_version']='Unspecified'
    st.metric('Archived unique predictions',len(saved))
    st.write(f"{int(saved.captured_before_kickoff.sum())} recorded before kickoff; {int((~saved.captured_before_kickoff).sum())} late/unverified")
    st.dataframe(saved[['date','home_team','away_team','model_version','snapshot_status']],hide_index=True,use_container_width=True)
    st.download_button('Download consolidated snapshot archive',archive_zip(saved),
        'cfb_snapshot_archive.zip','application/zip')
except Exception as exc:
    st.error(f'Archive validation failed: {exc}')
    st.stop()

st.subheader('3. Get completed game scores')
source=st.radio('Final scores source',['CFBD API (completed games)','Upload final scores CSV'],horizontal=True,key='v11_score_source')
results=None
if source.startswith('CFBD'):
    try:
        api_key=st.secrets.get('CFBD_API_KEY','')
    except Exception:
        api_key=''
    seasons=sorted(pd.to_datetime(saved.date,utc=True).dt.year.unique().tolist())
    years=st.multiselect('Seasons to fetch',seasons,default=seasons)
    @st.cache_data(ttl=3600,show_spinner=False)
    def fetch_results(key,year):
        response=requests.get('https://api.collegefootballdata.com/games',
            params={'year':int(year),'seasonType':'regular','classification':'fbs'},
            headers={'Authorization':f'Bearer {key}'},timeout=45)
        response.raise_for_status()
        return pd.DataFrame([{'date':g['startDate'],'home_team':g['homeTeam'],'away_team':g['awayTeam'],
            'home_score':g['homePoints'],'away_score':g['awayPoints']} for g in response.json()
            if g.get('completed') and g.get('homePoints') is not None and g.get('awayPoints') is not None])
    if st.button('Fetch completed scores',disabled=not bool(api_key) or not years):
        try:
            frames=[fetch_results(api_key,y) for y in years]
            st.session_state['v11_results']=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
            st.success(f'Fetched {len(st.session_state["v11_results"])} completed games.')
        except Exception as exc:
            st.error(f'CFBD score retrieval failed: {exc}. You can upload a scores CSV instead.')
    results=st.session_state.get('v11_results')
else:
    scores=st.file_uploader('Final scores CSV: date,home_team,away_team,home_score,away_score',type='csv',key='v11_scores')
    if scores is not None:
        results=pd.read_csv(scores)

if results is None:
    st.info('Load completed scores to see performance. Your snapshot archive can already be downloaded above.')
    st.stop()
if results.empty:
    st.info('No completed scores found yet.')
    st.stop()

try:
    graded=grade(saved,results)
    stats=summary(graded)
    st.subheader('4. Forward performance')
    a,b,c,d=st.columns(4)
    a.metric('Graded games',stats['graded_games'])
    b.metric('ATS accuracy',f"{stats['ats_accuracy']:.1%}" if stats['ats_accuracy'] is not None else 'N/A')
    c.metric('Winner accuracy',f"{stats['winner_accuracy']:.1%}" if stats['winner_accuracy'] is not None else 'N/A')
    d.metric('Awaiting results',int(graded.grade_status.eq('AWAITING RESULT').sum()))
    if stats['model_margin_MAE'] is not None:
        st.write(f"Model margin MAE: **{stats['model_margin_MAE']:.2f}** | Sportsbook margin MAE: **{stats['market_margin_MAE']:.2f}**")
    st.caption('Headline metrics exclude late/unverified snapshots; ATS pushes and no-lean games are excluded from ATS accuracy.')
    weekly=weekly_metrics(graded)
    st.subheader('Weekly performance by model version')
    st.dataframe(weekly,hide_index=True,use_container_width=True)
    if not weekly.empty:
        plot=weekly.dropna(subset=['ats_accuracy']).copy()
        if not plot.empty:
            plot['week_label']=plot.week_start+' · '+plot.model_version
            st.bar_chart(plot.set_index('week_label')['ats_accuracy'])
        error=weekly.dropna(subset=['model_margin_MAE','market_margin_MAE']).copy()
        if not error.empty:
            error['week_label']=error.week_start+' · '+error.model_version
            st.line_chart(error.set_index('week_label')[['model_margin_MAE','market_margin_MAE']])
    st.subheader('ATS performance by model disagreement')
    st.dataframe(by_edge(graded),hide_index=True,use_container_width=True)
    st.subheader('Game-by-game grading')
    st.dataframe(graded[['date','home_team','away_team','model_version','home_spread','projected_home_margin',
        'home_score','away_score','grade_status','ats_result','margin_absolute_error']],hide_index=True,use_container_width=True)
    st.download_button('Download graded results CSV',graded.to_csv(index=False).encode(),
        'cfb_graded_predictions.csv','text/csv')
    st.download_button('Download full archive ZIP (snapshots + results + weekly report)',archive_zip(saved,graded),
        'cfb_v11_performance_archive.zip','application/zip')
except Exception as exc:
    st.error(f'Grading failed: {exc}')
