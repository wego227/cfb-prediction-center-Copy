"""Streamlit CFB dashboard. Run: streamlit run dashboard.py"""
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from joblib import load
from predictor import validate,build_features,predictions
from odds import fetch_draftkings
from analysis import analyze,parlays
from cfbd_data import download_history
from predictor import fit_models,evaluate
from joblib import dump
import tempfile

st.set_page_config(page_title='CFB Prediction Center',layout='wide')
st.title('🏈 CFB Prediction Center')
st.caption('DraftKings line comparison • model research tool • no bets placed')
st.warning('The baseline model is NOT calibrated or validated for wagering. Estimated edges and parlay probabilities are experimental, not recommendations.')
with st.sidebar:
    st.header('Data & model')
    historical=st.file_uploader('Completed games CSV',type='csv')
    st.caption('Or use the historical-data download below.')
    model_file=st.file_uploader('Trained model (.joblib; trusted files only)',type='joblib')
    st.caption('Never load untrusted joblib files.')
    source=st.radio('Odds source',['DraftKings via The Odds API','Upload DraftKings odds CSV'])
    odds_file=st.file_uploader('Odds CSV',type='csv') if source.startswith('Upload') else None
    api_key=st.text_input('The Odds API key',type='password',value=os.getenv('THE_ODDS_API_KEY','')) if source.startswith('DraftKings') else None
    min_edge=st.slider('Minimum estimated probability edge',0.,.20,.03,.005)
    st.caption('Team names must match exactly across datasets; timestamps are UTC.')

st.subheader('Historical data and model training')
try:
    cfbd_key=st.secrets.get('CFBD_API_KEY','')
except Exception:
    cfbd_key=''
if not cfbd_key:
    st.info('Set CFBD_API_KEY in Streamlit app secrets to enable historical downloads.')
if st.button('Download history and train model', disabled=not bool(cfbd_key)):
    try:
        status=st.empty()
        with st.spinner('Downloading and training; this may take several minutes…'):
            history,report=download_history(cfbd_key,2019,2025,progress=lambda x:status.info(x))
            history=validate(history)
            X,details=build_features(history)
            train=details.season<2025
            test=details.season==2025
            if train.sum()<100 or test.sum()<10:raise ValueError('Not enough historical games to train/test.')
            backtest_model=fit_models(X.loc[train],details.loc[train])
            metrics,_=evaluate(backtest_model,X.loc[test],details.loc[test])
            final_model=fit_models(X,details)
            st.session_state['history_auto']=history
            st.session_state['model_auto']=final_model
            st.session_state['training_metrics']=metrics
            st.session_state['download_report']=report
        status.success('History downloaded and model trained. This is a research baseline, not a validated betting edge.')
    except Exception as exc:
        st.error(f'Historical data/training failed: {exc}')
if 'training_metrics' in st.session_state:
    st.write('Held-out 2025 evaluation (trained on earlier seasons):',st.session_state['training_metrics'])
    st.write('Download coverage:',st.session_state['download_report'])
    st.download_button('Download historical games CSV',st.session_state['history_auto'].to_csv(index=False),'cfb_games.csv','text/csv')
    with tempfile.NamedTemporaryFile(suffix='.joblib') as tmp:
        dump(st.session_state['model_auto'],tmp.name)
        with open(tmp.name,'rb') as f:
            st.download_button('Download trained model',f.read(),'cfb_model.joblib','application/octet-stream')

if source.startswith('DraftKings'):
    if st.button('Fetch current DraftKings lines'):
        try:
            st.session_state['odds']=fetch_draftkings(api_key)
            st.success(f"Received {len(st.session_state['odds'])} DraftKings games")
        except Exception as exc:st.error(f'Odds fetch failed: {exc}')
elif odds_file:
    try:st.session_state['odds']=pd.read_csv(odds_file)
    except Exception as exc:st.error(str(exc))

odds=st.session_state.get('odds')
if odds is not None and not odds.empty:
    st.subheader('DraftKings market lines')
    st.dataframe(odds,hide_index=True,use_container_width=True)
    st.download_button('Download odds snapshot',odds.to_csv(index=False),'draftkings_odds_snapshot.csv','text/csv')

if (historical and model_file or ('history_auto' in st.session_state and 'model_auto' in st.session_state)) and odds is not None and not odds.empty:
    try:
        games=validate(pd.read_csv(historical)) if historical and model_file else st.session_state['history_auto']
        model=load(model_file) if historical and model_file else st.session_state['model_auto']
        upcoming=odds[['date','home_team','away_team','home_spread']].copy()
        upcoming['season']=pd.to_datetime(upcoming.date,utc=True).dt.year
        upcoming['date']=pd.to_datetime(upcoming.date,utc=True).dt.tz_localize(None)
        upcoming['neutral_site']=0 # Supply neutral-site flags separately for accurate predictions.
        future=validate(upcoming,training=False)
        # Guard against historical outcomes after the game being predicted.
        if future.date.min() <= games.date.max():
            st.error('Upcoming games overlap or predate the last historical result. Use a strictly pregame history snapshot.')
            st.stop()
        X=build_features(games,future)
        pred=predictions(model,X,future)
        # For API timestamps use the date component consistently; do not mix local calendar dates.
        ranked=analyze(pred,odds)
        if ranked.empty:
            st.error('No matching games. Check exact team names, game dates, and spreads.')
        else:
            st.subheader('Model vs. DraftKings — research ranking')
            st.caption('Probabilities approximate and uncalibrated; do not treat ranking as proven betting edge.')
            st.dataframe(ranked,hide_index=True,use_container_width=True)
            st.download_button('Download matchup analysis',ranked.to_csv(index=False),'weekly_analysis.csv','text/csv')
            st.subheader('Hypothetical 2- and 3-leg combinations')
            combos=parlays(ranked,min_edge=min_edge)
            if combos.empty:st.info('No combinations pass the selected experimental edge threshold.')
            else:st.dataframe(combos.head(30),hide_index=True,use_container_width=True)
            st.caption('Parlay calculations assume independent games and ignore correlated outcomes, pushes, market movement, and model miscalibration.')
    except Exception as exc:st.error(f'Analysis failed: {exc}')
else:
    st.info('Upload completed-game history and a trained model, then fetch or upload DraftKings odds to analyze the slate.')
    st.markdown('**Weekly workflow:** refresh completed games → retrain/backtest outside the dashboard → fetch DraftKings lines → review matchups → export report. Verify injury and lineup news manually.')
