"""Independent audit page: does not modify the existing prediction dashboard."""
import streamlit as st
import pandas as pd
from cfbd_data import download_history
from v7_advanced import download_advanced
from v8_audit import audit_coverage, audit_feature_values, ablation

st.set_page_config(page_title='CFB V8 Data Audit',layout='wide')
st.title('CFB Version 8 — Advanced Data Audit')
st.caption('Data quality and identical-game ablation tests. This page does not change your trained prediction model.')
st.warning('2025 has been inspected repeatedly and is a diagnostic benchmark, not an independent holdout. Historical sportsbook lines may not reflect consistent pregame snapshots.')

try: key=st.secrets.get('CFBD_API_KEY','')
except Exception: key=''

if 'history_auto' in st.session_state and 'advanced_auto' in st.session_state and st.session_state['advanced_auto'] is not None:
    st.success('Using historical games and advanced statistics already downloaded in this Streamlit session.')
    history=st.session_state['history_auto']
    advanced=st.session_state['advanced_auto']
else:
    st.info('No V7 download found in this session. You can run Version 7 training first, or download the data here.')
    if st.button('Download 2019–2025 data for audit',disabled=not bool(key)):
        try:
            status=st.empty()
            with st.spinner('Downloading game and advanced data…'):
                history,report=download_history(key,2019,2025,progress=lambda msg:status.info(msg))
                advanced,adv_report=download_advanced(key,range(2019,2026),progress=lambda msg:status.info(msg))
                st.session_state['v8_history']=history
                st.session_state['v8_advanced']=advanced
                st.session_state['v8_report']={'history':report,'advanced':adv_report}
            status.success('Data downloaded for audit.')
        except Exception as exc: st.error(f'Download failed: {exc}')
    history=st.session_state.get('v8_history')
    advanced=st.session_state.get('v8_advanced')

if history is not None and advanced is not None:
    try:
        coverage,joined,summary=audit_coverage(history,advanced)
        feature_report,features,smoke_test=audit_feature_values(history,advanced)
        comparison=ablation(history,advanced)
        st.subheader('1. Historical game coverage')
        st.json(summary)
        st.dataframe(coverage,hide_index=True,use_container_width=True)
        st.download_button('Download coverage audit',coverage.to_csv(index=False),'v8_coverage.csv','text/csv')
        st.subheader('2. Pregame feature diagnostics')
        st.dataframe(feature_report,hide_index=True,use_container_width=True)
        st.download_button('Download feature diagnostics',feature_report.to_csv(index=False),'v8_feature_diagnostics.csv','text/csv')
        if smoke_test: st.success('Future-result exclusion smoke test passed.')
        elif smoke_test is False: st.error('Future-result exclusion smoke test failed; investigate possible leakage.')
        st.caption('Smoke test is limited; it does not prove all features are free from temporal leakage.')
        st.subheader('3. Same-game model ablation')
        st.caption('Fixed Ridge alpha=1000 for both models; each season is trained on prior seasons only. Positive advanced-minus-score MAE means advanced features hurt.')
        st.dataframe(comparison,hide_index=True,use_container_width=True)
        st.download_button('Download model ablation',comparison.to_csv(index=False),'v8_ablation.csv','text/csv')
        st.caption('Missing advanced values are encoded as zero by the existing V7 pipeline. High zero rates may conceal missingness; interpret ablation accordingly.')
    except Exception as exc: st.error(f'Audit failed: {exc}')
