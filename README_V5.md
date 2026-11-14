# CFB Prediction Center V5 — install and limitations

Upload **dashboard.py**, **market_model.py**, and **v5_features.py** into the root of your existing GitHub repository, replacing the first two. Leave existing `predictor.py`, `cfbd_data.py`, `odds.py`, `analysis.py`, `requirements.txt` intact. This package does not contain credentials.

## What actually changed
- Added pre-kickoff, score-derived features: recent/weighted scoring margin, opponent prior performance, prior-season margin, rest days, and schedule proxy. **This is NOT EPA/play or play-by-play success rate data.** Those require additional CFBD endpoints and separate validation.
- Compared Ridge and histogram gradient boosting on walk-forward 2022–2024 residual MAE, selected the better model before evaluating 2025.
- Estimated approximate probability dispersion using 2022–2024 out-of-sample residuals, never 2025.
- Kept the original V3 baseline display, and retained the same 2025 holdout and edge-threshold reports.

## Cautions
- The 2025 holdout has already been inspected during development of prior versions. Treat V5's 2025 comparison as diagnostic, not an untouched final test.
- Historical CFBD spreads can differ in provider and timestamp; they are not guaranteed to be pregame DraftKings closing lines.
- No demonstrated profitable betting edge. Future-game predictions need updated, completed history through the previous kickoff; the dashboard currently downloads only 2019–2025 and rejects overlapping future games.
- Probability values are rough normal approximations, not externally calibrated.
- If Streamlit dependencies are missing, confirm scikit-learn, numpy, pandas, scipy, streamlit and joblib are in requirements.txt.

## Local validation
`python -m compileall dashboard.py market_model.py v5_features.py`

The included synthetic-data smoke test was run separately during development. Real CFBD/Streamlit integration has not been verified in this environment.
