CFB Prediction Center V6 — diagnostic update

Upload dashboard.py, market_model.py, and v5_features.py together to the GitHub repository root. Keep existing predictor.py, cfbd_data.py, analysis.py, odds.py, and requirements.txt unchanged. Reboot Streamlit after commit.

What's new: Ridge regularization grid (alpha 1/10/100/300/1000), boosting candidate, 2022–24 walk-forward selection, sportsbook baseline margin MAE, residual variation/correlation, 2025 provider breakdown, and 1/2/3/5-point edge counts. No extra CFBD API endpoints are used. No true EPA or success rate data is included.

Important: Historical CFBD lines may reflect differing snapshot times; results cannot be interpreted as executable historical DraftKings returns. 2025 has already been inspected, so it is a diagnostic benchmark, not an untouched independent holdout. Current historical downloader only covers through 2025; future 2026 games require updated historical data and strict pregame snapshots.

This is research software. No profitable wagering edge has been established.
