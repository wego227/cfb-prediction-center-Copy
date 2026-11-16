CFB PREDICTION CENTER — VERSION 11

Upload exactly TWO files to your GitHub repository:

ROOT folder (same folder as dashboard.py):
  performance_v11.py                 [NEW]

pages folder:
  10_Prediction_Performance_Tracker.py [REPLACE]

Leave your existing prediction_tracker.py, cfb_team_matching.py,
9_2026_Game_Predictions.py, and all other files unchanged.
Do not upload __pycache__ or .pyc files.

After GitHub commits, refresh/reboot Streamlit, open Prediction Performance
Tracker, upload your existing cfb_pregame_snapshot_2026-11-21.csv under section 2.
You should see 53 unique archived predictions. Before November 21, no real
scores should be graded. Use the simulated score CSV ONLY to test grading,
then switch back to real CFBD results and do not archive simulated grading.

After the games, click Fetch completed scores. If CFBD returns HTTP 429,
wait for quota reset or upload a scores CSV.

IMPORTANT: This is portable FILE-BASED archiving, not automatic cloud persistence.
Keep your downloaded snapshots/ZIP locally. Snapshot timestamps are not
cryptographic proof; do not import postgame-generated predictions as pregame.
The model remains unchanged in V11; this is tracking infrastructure only.
