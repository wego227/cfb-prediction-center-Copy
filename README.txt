CFB PREDICTION CENTER — VERSION 10

INSTALL IN YOUR EXISTING GITHUB REPOSITORY
1. Upload/replace cfb_team_matching.py in the ROOT folder.
2. Upload NEW prediction_tracker.py in the ROOT folder.
3. Replace pages/9_2026_Game_Predictions.py inside the pages folder.
4. Add pages/10_Prediction_Performance_Tracker.py inside the pages folder.
5. Commit changes and reboot/refresh Streamlit.
6. Do NOT upload __pycache__, .pyc files, or patch.py.

WHAT CHANGED
- Team matching: Southern Mississippi can only match the actual CFBD
  Southern Mississippi name; never the unrelated Southern team.
  If the canonical name is absent from history, that game is blocked.
- Cache: 2019–2025 history remains cached on disk; active 2026 history
  and advanced stats refresh every 6 hours (subject to CFBD quota).
- New predictions page downloads a timestamped pregame snapshot.
- Tracker page imports saved snapshots and grades them against completed
  scores from CFBD /games or a scores CSV. It reports winner accuracy,
  ATS accuracy (pushes excluded), margin MAE, and market baseline MAE,
  plus edge-size buckets.

IMPORTANT TRACKING RULES
- Save the snapshot before kickoff. A timestamp created after kickoff is
  marked unverified and excluded from the verified headline metrics.
- Keep your downloaded snapshot CSVs. Streamlit session storage and disk
  caches are not reliable long-term archives; the tracker does NOT save
  snapshots to GitHub or a database automatically.
- Past predictions CSVs without original capture timestamps are treated
  as unverified; they cannot be retroactively proven pregame.
- CFBD API /games is requested only when you click Fetch completed scores.
- Only regular-season FBS games are retrieved by the automated grader.
  For postseason games or missing results, upload a scores CSV instead.
- Kickoff-date matching uses UTC calendar dates and canonical team names.
  A game moved across UTC midnight may require manual adjustment.
- A 6-hour TTL is not a guaranteed refresh if the quota is exhausted.
- V7 advanced stats remain optional. A first download may still hit 429.

LIMITATIONS
- Historical 2025 evaluation has been inspected repeatedly, so it is not
  a clean untouched holdout. This update is infrastructure, not a claim
  of predictive or betting advantage.
- The predictor currently assumes neutral_site=0 for future games.
- The tracker calculates ATS wins for nonzero model edges; it does not
  compute financial returns or infer unavailable betting prices.
