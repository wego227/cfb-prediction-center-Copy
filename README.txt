CFB Prediction Center Version 7.0

UPLOAD ALL FIVE FILES to your existing GitHub repository root, replacing matches:
 dashboard.py, market_model.py, v5_features.py, cfbd_data.py, v7_advanced.py
Keep predictor.py, odds.py, analysis.py, requirements.txt unchanged.

Streamlit secrets: CFBD_API_KEY must already be configured. Never upload it to GitHub.
Reboot Streamlit after committing. Run 'Download history and train model'.
Advanced mode requests 16 regular-season weeks for each of 2019-2025 (112 additional calls). API usage/quota and endpoint coverage may vary. The dashboard reports request failures and nonmissing metric counts. If access or quota fails, uncheck advanced mode to train a score-only comparison.

Data leakage protection: game-level advanced metrics enter each team's historical ledger only AFTER that game. Games with the same kickoff timestamp do not learn from one another. Season-wide aggregates are deliberately not used for historical game features.

LIMITATIONS:
- Advanced features are rolling team averages from previous games, not fully opponent-adjusted EPA. 'PPA' is CFBD's expected-points-derived metric.
- If a game or team is absent from the advanced API response, missing feature averages default to zero; check coverage carefully.
- The current historical download stops in 2025. For 2026 upcoming predictions, first extend the completed-game history and advanced stats through the latest completed games. Otherwise current-season advanced features are absent and predictions are unreliable.
- Historical lines may not be synchronized to a common pre-kickoff timestamp. Historical ROI is not executable returns.
- 2025 has been repeatedly inspected and is a development benchmark, NOT a fresh independent test.
- A saved model alone does not contain the advanced history needed to reconstruct upcoming features; retrain in the current Streamlit session before using advanced-model predictions.
- No betting edge has been demonstrated.
