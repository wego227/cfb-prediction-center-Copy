# FBS prediction model — prototype v1

This is an executable baseline, **not a trained or validated model**. No historical results or current odds are bundled. Predictions require your own historical game CSV and the model must be trained first.

## Setup

```bash
pip install -r requirements.txt
python predictor.py train --games historical_games.csv --test-season 2025
python predictor.py predict --games historical_games.csv --upcoming upcoming_games.csv --model cfb_model.joblib --out predictions.csv
```

## Input CSVs

`historical_games.csv` must have these columns:

`date,season,home_team,away_team,home_score,away_score,home_spread,neutral_site`

Example (illustrative only; not a real game):

`2024-09-07,2024,Example State,Example Tech,28,21,-3.5,0`

`upcoming_games.csv` must have:

`date,season,home_team,away_team,home_spread,neutral_site`

Example:

`2026-11-21,2026,Example Tech,Example State,2.5,0`

**Spread sign:** home spread is negative when the home team is favored. A home team covers if `home_score - away_score + home_spread > 0`. Pushes are excluded from ATS accuracy.

## Important limitations

- This is a starter model using prior results, prior opponent strength, recent margin, venue and the pregame spread; it does not yet ingest advanced efficiency, player injuries, or play-by-play data.
- `avg_margin_opp_adjusted` is a simple historical opponent-strength proxy, not a converged opponent-adjusted efficiency rating.
- The point-margin model uses the betting line as an input; its ATS results should be compared against a spread-only baseline. The win classifier also uses the spread.
- Cover probabilities use a Gaussian residual approximation with *training* residual variance and are **not calibrated**. Do not treat them as trustworthy wagering probabilities until validated on multiple later seasons.
- The script reports a single held-out season and then refits on all data. For serious use, add rolling-origin evaluation, probability calibration, line timestamps, and a strict no-lookahead data pipeline.
- Same-day games do not use each other's results. Different seasons reset team aggregates. Preseason team strength is therefore not yet modeled.
- Projected scores are not yet modeled: the code predicts margin only.
- Only use trustworthy **pregame** spreads; closing lines unavailable at prediction time must never be inserted into historical simulated predictions.
- `joblib` models must only be loaded from trusted sources.

## Version 2: DraftKings comparison dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

The dashboard requires a **trained model** and **completed-game history CSV** from the training workflow above. For live lines, enter an API key from [The Odds API](https://the-odds-api.com/) (a third-party data provider that supports DraftKings bookmaker filtering). API access, coverage, quotas, and pricing are controlled by that provider; DraftKings data may be unavailable in some regions/times. Alternatively upload an odds CSV with columns `date,home_team,away_team,home_spread,home_spread_odds,away_spread_odds,event_id,line_updated_at`. Dates must be ISO timestamps. A historical backtest requires archived *pregame* lines; today's lines must never be used for old games.

**Data integrity:** The dashboard requires exact team-name and game-date matches, rejects mismatched spreads, and blocks historical data extending past the upcoming games. It currently assumes `neutral_site=0`; add venue data before using neutral-site games. Historical team names may need standardization before the join. The odds snapshot can be downloaded. No data are bundled and no live odds are claimed in the repository.

**Critical limitation:** V1 model cover probabilities are uncalibrated Gaussian approximations. The model's market input makes its predictions market-dependent. The ranking and parlay calculations are exploratory only. Parlay calculations assume independence, which is not generally true; do not interpret naive EV as a verified advantage. Pushes and correlated parlays are not priced. No wager is placed. For a reliable system, add rolling-season validation, calibration, pregame injury inputs, archived line timestamps, team-name mapping, and independent held-out ROI analysis before acting on any edge.

**Weekly routine:** Update scores/history, run `predictor.py train` with a completed test season, inspect `backtest_predictions.csv` and validation metrics, then launch the dashboard, fetch DraftKings odds, and export `weekly_analysis.csv`. For automation, host the dashboard and schedule a separate data-refresh process; ChatGPT cannot access a private local model automatically.
