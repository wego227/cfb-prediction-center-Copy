# CFB Prediction Center V4 — market-relative research baseline

This version adds a ridge regression model for **actual home margin minus market-implied home margin**. It predicts the residual `actual_home_margin + home_spread` rather than directly predicting raw score difference. Historical team features are built sequentially from games completed before each kickoff date.

## Deploy

Replace the existing root-level repository files with the contents of this ZIP (do **not** upload the ZIP itself). Keep the existing Streamlit app pointing to `dashboard.py`. New file: `market_model.py`. Streamlit secret `CFBD_API_KEY` remains unchanged. The Odds API key is separate (`THE_ODDS_API_KEY`). No secrets belong in GitHub.

After redeployment click **Download history and train model**. Inspect the V4 season-forward evaluation, then download the trained model. The model is kept in the browser session until the app restarts; re-run training or upload the downloaded model after a restart.

## Validation design

- 2022, 2023, 2024: expanding-window predictions trained on strictly earlier seasons. The 2022-24 out-of-fold errors estimate uncertainty.
- 2025: held-out evaluation; trained on 2019-2024 and uncertainty estimated from 2022-24 only.
- Final model: trained on all available downloaded seasons for *future* predictions.
- Edge threshold results at 2/3/5 points are **descriptive**; choosing a threshold based on 2025 and claiming 2025 as untouched would be data snooping.
- ATS ROI assumes -110 on every wager, excludes pushes and does not reflect actual historical prices.

## Known limitations

This is **not yet the advanced EPA/success-rate model**. CFBD's games and lines endpoints do not provide those features directly. Adding those requires separate season/week-level advanced-stat endpoints, stable historical coverage, and careful pregame time alignment. This version does not silently invent them. The historical spread provider can vary if DraftKings is unavailable. Historical spread timestamps may not correspond to bettable closing lines. Current team names from The Odds API can differ from CFBD names; no unsafe fuzzy matching is attempted. The current dashboard has a manual retrain step and cannot yet produce automated Friday reports. The approximate normal residual distribution is not a calibrated betting probability. Parlays assume independence and should not be treated as verified value.

**No guarantee of profit. Do not rely on experimental rankings to wager.**
