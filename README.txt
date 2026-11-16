CFB Prediction Center — disk cache + prediction quality report

Install in GitHub repository:
1. Keep cfb_team_matching.py in repository ROOT (included here for convenience; unchanged).
2. Replace pages/9_2026_Game_Predictions.py with the file in this ZIP.
3. Commit and reboot Streamlit.

What's new:
- Disk-backed Streamlit cache for 2019–2025 historical games, 2026 completed games, and advanced stats (separately).
- Clear CFBD cache button. Use this only when refreshing 2026 results; it clears all cached historical and advanced datasets and may trigger quota errors if used repeatedly.
- Prediction quality report with odds names, canonical CFBD names, matching methods, 2026 game counts and caution flags. Downloadable as CSV.
- CSV predictions also include home/away 2026 game counts and quality status.

IMPORTANT:
- Disk cache is per Streamlit deployment and may be lost on redeployment, reboot, or host storage cleanup. It does not guarantee quota avoidance across cold deployments.
- Streamlit disk caching ignores TTL; use manual clear for freshness. Do not clear repeatedly.
- A failed first advanced-stats download is not cached. If CFBD returns 429, uncheck advanced stats and retry once; wait for quota reset before requesting advanced data again.
- Cached 2026 games stay frozen until manual clear. Never treat stale cached data as up to date.
- Does not change model training, odds, or prediction formulas. Not proof of ATS edge.
