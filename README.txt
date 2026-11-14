2026 Game Predictions — Team Name Matching Fix

INSTALL (GitHub web UI)
1. Extract ZIP.
2. Upload cfb_team_matching.py to the repository ROOT (next to dashboard.py).
3. Open the repository's pages folder and upload the replacement
   9_2026_Game_Predictions.py, overwriting the old version.
4. Commit both uploads and reboot Streamlit if necessary.
5. Open 2026 Game Predictions, fetch current DraftKings odds, and run predictions.

WHAT CHANGED
- Converts sportsbook names such as "Miami Hurricanes" to CFBD's "Miami"
  BEFORE constructing pregame features.
- Uses exact matches, explicit aliases, or uniquely identifiable team-name
  prefixes; never silently chooses an ambiguous fuzzy match.
- Shows a team matching report and each team's 2026 completed-game count.
- BLOCKS predictions when a team cannot be mapped or has zero eligible 2026
  completed games; this avoids reporting default-feature projections as real picks.
- Includes original sportsbook team names alongside canonical CFBD names.

IMPORTANT LIMITATIONS
- Not yet tested with live CFBD data or your exact 2026 team-name inventory.
  If blocked, share the matching report / screenshot so aliases can be fixed.
- Games without historical lines are omitted by cfbd_data.py, so game counts
  may understate actual games played.
- Neutral-site flags still default to zero. Check manually.
- 2025 was repeatedly inspected; no profitable betting edge has been shown.
- Refresh odds: the page only predicts games that have not started yet.
