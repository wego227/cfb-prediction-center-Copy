CFB VERSION 8: ADVANCED DATA AUDIT (non-destructive add-on)

1. Extract this ZIP.
2. In your GitHub repository, upload v8_audit.py to the repository ROOT.
3. Create a folder named pages if it does not exist.
4. Upload 8_Advanced_Data_Audit.py INSIDE pages/ (not in the root).
   On GitHub web UI: Add file > Create new file > enter
   pages/8_Advanced_Data_Audit.py and paste the supplied file contents,
   or use Add file > Upload files while browsing the pages folder.
5. Commit changes. Reboot Streamlit if necessary.
6. Use the sidebar page navigation to open 8 Advanced Data Audit.
7. If you have already trained V7 in this session, the audit will reuse its
   history and advanced data; otherwise press Download 2019-2025 data for audit.
8. Export and share v8_coverage.csv, v8_feature_diagnostics.csv,
   and v8_ablation.csv.

Keep existing dashboard.py, market_model.py, v5_features.py,
cfbd_data.py, and v7_advanced.py unchanged.

LIMITATIONS: This checks raw game/team coverage, pregame feature variation,
a limited future-result exclusion smoke test, and fixed Ridge ablation.
It does not prove all features are leakage-free, does not confirm that
historical spread snapshots are consistently pregame, and does not establish
any betting edge. 2025 has been inspected multiple times.
