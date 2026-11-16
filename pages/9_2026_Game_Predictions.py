"""Pregame 2026 predictions using the existing V7 model and fresh 2026 results."""
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import pandas as pd
import streamlit as st
from cfbd_data import download_history
from v7_advanced import download_advanced, build_advanced_features, ADV_FEATURES
from v5_features import build_v5_features
from market_model import train_and_evaluate, predict_upcoming
from odds import fetch_draftkings
from predictor import validate
from cfb_team_matching import match_team
from prediction_tracker import snapshot

# Disk-backed Streamlit caching prevents repeated multi-year CFBD downloads
# on ordinary widget reruns and app restarts. Manually clear when data needs refreshing.
@st.cache_data(persist="disk", show_spinner=False)
def cached_history(api_key, start_year, end_year):
    return download_history(api_key, start_year, end_year)

@st.cache_data(persist="disk", show_spinner=False)
def cached_advanced(api_key, years):
    return download_advanced(api_key, list(years))

# The active season refreshes every six hours; past seasons stay cached.
@st.cache_data(ttl=21600, show_spinner=False)
def current_season_history(api_key, year):
    return download_history(api_key, year, year)

@st.cache_data(ttl=21600, show_spinner=False)
def current_season_advanced(api_key, year):
    return download_advanced(api_key, [year])

st.title("🏈 2026 Game Predictions")
st.warning("Experimental projections, not validated betting advice. Historical 2025 has been repeatedly examined.")
st.caption("Uses 2019–2025 to fit the V7 model; uses completed 2026 games only for pregame team features. No 2026 outcome is used to fit the model.")
try:
    key=st.secrets.get("CFBD_API_KEY","")
except Exception:
    key=""
if not key:
    st.error("Set CFBD_API_KEY in Streamlit secrets.")
    st.stop()

today=datetime.now(ZoneInfo("America/New_York")).date()
date=st.date_input("Games scheduled for (Eastern Time)", value=today)
use_advanced=st.checkbox("Include V7 advanced statistics",value=True)
st.caption("2019–2025 history uses a persistent disk cache. Active-season games and advanced stats refresh after six hours; you can force a refresh below.")
if st.button("Clear CFBD cache (only when updating data)"):
    cached_history.clear()
    cached_advanced.clear()
    current_season_history.clear()
    current_season_advanced.clear()
    st.success("CFBD cache cleared. The next prediction run will download data again and consume API quota.")

st.subheader("1. Load 2026 odds")
source=st.radio("Odds source",["DraftKings via The Odds API","Upload odds CSV"],horizontal=True)
odds=None
if source.startswith("DraftKings"):
    try:
        default_key=st.secrets.get("THE_ODDS_API_KEY","") or os.getenv("THE_ODDS_API_KEY","")
    except Exception:
        default_key=os.getenv("THE_ODDS_API_KEY","")
    odds_key=st.text_input("The Odds API key",value=default_key,type="password")
    if st.button("Fetch DraftKings odds",disabled=not bool(odds_key)):
        try:
            st.session_state["today_odds"]=fetch_draftkings(odds_key)
        except Exception as exc:
            st.error(f"Odds download failed: {exc}")
else:
    uploaded=st.file_uploader("Upload odds CSV with date, home_team, away_team, home_spread",type="csv")
    if uploaded:
        st.session_state["today_odds"]=pd.read_csv(uploaded)

raw=st.session_state.get("today_odds")
if raw is not None and not raw.empty:
    try:
        required={"date","home_team","away_team","home_spread"}
        if not required.issubset(raw.columns):
            raise ValueError(f"Odds missing columns: {sorted(required-set(raw.columns))}")
        odds=raw.copy()
        odds["_kickoff_utc"]=pd.to_datetime(odds["date"],utc=True,errors="coerce")
        if odds["_kickoff_utc"].isna().any():
            raise ValueError("Invalid kickoff timestamps in odds.")
        odds["_date_et"]=odds["_kickoff_utc"].dt.tz_convert("America/New_York").dt.date
        odds=odds.loc[odds["_date_et"]==date].copy()
        now=pd.Timestamp.now(tz="UTC")
        odds=odds.loc[odds["_kickoff_utc"]>now].copy()
        odds["home_spread"]=pd.to_numeric(odds["home_spread"],errors="coerce")
        odds=odds.dropna(subset=["home_spread"])
        st.write(f"{len(odds)} not-yet-started games for {date} (Eastern)")
        st.dataframe(odds[["date","home_team","away_team","home_spread"]],hide_index=True)
    except Exception as exc:
        st.error(f"Odds format error: {exc}")
        odds=None
else:
    st.info("Fetch or upload today's odds first. Only future kickoffs can be predicted.")

st.subheader("2. Build pregame predictions")
if st.button("Download 2026 results and generate predictions",disabled=odds is None or odds.empty):
    try:
        status=st.empty()
        cutoff=odds["_kickoff_utc"].min().tz_localize(None)
        with st.spinner("Downloading historical data, updating 2026 team statistics, and fitting V7…"):
            # Historical model fitting is restricted to 2019–2025.
            old,_=cached_history(key,2019,2025)
            current,_=current_season_history(key,2026)
            old=validate(old)
            current=validate(current)
            # Never include a result at or after the earliest prediction kickoff.
            current=current.loc[current.date<cutoff].copy()
            if current.empty:
                st.warning("No eligible 2026 completed games with historical lines were found.")
            history=pd.concat([old,current],ignore_index=True).sort_values("date").reset_index(drop=True)
            x_old,d_old=build_v5_features(old)
            if use_advanced:
                adv_old,_=cached_advanced(key,tuple(range(2019,2026)))
                adv_2026,_=current_season_advanced(key,2026)
                # Restrict 2026 advanced rows to known, completed, eligible games.
                adv_2026=adv_2026.loc[adv_2026.game_id.astype(str).isin(current.game_id.astype(str))].copy()
                advanced=pd.concat([adv_old,adv_2026],ignore_index=True)
                x_old=pd.concat([x_old.reset_index(drop=True),
                    build_advanced_features(d_old,adv_old).reset_index(drop=True)],axis=1)
            else:
                advanced=None
            model,_,_,_,_=train_and_evaluate(x_old,d_old,features=list(x_old.columns))
            # Map sportsbook display names to the exact CFBD names used in feature ledgers.
            canonical=pd.unique(pd.concat([history.home_team,history.away_team],ignore_index=True)).tolist()
            names=pd.unique(pd.concat([odds.home_team,odds.away_team],ignore_index=True)).tolist()
            matches={name:match_team(name,canonical) for name in names}
            match_report=pd.DataFrame([
                {"sportsbook_team":name,"cfbd_team":matches[name][0],
                 "match_method":matches[name][1],
                 "suggestions_only":", ".join(matches[name][2])}
                for name in names
            ])
            st.subheader("Team-name matching diagnostics")
            st.dataframe(match_report,hide_index=True,use_container_width=True)
            st.download_button("Download team-name matching report",
                match_report.to_csv(index=False),"team_matching_report.csv","text/csv")
            unmatched=match_report[match_report.cfbd_team.isna()]
            if not unmatched.empty:
                raise ValueError(
                    f"{len(unmatched)} sportsbook team names could not be matched safely. "
                    "Download the matching report and share it; predictions were BLOCKED to avoid false edges."
                )
            # Require actual 2026 team history rather than all-zero fallback.
            team_counts=pd.concat([current.home_team,current.away_team]).value_counts()
            upcoming=odds[["date","home_team","away_team","home_spread"]].copy()
            upcoming["odds_home_team"]=upcoming["home_team"]
            upcoming["odds_away_team"]=upcoming["away_team"]
            upcoming["home_team"]=upcoming["home_team"].map(lambda n:matches[n][0])
            upcoming["away_team"]=upcoming["away_team"].map(lambda n:matches[n][0])
            upcoming["home_2026_games"]=upcoming.home_team.map(team_counts).fillna(0).astype(int)
            upcoming["away_2026_games"]=upcoming.away_team.map(team_counts).fillna(0).astype(int)
            # Keep a quality report available even if a subsequent validation blocks predictions.
            quality=upcoming[["odds_home_team","home_team","home_2026_games",
                "odds_away_team","away_team","away_2026_games"]].copy()
            quality["home_match_method"]=quality.odds_home_team.map(lambda n:matches[n][1])
            quality["away_match_method"]=quality.odds_away_team.map(lambda n:matches[n][1])
            quality["quality_status"]=quality.apply(lambda r:
                "BLOCKED: no 2026 history" if min(r.home_2026_games,r.away_2026_games)==0
                else "CAUTION: <3 games" if min(r.home_2026_games,r.away_2026_games)<3
                else "PASS: team history present",axis=1)
            quality["advanced_requested"]=use_advanced
            st.session_state["today_quality_report"]=quality
            st.session_state["today_quality_date"]=date
            st.subheader("2026 completed-game coverage")
            st.dataframe(upcoming[["odds_home_team","home_team","home_2026_games",
                "odds_away_team","away_team","away_2026_games"]],hide_index=True,
                use_container_width=True)
            st.download_button("Download game-history coverage",
                upcoming[["odds_home_team","home_team","home_2026_games",
                "odds_away_team","away_team","away_2026_games"]].to_csv(index=False),
                "team_2026_history_coverage.csv","text/csv")
            insufficient=upcoming[(upcoming.home_2026_games==0)|(upcoming.away_2026_games==0)]
            if not insufficient.empty:
                raise ValueError(
                    f"{len(insufficient)} games have a team with zero 2026 completed games "
                    "in the CFBD spread-filtered history. Predictions BLOCKED; check coverage."
                )
            upcoming=upcoming[["date","home_team","away_team","home_spread","season","neutral_site"]] if "season" in upcoming.columns else upcoming
            upcoming["date"]=pd.to_datetime(upcoming.date,utc=True).dt.tz_localize(None)
            upcoming["season"]=2026
            upcoming["neutral_site"]=0
            upcoming=validate(upcoming,training=False)
            if upcoming.date.min()<=history.date.max():
                raise ValueError("A completed game overlaps an upcoming kickoff; refresh odds or check timestamps.")
            features=build_v5_features(history,upcoming)
            if use_advanced:
                features=pd.concat([features.reset_index(drop=True),
                    build_advanced_features(history,advanced,upcoming).reset_index(drop=True)],axis=1)
            result=predict_upcoming(model,features,upcoming)
            # Keep CFBD canonical names for modeling, add original sportsbook labels for display.
            display_map={matches[n][0]:n for n in names}
            result["sportsbook_home_team"]=result.home_team.map(display_map)
            result["sportsbook_away_team"]=result.away_team.map(display_map)
            if len(result)>1 and result.model_vs_market_points.nunique()==1:
                st.warning("All model adjustments are identical. This is suspicious; "
                           "inspect the matching and game-history diagnostics before use.")
            result["market_favored_team"]=result.apply(
                lambda r:r.home_team if r.home_spread<0 else (r.away_team if r.home_spread>0 else "Pick'em"),axis=1)
            result["projected_winner"]=result.apply(
                lambda r:r.home_team if r.projected_home_margin>0 else r.away_team,axis=1)
            result["model_favored_ATS"]=result.apply(
                lambda r:r.home_team if r.model_vs_market_points>0 else (r.away_team if r.model_vs_market_points<0 else "No edge"),axis=1)
            result["2026_completed_games_used"]=len(current)
            # Model outputs and diagnostics share the same validated matchup ordering.
            coverage=quality.set_index(["home_team","away_team"])
            result["home_2026_games"]=result.apply(lambda r:int(coverage.loc[(r.home_team,r.away_team),"home_2026_games"]),axis=1)
            result["away_2026_games"]=result.apply(lambda r:int(coverage.loc[(r.home_team,r.away_team),"away_2026_games"]),axis=1)
            result["quality_status"]=result.apply(lambda r:"CAUTION: <3 games" if min(r.home_2026_games,r.away_2026_games)<3 else "PASS",axis=1)
            result["advanced_enabled"]=use_advanced
            st.session_state["today_predictions"]=result
            st.session_state["today_prediction_snapshot"]=snapshot(result)
            st.session_state["today_prediction_date"]=date
            status.success("Pregame predictions generated.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info("If team names differ between the odds feed and CFBD, predictions can silently use empty team histories. Check team-name mappings before interpreting results.")

if "today_predictions" in st.session_state:
    st.subheader("3. Predicted games")
    if st.session_state.get("today_prediction_date")!=date:
        st.warning("These predictions were generated for a different selected date.")
    result=st.session_state["today_predictions"]
    st.dataframe(result,hide_index=True,use_container_width=True)
    st.download_button("Download predictions CSV",result.to_csv(index=False),
        f"cfb_predictions_{st.session_state['today_prediction_date']}.csv","text/csv")
    if st.session_state.get("today_prediction_date")==date and "today_prediction_snapshot" in st.session_state:
        saved=st.session_state["today_prediction_snapshot"]
        st.download_button("Download timestamped pregame snapshot (for V10 tracker)",
            saved.to_csv(index=False).encode(),f"cfb_pregame_snapshot_{date}.csv","text/csv")
        st.caption("Save this file before kickoff. It preserves the original predictions and lines for later grading.")
    st.caption("Model edge is the predicted scoring-margin adjustment to the market, not a proven profitable edge. Neutral-site flags default to zero; verify these and exact team names.")

if st.session_state.get("today_quality_date")==date and "today_quality_report" in st.session_state:
    st.subheader("4. Prediction-quality report")
    quality=st.session_state["today_quality_report"]
    st.dataframe(quality,hide_index=True,use_container_width=True)
    st.download_button("Download prediction-quality report",quality.to_csv(index=False),
        f"cfb_prediction_quality_{date}.csv","text/csv")
    st.caption("Counts include only completed 2026 games with CFBD historical spreads before the earliest selected kickoff. This is a coverage check, not model validation. Historical model fit still uses 2019–2025.")
