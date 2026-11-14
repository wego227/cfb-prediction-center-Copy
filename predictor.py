"""Leakage-aware FBS football winner, margin, and spread predictor.

Input: one row per completed game, sorted internally by date. Columns:
date,season,home_team,away_team,home_score,away_score,home_spread
Optional: neutral_site (0/1), home_odds, away_odds.
Spread convention: -7 means the home team is favored by 7.
Future rows may omit scores and should be passed to the predict command.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, brier_score_loss, log_loss

STAT_NAMES = ['games','win_pct','avg_margin','avg_points_for','avg_points_against','avg_margin_opp_adjusted','recent_margin','home_win_pct']
FEATURES = [f'diff_{s}' for s in STAT_NAMES] + ['neutral_site','home_spread','season_week']

def validate(df, training=True):
    required = ['date','season','home_team','away_team','home_spread']
    if training: required += ['home_score','away_score']
    missing = sorted(set(required)-set(df.columns))
    if missing: raise ValueError(f'Missing columns: {missing}')
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    for c in ['home_spread','season'] + (['home_score','away_score'] if training else []):
        df[c] = pd.to_numeric(df[c],errors='coerce')
    if 'neutral_site' not in df: df['neutral_site'] = 0
    df['neutral_site'] = df['neutral_site'].fillna(0).astype(int)
    if df[['home_spread','season']].isna().any().any(): raise ValueError('Missing spreads or seasons')
    if training and df[['home_score','away_score']].isna().any().any(): raise ValueError('Missing game scores')
    return df.sort_values(['date','home_team','away_team']).reset_index(drop=True)

def team_stats(history, team, season):
    games = [g for g in history if g['season']==season and team in (g['home_team'],g['away_team'])]
    if not games: return np.zeros(len(STAT_NAMES))
    margins=[]; points_for=[]; points_against=[]; home_wins=[]; adjusted=[]
    for g in games:
        home = g['home_team']==team
        margin = (g['home_score']-g['away_score'])*(1 if home else -1)
        opponent = g['away_team'] if home else g['home_team']
        # Prior opponent rating, recorded at the time the game was played.
        opponent_prior = g.get('prior_away_rating',0) if home else g.get('prior_home_rating',0)
        margins.append(margin)
        points_for.append(g['home_score'] if home else g['away_score'])
        points_against.append(g['away_score'] if home else g['home_score'])
        adjusted.append(margin+opponent_prior)
        if home: home_wins.append(float(margin>0))
    return np.array([len(games),np.mean(np.array(margins)>0),np.mean(margins),
        np.mean(points_for),np.mean(points_against),np.mean(adjusted),
        np.mean(margins[-3:]),np.mean(home_wins) if home_wins else .5],dtype=float)

def build_features(games, future=None):
    """Sequential pregame features. Same-day games don't learn from each other."""
    history=[]; rows=[]; meta=[]
    dates=sorted(games['date'].unique())
    for date in dates:
        day=games.loc[games['date']==date]
        for _,g in day.iterrows():
            h=team_stats(history,g.home_team,g.season)
            a=team_stats(history,g.away_team,g.season)
            feat={f'diff_{n}':float(h[i]-a[i]) for i,n in enumerate(STAT_NAMES)}
            feat.update(neutral_site=int(g.neutral_site),home_spread=float(g.home_spread),season_week=max(0,int((g.date-pd.Timestamp(int(g.season),8,1)).days//7)))
            rows.append(feat)
            meta.append(g.to_dict())
        for _,g in day.iterrows():
            d=g.to_dict()
            # Strength-adjusted rating using only games that preceded this date.
            d['prior_home_rating']=float(team_stats(history,g.home_team,g.season)[2])
            d['prior_away_rating']=float(team_stats(history,g.away_team,g.season)[2])
            history.append(d)
    if future is not None:
        future_rows=[]
        for _,g in future.iterrows():
            past=[x for x in history if pd.Timestamp(x['date']) < g.date]
            h=team_stats(past,g.home_team,g.season)
            a=team_stats(past,g.away_team,g.season)
            f={f'diff_{n}':float(h[i]-a[i]) for i,n in enumerate(STAT_NAMES)}
            f.update(neutral_site=int(g.neutral_site),home_spread=float(g.home_spread),season_week=max(0,int((g.date-pd.Timestamp(int(g.season),8,1)).days//7)))
            future_rows.append(f)
        return pd.DataFrame(future_rows,columns=FEATURES)
    result=pd.DataFrame(rows,columns=FEATURES)
    details=pd.DataFrame(meta)
    return result,details

def fit_models(X, details):
    y_margin=(details.home_score-details.away_score).to_numpy()
    y_win=(y_margin>0).astype(int)
    if len(np.unique(y_win))<2: raise ValueError('Need games won by both home and away teams')
    win=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=0.5))
    margin=HistGradientBoostingRegressor(max_iter=130,max_leaf_nodes=12,l2_regularization=10,learning_rate=.055,random_state=42)
    win.fit(X,y_win); margin.fit(X,y_margin)
    # Use training residuals as a fallback dispersion estimate. This is not calibrated.
    sigma=max(7.,float(np.std(y_margin-margin.predict(X),ddof=1)))
    return {'win':win,'margin':margin,'sigma':sigma,'features':FEATURES}

def normal_cdf(x):
    from math import erf,sqrt
    return np.array([.5*(1+erf(float(z)/sqrt(2))) for z in np.asarray(x).ravel()])

def predictions(model,X,details):
    p=model['win'].predict_proba(X)[:,1]
    m=model['margin'].predict(X)
    # Home covers when home_margin + home_spread > 0.
    cover=normal_cdf((m+details.home_spread.to_numpy())/model['sigma'])
    out=details[['date','home_team','away_team','home_spread']].copy().reset_index(drop=True)
    out['projected_home_margin']=np.round(m,2)
    out['home_win_probability']=np.round(p,4)
    out['home_cover_probability_approx']=np.round(cover,4)
    out['away_cover_probability_approx']=np.round(1-cover,4)
    out['model_vs_market_points']=np.round(m+out.home_spread,2)
    return out

def evaluate(model,X,details):
    out=predictions(model,X,details)
    actual=(details.home_score-details.away_score).to_numpy()
    win=(actual>0).astype(int)
    spread=details.home_spread.to_numpy()
    covered=actual+spread>0
    nonpush=actual+spread!=0
    pick=out.home_cover_probability_approx.to_numpy()>.5
    results={'games':len(actual),'winner_accuracy':round(float(accuracy_score(win,out.home_win_probability>=.5)),4),
      'winner_brier':round(float(brier_score_loss(win,out.home_win_probability)),4),
      'margin_MAE':round(float(mean_absolute_error(actual,out.projected_home_margin)),2),
      'ATS_accuracy_excluding_pushes':round(float(np.mean(pick[nonpush]==covered[nonpush])),4) if nonpush.any() else None,
      'ATS_graded_games':int(nonpush.sum())}
    # At standard -110 odds, a correct bet wins 100/110 units per unit risked.
    if nonpush.any():
        correct=pick[nonpush]==covered[nonpush]
        results['flat_bet_ROI_at_minus_110']=round(float(np.mean(np.where(correct,100/110,-1))),4)
    return results,out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    train=sub.add_parser('train');train.add_argument('--games',required=True);train.add_argument('--out',default='cfb_model.joblib');train.add_argument('--test-season',type=int,required=True)
    predict=sub.add_parser('predict');predict.add_argument('--games',required=True);predict.add_argument('--upcoming',required=True);predict.add_argument('--model',default='cfb_model.joblib');predict.add_argument('--out',default='predictions.csv')
    args=ap.parse_args()
    games=validate(pd.read_csv(args.games))
    if args.cmd=='train':
        X,d=build_features(games)
        tr=d.season<args.test_season; te=d.season==args.test_season
        if tr.sum()<100 or te.sum()<10: raise ValueError('Need >=100 prior-season training games and >=10 test-season games')
        model=fit_models(X.loc[tr],d.loc[tr]); metrics,out=evaluate(model,X.loc[te],d.loc[te])
        print('Held-out season:',args.test_season,'\n',metrics)
        out.to_csv('backtest_predictions.csv',index=False)
        # Refit on all available seasons for future predictions.
        final=fit_models(X,d)
        dump(final,args.out)
        print('Saved',args.out,'and backtest_predictions.csv')
    else:
        future=validate(pd.read_csv(args.upcoming),training=False)
        if future.date.min()<=games.date.max():
            print('WARNING: future dates overlap historical data; verify scores are strictly pregame.')
        X=build_features(games,future)
        model=load(args.model)
        out=predictions(model,X,future)
        out.to_csv(args.out,index=False)
        print(out.to_string(index=False));print('Saved',args.out)

if __name__=='__main__': main()
