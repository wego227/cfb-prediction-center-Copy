"""Market-relative model with season-forward validation and out-of-fold uncertainty.

This is a research baseline, NOT proof of a wagering edge. Uses only pregame
features from predictor.build_features. Market spreads are required.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, accuracy_score, brier_score_loss
from scipy.special import ndtr

# Keep residual features intentionally modest to limit overfitting.
RESIDUAL_FEATURES = ['diff_games','diff_win_pct','diff_avg_margin',
                     'diff_avg_margin_opp_adjusted','diff_recent_margin',
                     'diff_avg_points_for','diff_avg_points_against',
                     'neutral_site','season_week']

def _fit(X, d):
    y = (d.home_score - d.away_score + d.home_spread).to_numpy(dtype=float)
    model = make_pipeline(StandardScaler(), Ridge(alpha=500.0))
    model.fit(X[RESIDUAL_FEATURES], y)
    return model

def walk_forward(X, details, test_seasons=(2022,2023,2024,2025)):
    """Each test season uses strictly earlier seasons. No random split."""
    frames=[]
    for season in test_seasons:
        train=details.season < season
        test=details.season == season
        if train.sum()<100 or test.sum()<10: continue
        fitted=_fit(X.loc[train], details.loc[train])
        sub=details.loc[test].copy()
        sub['pred_residual']=fitted.predict(X.loc[test, RESIDUAL_FEATURES])
        sub['actual_residual']=sub.home_score-sub.away_score+sub.home_spread
        sub['eval_season']=season
        frames.append(sub)
    if not frames: raise ValueError('Not enough games for season-forward evaluation')
    return pd.concat(frames,ignore_index=True)

def _dispersion(cal):
    # Dispersion estimated ONLY from past out-of-sample seasons.
    errors=(cal.actual_residual-cal.pred_residual).to_numpy(dtype=float)
    return max(7.0,float(np.std(errors,ddof=1))) if len(errors)>1 else 14.0

def _report(frame, sigma):
    margin=(frame.home_score-frame.away_score).to_numpy(dtype=float)
    spread=frame.home_spread.to_numpy(dtype=float)
    proj=-spread+frame.pred_residual.to_numpy(dtype=float)
    nonpush=(margin+spread)!=0
    cover=(margin+spread)>0
    prob=ndtr(frame.pred_residual.to_numpy(dtype=float)/sigma)
    pick=prob>.5
    result={'games':len(frame),
        'winner_accuracy_market_relative':round(float(np.mean((proj>0)==(margin>0))),4),
        'margin_MAE':round(float(mean_absolute_error(margin,proj)),2),
        'ATS_accuracy_all_nonpush':round(float(np.mean(pick[nonpush]==cover[nonpush])),4) if nonpush.any() else None,
        'ATS_graded_games':int(nonpush.sum()),
        'cover_brier_nonpush':round(float(brier_score_loss(cover[nonpush].astype(int),prob[nonpush])),4) if nonpush.any() else None,
        'probability_sigma':round(float(sigma),2)}
    # Fixed thresholds are diagnostic only. Do not select thresholds based on 2025 results.
    for threshold in (2.0,3.0,5.0):
        chosen=nonpush & (np.abs(frame.pred_residual.to_numpy(dtype=float))>=threshold)
        n=int(chosen.sum())
        result[f'ATS_at_{threshold:g}pt_edge']={'bets':n,
            'accuracy':round(float(np.mean(pick[chosen]==cover[chosen])),4) if n else None,
            'ROI_flat_minus110':round(float(np.mean(np.where(pick[chosen]==cover[chosen],100/110,-1))),4) if n else None}
    return result

def train_and_evaluate(X, details):
    years=sorted(int(y) for y in details.season.unique())
    if 2025 not in years: raise ValueError('2025 is required as the held-out evaluation season')
    # OOF calibration uses 2022-2024 ONLY; 2025 never informs fit or sigma.
    cal=walk_forward(X,details,(2022,2023,2024))
    sigma=_dispersion(cal)
    test=walk_forward(X,details,(2025,))
    per_year=[]
    for y in (2022,2023,2024):
        earlier=cal[cal.eval_season<y]
        if len(earlier)>=20:
            per_year.append({'season':y,**_report(cal[cal.eval_season==y],_dispersion(earlier))})
    per_year.append({'season':2025,**_report(test,sigma)})
    # Full-history refit is for NEXT games only, never for the 2025 evaluation.
    final=_fit(X,details)
    return {'type':'market_residual_v4','residual_model':final,'sigma':sigma,
            'features':RESIDUAL_FEATURES,'last_training_date':str(details.date.max())},per_year,test

def predict_upcoming(model,X,upcoming):
    residual=model['residual_model'].predict(X[RESIDUAL_FEATURES])
    sigma=model['sigma']
    p=ndtr(residual/sigma)
    out=upcoming[['date','home_team','away_team','home_spread']].copy().reset_index(drop=True)
    out['projected_home_margin']=np.round(-out.home_spread.to_numpy()+residual,2)
    out['home_cover_probability_approx']=np.round(p,4)
    out['away_cover_probability_approx']=np.round(1-p,4)
    out['model_vs_market_points']=np.round(residual,2)
    return out
