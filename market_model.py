"""V7 market-residual diagnostics. Model selection uses 2022-24 only.
2025 is a previously inspected benchmark, NOT an untouched test.
"""
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, brier_score_loss
from v5_features import FEATURES

MODEL_VERSION = '7.0'
CANDIDATES = ('ridge_1','ridge_10','ridge_100','ridge_300','ridge_1000','boosting')

def _new(kind):
    if kind.startswith('ridge_'):
        return make_pipeline(StandardScaler(), Ridge(alpha=float(kind.split('_')[1])))
    if kind == 'boosting':
        return HistGradientBoostingRegressor(max_iter=90,max_leaf_nodes=8,learning_rate=.035,
                l2_regularization=100.,min_samples_leaf=45,random_state=42)
    raise ValueError(f'Unknown candidate {kind}')

def _forward(X,d,year,kind,features):
    tr=d.season<year; te=d.season==year
    if tr.sum()<100 or te.sum()<10: return None
    target=(d.home_score-d.away_score+d.home_spread).to_numpy(dtype=float)
    m=_new(kind)
    m.fit(X.loc[tr,features],target[tr.to_numpy()])
    sub=d.loc[te].copy()
    sub['pred_residual']=m.predict(X.loc[te,features])
    sub['actual_residual']=target[te.to_numpy()]
    sub['eval_season']=year
    return sub

def _sigma(frame):
    if frame is None or len(frame)<2: return 14.
    return max(7.,float(np.std(frame.actual_residual-frame.pred_residual,ddof=1)))

def _metrics(f,sigma):
    actual=f.actual_residual.to_numpy(dtype=float)
    pred=f.pred_residual.to_numpy(dtype=float)
    margin=(f.home_score-f.away_score).to_numpy(dtype=float)
    spread=f.home_spread.to_numpy(dtype=float)
    graded=actual!=0
    correct=(pred>0)==(actual>0)
    p=ndtr(pred/sigma)
    market_margin=-spread
    result={'games':len(f), 'ATS_graded_games':int(graded.sum()),
        'market_margin_MAE':round(float(mean_absolute_error(margin,market_margin)),3),
        'model_margin_MAE':round(float(mean_absolute_error(margin,market_margin+pred)),3),
        'model_winner_accuracy':round(float(np.mean((market_margin+pred>0)==(margin>0))),4),
        'market_winner_accuracy':round(float(np.mean((market_margin>0)==(margin>0))),4),
        'ATS_accuracy_nonpush':round(float(np.mean(correct[graded])),4) if graded.any() else None,
        'cover_brier_nonpush':round(float(brier_score_loss((actual[graded]>0).astype(int),p[graded])),4) if graded.any() else None,
        'pred_residual_mean':round(float(np.mean(pred)),3),
        'pred_residual_sd':round(float(np.std(pred)),3),
        'actual_residual_sd':round(float(np.std(actual)),3),
        'pred_actual_residual_correlation':round(float(np.corrcoef(pred,actual)[0,1]),4) if np.std(pred)>1e-10 and np.std(actual)>1e-10 else None,
        'sigma_from_prior_validation':round(float(sigma),3)}
    for edge in (1,2,3,5):
        mask=graded & (np.abs(pred)>=edge)
        n=int(mask.sum())
        result[f'edge_{edge}pt_bets']=n
        result[f'edge_{edge}pt_accuracy']=round(float(np.mean(correct[mask])),4) if n else None
        result[f'edge_{edge}pt_ROI_minus110']=round(float(np.mean(np.where(correct[mask],100/110,-1))),4) if n else None
    return result

def _provider_report(f):
    if 'spread_provider' not in f: return pd.DataFrame()
    records=[]
    for provider,part in f.groupby('spread_provider',dropna=False):
        if len(part)<5: continue
        metrics=_metrics(part,14.)
        records.append({'provider':str(provider),'games':len(part),
            'market_margin_MAE':metrics['market_margin_MAE'],
            'model_margin_MAE':metrics['model_margin_MAE'],
            'ATS_accuracy_nonpush':metrics['ATS_accuracy_nonpush'],
            'pred_residual_sd':metrics['pred_residual_sd']})
    return pd.DataFrame(records)

def train_and_evaluate(X,details,features=None):
    features=list(FEATURES) if features is None else list(features)
    if list(X.columns)!=features:
        raise ValueError('V7 feature mismatch. Replace market_model.py and v5_features.py together.')
    if 2025 not in set(details.season): raise ValueError('2025 benchmark season missing')
    candidate_rows=[]; frames={}
    for kind in CANDIDATES:
        valid=[_forward(X,details,year,kind,features) for year in (2022,2023,2024)]
        valid=[f for f in valid if f is not None]
        if not valid: raise ValueError('Insufficient 2022-24 validation games')
        joined=pd.concat(valid,ignore_index=True)
        frames[kind]=joined
        candidate_rows.append({'model':kind,'validation_games':len(joined),
          'validation_margin_MAE':float(np.mean(np.abs(joined.actual_residual-joined.pred_residual))),
          'validation_market_MAE':float(np.mean(np.abs(joined.actual_residual))),
          'predicted_residual_sd':float(np.std(joined.pred_residual))})
    candidates=pd.DataFrame(candidate_rows).sort_values(['validation_margin_MAE','model']).reset_index(drop=True)
    chosen=str(candidates.iloc[0]['model'])
    calibration=frames[chosen]
    sigma=_sigma(calibration)
    holdout=_forward(X,details,2025,chosen,features)
    if holdout is None: raise ValueError('Insufficient 2025 games')
    yearly=[]
    for year in (2023,2024):
        earlier=calibration[calibration.eval_season<year]
        part=calibration[calibration.eval_season==year]
        if len(earlier)>=20 and len(part): yearly.append({'season':year,'model':chosen,**_metrics(part,_sigma(earlier))})
    yearly.append({'season':2025,'model':chosen,**_metrics(holdout,sigma)})
    target=(details.home_score-details.away_score+details.home_spread).to_numpy(dtype=float)
    final=_new(chosen)
    final.fit(X[features],target)
    model={'type':'market_residual_v7','residual_model':final,'sigma':sigma,
        'features':features,'selected_model':chosen,
        'validation_mae_2022_2024':dict(zip(candidates.model,candidates.validation_margin_MAE.round(3))),
        'last_training_date':str(details.date.max()),
        'version':MODEL_VERSION}
    return model,yearly,holdout,candidates,_provider_report(holdout)

def predict_upcoming(model,X,upcoming):
    required=model['features']
    missing=[c for c in required if c not in X.columns]
    if missing: raise ValueError(f'Upcoming features missing: {missing}')
    residual=model['residual_model'].predict(X[required])
    p=ndtr(residual/model['sigma'])
    out=upcoming[['date','home_team','away_team','home_spread']].copy().reset_index(drop=True)
    out['projected_home_margin']=np.round(-out.home_spread.to_numpy()+residual,2)
    out['home_cover_probability_approx']=np.round(p,4)
    out['away_cover_probability_approx']=np.round(1-p,4)
    out['model_vs_market_points']=np.round(residual,2)
    return out
