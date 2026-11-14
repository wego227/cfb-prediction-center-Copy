"""V5 market residual model: choose Ridge vs boosting using 2022-24 only."""
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,brier_score_loss
from v5_features import FEATURES

def _new(kind):
    if kind=='ridge': return make_pipeline(StandardScaler(),Ridge(alpha=300.))
    return HistGradientBoostingRegressor(max_iter=90,max_leaf_nodes=8,learning_rate=.035,l2_regularization=100.,min_samples_leaf=45,random_state=42)

def _forward(X,d,season,kind):
    tr=d.season<season; te=d.season==season
    if tr.sum()<100 or te.sum()<10: return None,None
    m=_new(kind)
    y=(d.home_score-d.away_score+d.home_spread).to_numpy(dtype=float)
    m.fit(X.loc[tr,FEATURES],y[tr.to_numpy()])
    sub=d.loc[te].copy()
    sub['pred_residual']=m.predict(X.loc[te,FEATURES])
    sub['actual_residual']=y[te.to_numpy()]
    return m,sub

def _sigma(frames):
    errors=np.concatenate([(f.actual_residual-f.pred_residual).to_numpy(dtype=float) for f in frames if f is not None])
    return max(7.,float(np.std(errors,ddof=1))) if len(errors)>1 else 14.

def _metrics(f,sigma):
    actual=f.actual_residual.to_numpy(dtype=float)
    predicted=f.pred_residual.to_numpy(dtype=float)
    margin=(f.home_score-f.away_score).to_numpy(dtype=float)
    spread=f.home_spread.to_numpy(dtype=float)
    graded=actual!=0
    correct=(predicted>0)==(actual>0)
    result={'games':len(f),'winner_accuracy_market_relative':round(float(np.mean((-spread+predicted>0)==(margin>0))),4),
      'margin_MAE':round(float(mean_absolute_error(margin,-spread+predicted)),2),
      'ATS_accuracy_all_nonpush':round(float(np.mean(correct[graded])),4) if graded.any() else None,
      'ATS_graded_games':int(graded.sum()),
      'cover_brier_nonpush':round(float(brier_score_loss((actual[graded]>0).astype(int),ndtr(predicted[graded]/sigma))),4) if graded.any() else None,
      'probability_sigma':round(sigma,2)}
    for edge in (2,3,5):
        mask=graded & (np.abs(predicted)>=edge)
        n=int(mask.sum())
        result[f'ATS_at_{edge}pt_edge']={'bets':n,'accuracy':round(float(np.mean(correct[mask])),4) if n else None,
          'ROI_flat_minus110':round(float(np.mean(np.where(correct[mask],100/110,-1))),4) if n else None}
    return result

def train_and_evaluate(X,details):
    if 2025 not in details.season.unique(): raise ValueError('2025 holdout season missing')
    candidates={}
    for kind in ('ridge','boosting'):
        frames=[_forward(X,details,yr,kind)[1] for yr in (2022,2023,2024)]
        frames=[f for f in frames if f is not None]
        if not frames: raise ValueError('Not enough pre-2025 seasons for validation')
        joined=pd.concat(frames,ignore_index=True)
        candidates[kind]={'mae':float(np.mean(np.abs(joined.actual_residual-joined.pred_residual))),'frames':frames}
    # Select on 2022-24 residual MAE, NEVER 2025.
    chosen=min(candidates,key=lambda k:candidates[k]['mae'])
    cal=candidates[chosen]['frames']
    sigma=_sigma(cal)
    _,test=_forward(X,details,2025,chosen)
    if test is None: raise ValueError('2025 test data insufficient')
    yearly=[]
    for f in cal:
        yr=int(f.season.iloc[0]); earlier=[x for x in cal if int(x.season.iloc[0])<yr]
        if earlier: yearly.append({'season':yr,'model':chosen,**_metrics(f,_sigma(earlier))})
    yearly.append({'season':2025,'model':chosen,**_metrics(test,sigma)})
    final=_new(chosen)
    y=(details.home_score-details.away_score+details.home_spread).to_numpy(dtype=float)
    final.fit(X[FEATURES],y)
    model={'type':'market_residual_v5','residual_model':final,'sigma':sigma,'features':FEATURES,
           'selected_model':chosen,'validation_mae_2022_2024':{k:round(v['mae'],3) for k,v in candidates.items()},
           'last_training_date':str(details.date.max())}
    return model,yearly,test

def predict_upcoming(model,X,upcoming):
    features=model['features']
    residual=model['residual_model'].predict(X[features])
    p=ndtr(residual/model['sigma'])
    out=upcoming[['date','home_team','away_team','home_spread']].copy().reset_index(drop=True)
    out['projected_home_margin']=np.round(-out.home_spread.to_numpy()+residual,2)
    out['home_cover_probability_approx']=np.round(p,4)
    out['away_cover_probability_approx']=np.round(1-p,4)
    out['model_vs_market_points']=np.round(residual,2)
    return out
