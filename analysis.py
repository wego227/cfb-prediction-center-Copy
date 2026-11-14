"""Odds arithmetic and hypothetical parlay comparison; no bets placed."""
import itertools
import numpy as np
import pandas as pd

def american_decimal(odds):
    odds=float(odds)
    if odds==0: raise ValueError('Odds cannot be zero')
    return 1+odds/100 if odds>0 else 1+100/abs(odds)

def breakeven(odds): return 1/american_decimal(odds)

def analyze(pred, odds):
    # Exact team/date join prevents fuzzy name matches from silently selecting the wrong game.
    p=pred.copy(); o=odds.copy()
    p['match_date']=pd.to_datetime(p['date'],utc=True).dt.date.astype(str)
    o['match_date']=pd.to_datetime(o['date'],utc=True).dt.date.astype(str)
    cols=['match_date','home_team','away_team']
    m=p.merge(o,on=cols,how='inner',suffixes=('_model','_dk'))
    if m.empty: return pd.DataFrame()
    # Predictions must have been made with the same spread being evaluated.
    m=m[np.isclose(m['home_spread_model'],m['home_spread_dk'])].copy()
    picks=[]
    for _,r in m.iterrows():
        for side in ['home','away']:
            prob=float(r[f'{side}_cover_probability_approx'])
            odds_price=int(r[f'{side}_spread_odds'])
            dec=american_decimal(odds_price)
            picks.append({'event_id':r.get('event_id',''),'date':r['date_model'],'matchup':f"{r['away_team']} @ {r['home_team']}",
                'selection':r[f'{side}_team'],'side':side,'line':float(r['home_spread_dk'])*(1 if side=='home' else -1),
                'american_odds':odds_price,'model_cover_prob':prob,'break_even_prob':1/dec,
                'estimated_edge':prob-1/dec,'estimated_EV_per_1_risked':prob*(dec-1)-(1-prob),
                'line_updated_at':r.get('line_updated_at','')})
    return pd.DataFrame(picks).sort_values('estimated_EV_per_1_risked',ascending=False).reset_index(drop=True)

def parlays(picks, max_legs=3, min_edge=0, top_n=12):
    """Naive independence only, separate games; not a reliable correlated parlay probability."""
    if picks.empty:return pd.DataFrame()
    choices=picks[picks.estimated_edge>min_edge].sort_values('estimated_EV_per_1_risked',ascending=False).head(top_n)
    results=[]
    for n in range(2,max_legs+1):
        for idx in itertools.combinations(choices.index,n):
            legs=choices.loc[list(idx)]
            if legs['event_id'].nunique()!=n:continue
            p=float(np.prod(legs.model_cover_prob))
            decimal=float(np.prod([american_decimal(x) for x in legs.american_odds]))
            results.append({'legs':n,'selections':' + '.join(f"{r.selection} {r.line:+g}" for r in legs.itertuples()),
                'naive_joint_prob':p,'decimal_odds':decimal,'naive_EV_per_1_risked':p*decimal-1,
                'warning':'Assumes independence; uncalibrated probabilities; correlation not modeled'})
    return pd.DataFrame(results).sort_values('naive_EV_per_1_risked',ascending=False) if results else pd.DataFrame()
