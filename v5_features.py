"""Leakage-aware, score-derived pregame features. No EPA/play data is claimed."""
from collections import defaultdict
import numpy as np
import pandas as pd

FEATURES = [f'diff_{x}' for x in ('games','win_pct','margin','points_for','points_against','recent_margin','weighted_margin','opponent_margin','schedule_strength','prior_season_margin','rest_days')] + ['neutral_site','season_week']

def _stats(team, season, date, ledger):
    games=[g for g in ledger[team] if g['season']==season]
    prior=[g for g in ledger[team] if g['season']==season-1]
    if not games:
        base=[0.,.5,0.,0.,0.,0.,0.,0.,0.]
        rest=14.
    else:
        margins=np.array([g['margin'] for g in games],dtype=float)
        weights=np.exp(-np.arange(len(games)-1,-1,-1)/4.)
        base=[float(len(games)),float(np.mean(margins>0)),float(np.mean(margins)),
              float(np.mean([g['pf'] for g in games])),float(np.mean([g['pa'] for g in games])),
              float(np.mean(margins[-3:])),float(np.average(margins,weights=weights)),
              float(np.mean([g['opp_prior_margin'] for g in games])),
              float(np.mean([g['opp_prior_margin'] for g in games]))]
        rest=float(np.clip((date-games[-1]['date']).total_seconds()/86400,0,30))
    prior_margin=float(np.mean([g['margin'] for g in prior])) if prior else 0.
    return np.array(base+[prior_margin,rest],dtype=float)

def build_v5_features(games, future=None):
    """Game rows use only completed games with strictly earlier kickoff timestamps."""
    games=games.sort_values(['date','home_team','away_team']).reset_index(drop=True)
    ledger=defaultdict(list)
    def feature(g):
        h=_stats(g.home_team,int(g.season),g.date,ledger)
        a=_stats(g.away_team,int(g.season),g.date,ledger)
        row={f'diff_{name}':float(val) for name,val in zip(('games','win_pct','margin','points_for','points_against','recent_margin','weighted_margin','opponent_margin','schedule_strength','prior_season_margin','rest_days'),h-a)}
        row['neutral_site']=int(g.neutral_site)
        row['season_week']=max(0,int((g.date-pd.Timestamp(int(g.season),8,1)).days//7))
        return row
    rows=[]
    # Only games strictly earlier than a kickoff are added to the ledger.
    for date,group in games.groupby('date',sort=True):
        for g in group.itertuples(index=False): rows.append(feature(g))
        for g in group.itertuples(index=False):
            home_margin=float(g.home_score-g.away_score)
            hprior=_stats(g.home_team,int(g.season),g.date,ledger)[2]
            aprior=_stats(g.away_team,int(g.season),g.date,ledger)[2]
            ledger[g.home_team].append(dict(season=int(g.season),date=g.date,margin=home_margin,pf=float(g.home_score),pa=float(g.away_score),opp_prior_margin=aprior))
            ledger[g.away_team].append(dict(season=int(g.season),date=g.date,margin=-home_margin,pf=float(g.away_score),pa=float(g.home_score),opp_prior_margin=hprior))
    if future is None:
        return pd.DataFrame(rows,columns=FEATURES),games.copy()
    future=future.sort_values(['date','home_team','away_team']).reset_index(drop=True)
    # Rebuild ledger to each requested future kickoff if there are future rows before the last historical game.
    if len(future) and len(games) and future.date.min()<=games.date.max():
        raise ValueError('Future kickoffs must be after the last completed historical game.')
    return pd.DataFrame([feature(g) for g in future.itertuples(index=False)],columns=FEATURES)
