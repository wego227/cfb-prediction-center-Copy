"""Portable pregame snapshots and grading; no bets, odds or outcomes are inferred."""
import hashlib
import pandas as pd
import numpy as np

KEYS=['date','home_team','away_team']
REQUIRED=KEYS+['home_spread','projected_home_margin','model_vs_market_points']

def snapshot(predictions, captured_at=None):
    missing=set(REQUIRED)-set(predictions.columns)
    if missing: raise ValueError('Predictions CSV missing '+', '.join(sorted(missing)))
    captured_at=captured_at or pd.Timestamp.now(tz='UTC').isoformat()
    out=predictions.copy()
    out['date']=pd.to_datetime(out.date,utc=True,errors='coerce')
    if out.date.isna().any(): raise ValueError('Invalid kickoff dates')
    out['captured_at_utc']=captured_at
    out['snapshot_id']=[hashlib.sha256(f'{captured_at}|{r.date}|{r.home_team}|{r.away_team}|{r.home_spread}'.encode()).hexdigest()[:16] for r in out.itertuples()]
    out['captured_before_kickoff']=pd.to_datetime(captured_at,utc=True)<out.date
    out['snapshot_status']=np.where(out.captured_before_kickoff,'PREKICKOFF','LATE / UNVERIFIED')
    return out

def grade(snapshot_df, results):
    missing=set(REQUIRED)-set(snapshot_df.columns)
    if missing: raise ValueError('Snapshot missing '+', '.join(sorted(missing)))
    needed=set(KEYS+['home_score','away_score'])-set(results.columns)
    if needed: raise ValueError('Results missing '+', '.join(sorted(needed)))
    p=snapshot_df.copy(); r=results.copy()
    p['date']=pd.to_datetime(p.date,utc=True,errors='coerce')
    r['date']=pd.to_datetime(r.date,utc=True,errors='coerce')
    if p.date.isna().any() or r.date.isna().any(): raise ValueError('Invalid kickoff dates')
    if 'snapshot_id' not in p.columns:
        p['snapshot_id']=[f'import-{i}' for i in range(len(p))]
    if p.snapshot_id.duplicated().any(): raise ValueError('Duplicate snapshot_id; remove duplicates')
    if 'captured_before_kickoff' not in p.columns:
        p['captured_before_kickoff']=False
    else:
        p['captured_before_kickoff']=p['captured_before_kickoff'].astype(str).str.lower().eq('true')
    # Match canonical teams and calendar date, not exact kickoff timestamps, which can shift.
    p['_game_day']=p.date.dt.date; r['_game_day']=r.date.dt.date
    join=['_game_day','home_team','away_team']
    if r.duplicated(join).any(): raise ValueError('Duplicate result matchups on a date')
    r['home_score']=pd.to_numeric(r.home_score,errors='coerce')
    r['away_score']=pd.to_numeric(r.away_score,errors='coerce')
    out=p.merge(r[join+['home_score','away_score']],on=join,how='left',validate='many_to_one')
    out['home_spread']=pd.to_numeric(out.home_spread,errors='coerce')
    out['projected_home_margin']=pd.to_numeric(out.projected_home_margin,errors='coerce')
    out['model_vs_market_points']=pd.to_numeric(out.model_vs_market_points,errors='coerce')
    out['actual_home_margin']=out.home_score-out.away_score
    completed=out.actual_home_margin.notna()
    out['grade_status']=np.where(~completed,'AWAITING RESULT',np.where(~out.captured_before_kickoff,'UNVERIFIED TIMING','GRADED'))
    out['winner_correct']=np.where(completed,(out.projected_home_margin.gt(0)==out.actual_home_margin.gt(0)).astype(float),np.nan)
    cover_margin=out.actual_home_margin+out.home_spread
    out['ats_result']=np.select([~completed,cover_margin.eq(0),out.model_vs_market_points.eq(0),
        out.model_vs_market_points.gt(0).eq(cover_margin.gt(0))],['PENDING','PUSH','NO MODEL LEAN','WIN'],default='LOSS')
    out['margin_absolute_error']=(out.actual_home_margin-out.projected_home_margin).abs()
    out['market_margin_absolute_error']=(out.actual_home_margin+out.home_spread).abs()
    return out.drop(columns=['_game_day'])

def summary(graded,verified_only=True):
    d=graded[graded.grade_status.eq('GRADED')].copy() if verified_only else graded[graded.actual_home_margin.notna()].copy()
    eligible=d[d.ats_result.isin(['WIN','LOSS'])]
    return {'graded_games':len(d),'ats_decisions':len(eligible),'ats_pushes':int(d.ats_result.eq('PUSH').sum()),
        'ats_accuracy':float(eligible.ats_result.eq('WIN').mean()) if len(eligible) else None,
        'winner_accuracy':float(d.winner_correct.mean()) if len(d) else None,
        'model_margin_MAE':float(d.margin_absolute_error.mean()) if len(d) else None,
        'market_margin_MAE':float(d.market_margin_absolute_error.mean()) if len(d) else None}

def by_edge(graded,verified_only=True):
    d=graded[graded.grade_status.eq('GRADED')].copy() if verified_only else graded[graded.actual_home_margin.notna()].copy()
    d=d[d.ats_result.isin(['WIN','LOSS'])]
    d['edge_bucket']=pd.cut(d.model_vs_market_points.abs(),bins=[-0.001,0.5,1,2,5,float('inf')],labels=['0–0.5','0.5–1','1–2','2–5','5+'])
    return d.groupby('edge_bucket',observed=False).agg(games=('ats_result','size'),ats_accuracy=('ats_result',lambda s:s.eq('WIN').mean())).reset_index()
