"""Historical game-level advanced metrics, joined by game ID and team.

Advanced stats from each game enter the ledger only AFTER that game's kickoff.
Never use season-to-date endpoint values directly as historical pregame features.
"""
import requests
import numpy as np
import pandas as pd
from collections import defaultdict

METRICS = ('ppa','success','explosiveness')
ADV_FEATURES = [f'diff_{side}_{metric}' for side in ('offense','defense') for metric in METRICS] + ['diff_advanced_games']


def _number(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _nested(obj, *paths):
    for path in paths:
        node = obj
        for part in path:
            if not isinstance(node, dict):
                node = None; break
            node = node.get(part)
        value = _number(node)
        if value is not None:
            return value
    return None


def extract_team_row(record):
    """CFBD advanced game stats use offense/defense nested blocks."""
    stats = record.get('stats') or {}
    out = {'game_id':str(record.get('gameId',record.get('game_id', ''))),
           'team':str(record.get('team',''))}
    for side in ('offense','defense'):
        block = stats.get(side) or record.get(side) or {}
        out[f'{side}_ppa'] = _nested(block,('ppa',),('ppa','total'),('ppa','overall'))
        out[f'{side}_success'] = _nested(block,('successRate',),('successRate','total'),('successRate','overall'))
        out[f'{side}_explosiveness'] = _nested(block,('explosiveness',),('explosiveness','total'),('explosiveness','overall'))
    return out


def download_advanced(api_key, years, progress=None):
    """Retrieve regular-season game advanced stats week-by-week; report coverage."""
    session=requests.Session()
    session.headers.update({'Authorization':f'Bearer {api_key}','Accept':'application/json'})
    rows=[]; failures=[]
    for year in years:
        for week in range(1,17):
            if progress: progress(f'Advanced stats: {year}, week {week}…')
            try:
                r=session.get('https://api.collegefootballdata.com/stats/game/advanced',
                    params={'year':int(year),'week':week,'seasonType':'regular'},timeout=45)
                if r.status_code in (401,403,429):
                    raise RuntimeError(f'CFBD advanced stats HTTP {r.status_code} (entitlement/quota).')
                r.raise_for_status()
                payload=r.json()
                if not isinstance(payload,list): raise ValueError('Unexpected CFBD advanced stats response')
                rows.extend(extract_team_row(x) for x in payload if isinstance(x,dict))
            except (requests.RequestException,ValueError) as exc:
                failures.append({'year':int(year),'week':week,'error':str(exc)[:160]})
    frame=pd.DataFrame(rows,columns=['game_id','team']+[f'{s}_{m}' for s in ('offense','defense') for m in METRICS])
    if frame.empty: raise ValueError('No advanced game statistics downloaded. Check access and API quota.')
    frame=frame.drop_duplicates(['game_id','team'],keep='last')
    return frame,{'advanced_team_game_rows':len(frame),'failed_requests':failures,
                  'nonmissing_by_metric':{c:int(frame[c].notna().sum()) for c in frame.columns if c not in ('game_id','team')}}


def build_advanced_features(games, advanced, future=None):
    """Pregame features: only prior completed game advanced stats, with same-kickoff isolation."""
    if 'game_id' not in games:
        raise ValueError('Historical games missing game_id. Replace cfbd_data.py and re-download history.')
    games=games.sort_values(['date','home_team','away_team']).reset_index(drop=True)
    lookup={(str(r.game_id),str(r.team)):r for r in advanced.itertuples(index=False)}
    ledger=defaultdict(list)
    def features(g):
        home=ledger[str(g.home_team)]; away=ledger[str(g.away_team)]
        result={}
        for side in ('offense','defense'):
            for metric in METRICS:
                col=f'{side}_{metric}'
                def avg(records):
                    values=[x[col] for x in records if x.get(col) is not None]
                    return float(np.mean(values[-5:])) if values else 0.
                result[f'diff_{col}']=avg(home)-avg(away)
        result['diff_advanced_games']=float(len(home)-len(away))
        return result
    rows=[]
    for date, group in games.groupby('date',sort=True):
        for g in group.itertuples(index=False): rows.append(features(g))
        for g in group.itertuples(index=False):
            for team in (str(g.home_team),str(g.away_team)):
                rec=lookup.get((str(g.game_id),team))
                if rec is not None:
                    vals={f'{s}_{m}':getattr(rec,f'{s}_{m}') for s in ('offense','defense') for m in METRICS}
                    if any(v is not None for v in vals.values()): ledger[team].append(vals)
    if future is None: return pd.DataFrame(rows,columns=ADV_FEATURES)
    future=future.sort_values(['date','home_team','away_team']).reset_index(drop=True)
    if len(games) and len(future) and future.date.min()<=games.date.max():
        raise ValueError('Future kickoffs must follow all historical games')
    return pd.DataFrame([features(g) for g in future.itertuples(index=False)],columns=ADV_FEATURES)
