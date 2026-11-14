"""Historical completed FBS games and historical home spreads from CollegeFootballData.
Historical line availability depends on the CFBD API plan and game coverage.
"""
import requests
import pandas as pd

BASE='https://api.collegefootballdata.com'

def download_history(api_key, start_year=2019, end_year=2025, progress=None):
    if not api_key: raise ValueError('CFBD_API_KEY is missing from Streamlit secrets.')
    session=requests.Session()
    session.headers.update({'Authorization':f'Bearer {api_key}','Accept':'application/json'})
    rows=[]; missing_lines=0; seasons=[]
    for year in range(start_year,end_year+1):
        if progress: progress(f'Downloading {year} games and historical spreads…')
        r=session.get(BASE+'/games',params={'year':year,'seasonType':'regular','classification':'fbs'},timeout=45)
        r.raise_for_status(); games=r.json()
        r=session.get(BASE+'/lines',params={'year':year,'seasonType':'regular'},timeout=45)
        if r.status_code in (401,403):
            raise ValueError('The CFBD API plan/key does not permit historical betting lines (/lines). This baseline needs historical spreads; check CFBD plan access.')
        r.raise_for_status(); line_games=r.json()
        by_id={str(x.get('id')):x for x in line_games}
        count=0
        for g in games:
            if not g.get('completed') or g.get('homePoints') is None or g.get('awayPoints') is None: continue
            lg=by_id.get(str(g.get('id')), {})
            providers=lg.get('lines') or []
            # Use one consistently selected provider, not an average of mismatched prices.
            selected=next((x for x in providers if x.get('spread') is not None and 'draftkings' in str(x.get('provider','')).lower()),None)
            if selected is None: selected=next((x for x in providers if x.get('spread') is not None),None)
            if selected is None:
                missing_lines+=1; continue
            # CFBD spread is home-team spread (negative when home favored).
            rows.append({'date':g.get('startDate'),'season':year,'home_team':g['homeTeam'],'away_team':g['awayTeam'],
                'home_score':g['homePoints'],'away_score':g['awayPoints'],'home_spread':float(selected['spread']),
                'game_id':str(g.get('id')),'neutral_site':int(bool(g.get('neutralSite',False))),'spread_provider':selected.get('provider','unknown')})
            count+=1
        seasons.append((year,count))
    df=pd.DataFrame(rows)
    if df.empty: raise ValueError('No completed games with historical spreads were returned. Verify key permissions and available years.')
    # Preserve UTC instants as timezone-naive UTC for existing feature pipeline.
    df['date']=pd.to_datetime(df['date'],utc=True).dt.tz_localize(None)
    return df.sort_values('date').reset_index(drop=True), {'seasons':seasons,'skipped_without_spread':missing_lines}
