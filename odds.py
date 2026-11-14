"""DraftKings NCAAF odds via The Odds API (licensed third-party aggregator).
Requires THE_ODDS_API_KEY. No scraping, no invented lines.
"""
import os
import requests
import pandas as pd

URL='https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds/'

def fetch_draftkings(api_key=None):
    key=api_key or os.getenv('THE_ODDS_API_KEY')
    if not key: raise ValueError('Set THE_ODDS_API_KEY or enter an API key in the dashboard.')
    response=requests.get(URL,params={'apiKey':key,'regions':'us','markets':'spreads,h2h','bookmakers':'draftkings','oddsFormat':'american','dateFormat':'iso'},timeout=25)
    response.raise_for_status()
    rows=[]
    for event in response.json():
        for book in event.get('bookmakers',[]):
            if book.get('key')!='draftkings': continue
            market={m['key']:m.get('outcomes',[]) for m in book.get('markets',[])}
            spread={x['name']:x for x in market.get('spreads',[])}
            ml={x['name']:x for x in market.get('h2h',[])}
            home=event['home_team']; away=event['away_team']
            if home not in spread or away not in spread: continue
            rows.append({'event_id':event['id'],'date':event['commence_time'],'home_team':home,'away_team':away,
                'home_spread':float(spread[home]['point']),'home_spread_odds':int(spread[home]['price']),
                'away_spread_odds':int(spread[away]['price']),
                'home_moneyline':ml.get(home,{}).get('price'),'away_moneyline':ml.get(away,{}).get('price'),
                'bookmaker':'DraftKings','line_updated_at':book.get('last_update')})
    return pd.DataFrame(rows)
