"""Version 8 diagnostics. No changes to production model or training artifacts."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from v5_features import build_v5_features
from v7_advanced import build_advanced_features, ADV_FEATURES

METRICS = [f'{s}_{m}' for s in ('offense','defense') for m in ('ppa','success','explosiveness')]

def audit_coverage(games, advanced):
    """Return season-level game joins and raw metric coverage, including zeros/missingness."""
    required={'game_id','season','home_team','away_team','date'}
    if not required.issubset(games.columns):
        raise ValueError(f'Historical games need {sorted(required - set(games.columns))}')
    required_adv={'game_id','team',*METRICS}
    if not required_adv.issubset(advanced.columns):
        raise ValueError(f'Advanced rows need {sorted(required_adv - set(advanced.columns))}')
    g=games.copy().reset_index(drop=True)
    a=advanced.copy()
    g['game_id']=g.game_id.astype(str)
    a['game_id']=a.game_id.astype(str)
    a['team']=a.team.astype(str)
    duplicate_rows=int(a.duplicated(['game_id','team']).sum())
    a=a.drop_duplicates(['game_id','team'],keep='last')
    home=a.rename(columns={'team':'home_team',**{c:'home_'+c for c in METRICS}})
    away=a.rename(columns={'team':'away_team',**{c:'away_'+c for c in METRICS}})
    joined=g.merge(home[['game_id','home_team']+['home_'+c for c in METRICS]],on=['game_id','home_team'],how='left',validate='many_to_one')
    joined=joined.merge(away[['game_id','away_team']+['away_'+c for c in METRICS]],on=['game_id','away_team'],how='left',validate='many_to_one')
    cols=['home_'+c for c in METRICS]+['away_'+c for c in METRICS]
    for c in cols: joined[c]=pd.to_numeric(joined[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    joined['home_has_any']=joined[['home_'+c for c in METRICS]].notna().any(axis=1)
    joined['away_has_any']=joined[['away_'+c for c in METRICS]].notna().any(axis=1)
    joined['both_teams_have_any']=joined.home_has_any & joined.away_has_any
    joined['both_teams_complete']=joined[cols].notna().all(axis=1)
    records=[]
    for season,part in joined.groupby('season',sort=True):
        r={'season':int(season),'games':len(part),
           'both_teams_any_pct':round(100*part.both_teams_have_any.mean(),1),
           'both_teams_all_six_pct':round(100*part.both_teams_complete.mean(),1)}
        for c in METRICS:
            r[c+'_both_pct']=round(100*(part['home_'+c].notna() & part['away_'+c].notna()).mean(),1)
        records.append(r)
    summary={'historical_games':len(g),'raw_advanced_rows':len(advanced),
             'duplicate_team_game_rows':duplicate_rows,
             'both_teams_any_games':int(joined.both_teams_have_any.sum()),
             'both_teams_complete_games':int(joined.both_teams_complete.sum())}
    return pd.DataFrame(records), joined, summary

def audit_feature_values(games,advanced):
    """Pregame feature diagnostics and a simple future-result perturbation leakage test."""
    ordered=games.sort_values(['date','home_team','away_team']).reset_index(drop=True)
    x=build_advanced_features(ordered,advanced)
    records=[]
    for c in ADV_FEATURES:
        s=pd.to_numeric(x[c],errors='coerce')
        records.append({'feature':c,'nonmissing_pct':round(100*s.notna().mean(),1),
                        'zero_pct':round(100*(s==0).mean(),1),
                        'std_dev':round(float(s.std(ddof=0)),5),
                        'unique_values':int(s.nunique())})
    # Verify the final game's stats do not change any features at or before its kickoff.
    # This is a limited smoke test, not a formal proof of leakage-free engineering.
    if len(ordered)>=2:
        last_id=str(ordered.iloc[-1].game_id)
        altered=advanced.loc[advanced.game_id.astype(str)!=last_id].copy()
        x_alt=build_advanced_features(ordered,altered)
        unchanged=bool(np.allclose(x.to_numpy(dtype=float),x_alt.to_numpy(dtype=float),equal_nan=True))
    else: unchanged=None
    return pd.DataFrame(records),x,unchanged

def ablation(games,advanced,years=(2022,2023,2024,2025),alpha=1000.):
    """Same-game fixed-model walk-forward comparison, no hyperparameter selection on 2025."""
    ordered=games.sort_values(['date','home_team','away_team']).reset_index(drop=True)
    x_base,d=build_v5_features(ordered)
    x_adv=build_advanced_features(d,advanced)
    x_full=pd.concat([x_base.reset_index(drop=True),x_adv.reset_index(drop=True)],axis=1)
    y=(d.home_score-d.away_score+d.home_spread).to_numpy(dtype=float)
    rows=[]
    for year in years:
        train=(d.season<year).to_numpy(); test=(d.season==year).to_numpy()
        if train.sum()<100 or test.sum()<10: continue
        record={'season':year,'games':int(test.sum()),'market_MAE':round(float(np.mean(np.abs(y[test]))),3)}
        for name,xx in [('score_only',x_base),('score_plus_advanced',x_full)]:
            model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
            model.fit(xx.iloc[train],y[train])
            pred=model.predict(xx.iloc[test])
            record[name+'_MAE']=round(float(mean_absolute_error(y[test],pred)),3)
            nonpush=y[test]!=0
            record[name+'_ATS']=round(float(np.mean((pred[nonpush]>0)==(y[test][nonpush]>0))),4) if nonpush.any() else None
            record[name+'_edge_sd']=round(float(np.std(pred)),3)
        record['advanced_minus_score_MAE']=round(record['score_plus_advanced_MAE']-record['score_only_MAE'],3)
        rows.append(record)
    return pd.DataFrame(rows)
