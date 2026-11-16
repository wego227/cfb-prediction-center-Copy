"""Pure helpers for multi-week forward testing; no network or disk writes."""
import io
import zipfile
import pandas as pd
from prediction_tracker import grade, summary

REQUIRED_SNAPSHOT = {'date','home_team','away_team','home_spread','projected_home_margin','model_vs_market_points','snapshot_id','captured_at_utc'}

def load_snapshots(files):
    frames=[]
    for f in files:
        df=pd.read_csv(f)
        missing=REQUIRED_SNAPSHOT-set(df.columns)
        if missing:
            raise ValueError(f'{getattr(f,"name","Snapshot")}: missing {sorted(missing)}')
        df['source_filename']=getattr(f,'name','uploaded.csv')
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out=pd.concat(frames,ignore_index=True)
    # Re-uploading an identical file is safe; conflicting snapshot IDs are not.
    duplicated=out.duplicated('snapshot_id',keep=False)
    if duplicated.any():
        cols=[c for c in out.columns if c!='source_filename']
        for _, group in out[duplicated].groupby('snapshot_id'):
            if len(group[cols].drop_duplicates())!=1:
                raise ValueError('Conflicting rows share a snapshot_id; fix the archive before grading.')
        out=out.drop_duplicates('snapshot_id',keep='first')
    out['captured_at_utc']=pd.to_datetime(out['captured_at_utc'],utc=True,errors='coerce')
    if out.captured_at_utc.isna().any():
        raise ValueError('Invalid captured_at_utc values.')
    out['date']=pd.to_datetime(out.date,utc=True,errors='coerce')
    if out.date.isna().any():
        raise ValueError('Invalid kickoff timestamps.')
    # Recompute pre-kickoff status; don't trust an editable CSV boolean.
    out['captured_before_kickoff']=out.captured_at_utc.lt(out.date)
    out['snapshot_status']=out.captured_before_kickoff.map({True:'PREKICKOFF',False:'LATE / UNVERIFIED'})
    return out

def weekly_metrics(graded):
    d=graded.copy()
    d['week_start']=pd.to_datetime(d.date,utc=True).dt.tz_convert('America/New_York').dt.date.map(
        lambda x: x-pd.Timedelta(days=x.weekday()))
    rows=[]
    for (week,version),g in d.groupby(['week_start','model_version'],dropna=False,sort=True):
        s=summary(g)
        rows.append({'week_start':str(week),'model_version':version,'snapshots':len(g),
                     'graded_games':s['graded_games'],'ats_decisions':s['ats_decisions'],
                     'ats_accuracy':s['ats_accuracy'],'winner_accuracy':s['winner_accuracy'],
                     'model_margin_MAE':s['model_margin_MAE'],'market_margin_MAE':s['market_margin_MAE']})
    return pd.DataFrame(rows)

def archive_zip(snapshots,graded=None):
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('archived_pregame_snapshots.csv',snapshots.to_csv(index=False))
        if graded is not None:
            z.writestr('graded_predictions.csv',graded.to_csv(index=False))
            z.writestr('weekly_performance.csv',weekly_metrics(graded).to_csv(index=False))
        z.writestr('README.txt','Keep this archive locally. Snapshot capture timestamps are metadata, not cryptographic proof of historical capture. Do not use simulated scores for live evaluation.\n')
    return output.getvalue()
