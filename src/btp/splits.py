import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .common import dump, sha256


def ensure_splits(cfg):
    local = Path(cfg['local_dir'])
    source_hash = sha256(local / 'master_baseline.csv')
    path = local / 'splits.json'
    if path.exists():
        result = json.loads(path.read_text())
        if result['master_sha256'] != source_hash or result['seed'] != cfg['seed']:
            raise ValueError('Existing splits belong to different data/seed; use a new local_dir')
        return result
    fractions = [cfg[k] for k in ['development_fraction','calibration_fraction','validation_fraction']]
    if fractions != [0.6, 0.2, 0.2]:
        raise ValueError('This protocol requires a fixed 60/20/20 split')
    data = pd.read_csv(local / 'master_baseline.csv')
    data = data[data.Label.notna()].sort_values('RID').reset_index(drop=True)
    devcal, val = train_test_split(data, test_size=0.2, random_state=cfg['seed'], stratify=data.Label)
    dev, cal = train_test_split(devcal, test_size=0.25, random_state=cfg['seed'], stratify=devcal.Label)
    groups = {'development':dev, 'calibration':cal, 'validation':val}
    result = dict(master_sha256=source_hash, seed=cfg['seed'],
        subjects={name:frame.RID.astype(int).tolist() for name,frame in groups.items()},
        counts={name:dict(n=len(frame), positives=int(frame.Label.sum()),
            negatives=int(frame.Label.eq(0).sum())) for name,frame in groups.items()})
    sets = [set(ids) for ids in result['subjects'].values()]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert set.union(*sets) == set(data.RID)
    dump(path, result)
    return result


def load_partition(cfg, name):
    split = ensure_splits(cfg)
    data = pd.read_csv(Path(cfg['local_dir']) / 'master_baseline.csv')
    return data.set_index('RID').loc[split['subjects'][name]].reset_index()
