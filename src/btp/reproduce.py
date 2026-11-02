"""Rebuild preparation and selected fit in a fresh process without changing the frozen run."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .common import dump, sha256
from .prepare import prepare
from .search import finalize
from .splits import ensure_splits


def reproduce(cfg):
    original=Path(cfg['local_dir'])
    output=original/'reproduction'
    if output.exists():
        raise RuntimeError('Reproduction output already exists; preserve it or choose a new experiment directory')
    output.mkdir(parents=True)
    repeated=dict(cfg,local_dir=str(output))
    prepare(repeated)
    assert sha256(output/'master_baseline.csv')==sha256(original/'master_baseline.csv')
    ensure_splits(repeated)
    assert sha256(output/'splits.json')==sha256(original/'splits.json')
    selection=json.loads((original/'selection.json').read_text())
    dump(output/'selection.json',selection)
    finalize(repeated,selection['selected'])
    reference=joblib.load(original/'model.joblib')
    rebuilt=joblib.load(output/'model.joblib')
    data=pd.read_csv(original/'master_baseline.csv')
    np.testing.assert_allclose(reference.predict_proba(data),rebuilt.predict_proba(data),rtol=0,atol=1e-12)
    assert reference.threshold==rebuilt.threshold
    result=dict(master_identical=True,splits_identical=True,predictions_identical_within=1e-12,
        threshold_identical=True,rows_checked=len(data),
        note='Recreated source preparation and selected fit; no search or threshold changes based on final validation.')
    dump(original/'reproduction_check.json',result)
    print(json.dumps(result,indent=2),flush=True)
