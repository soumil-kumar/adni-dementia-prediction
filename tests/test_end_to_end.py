"""Exercise the real preparation-to-prediction path without ADNI subject records."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from btp.prepare import prepare
from btp.schema import A_FEATURES,B_FEATURES,CATEGORICAL,NEURO,HISTORY
from btp.splits import ensure_splits, load_partition
from btp.search import finalize
from btp.evaluation import evaluate
from btp.training import assert_development_open
from btp.common import dump


def test_synthetic_end_to_end_and_freeze_guard(tmp_path):
    source=tmp_path/'source'
    source.mkdir()
    local=tmp_path/'local'
    n=100
    rid=np.arange(1,n+1)
    y=(rid%3==0).astype(int)
    a=pd.DataFrame({'RID':rid})
    for col in A_FEATURES:
        a[col]='Male' if col in CATEGORICAL else 1.0
    a['AGE']=60+rid/5
    a['MMSE_bl']=29-7*y+(rid%2)
    a['APOE4']=rid%3
    a.to_csv(source/'person_A_features.csv',index=False)
    b=pd.DataFrame({'RID':rid,**{col:np.zeros(n) for col in B_FEATURES}})
    b.to_csv(source/'person_B_features.csv',index=False)
    pd.DataFrame({'RID':rid,'EXAMDATE':'2020-01-01'}).to_csv(source/'shared_subject_list.csv',index=False)
    merged=a.copy()
    merged['EXAMDATE']='01-01-2020'
    merged['VISCODE']='bl'
    merged['DX_bl']='CN'
    merged['COLPROT']='ADNI2'
    adni=source/'tadpole_challenge_201911210/tadpole_challenge'
    adni.mkdir(parents=True)
    merged.to_csv(adni/'ADNIMERGE.csv',index=False)
    pd.DataFrame({'RID':rid,'EXAMDATE':np.where(y==1,'2022-01-01','2026-01-01'),
        'DIAGNOSIS':np.where(y==1,3,1)}).to_csv(source/'DXSUM_05Sep2026.csv',index=False)
    pd.DataFrame({'RID':rid,'VISDATE':'2019-12-15','VISCODE':'sc','VISCODE2':'sc',
        **{col:np.ones(n) for col in NEURO}}).to_csv(source/'NEUROBAT_06Sep2026.csv',index=False)
    pd.DataFrame({'RID':rid,'VISDATE':'2019-12-15',**{col:np.zeros(n) for col in HISTORY}}).to_csv(
        source/'MEDHIST_06Sep2026.csv',index=False)
    pd.DataFrame({'RID':rid,'Label':y}).to_csv(source/'outcome_labels.csv',index=False)
    cfg=dict(data_dir=str(source),local_dir=str(local),config_path=str(tmp_path/'config.json'),
        seed=42,development_fraction=.6,calibration_fraction=.2,validation_fraction=.2,
        folds=5,screening_lookback_days=90,bootstrap_iterations=10)
    report=prepare(cfg)
    assert report['labeled_subjects']==n
    splits=ensure_splits(cfg)
    sets=[set(v) for v in splits['subjects'].values()]
    assert len(set.union(*sets))==n
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    selected={'spec':{'model':'logistic','panel':'clinical','params':{'C':1.0}}}
    dump(local/'selection.json',{'selected':selected})
    finalize(cfg,selected)
    model=joblib.load(local/'model.joblib')
    assert np.isfinite(model.predict_proba(load_partition(cfg,'calibration'))).all()
    results=evaluate(cfg)
    assert results['primary']['n']==20
    assert evaluate(cfg)==results
    with pytest.raises(RuntimeError):
        assert_development_open(cfg)
    with pytest.raises(RuntimeError):
        prepare(cfg)
