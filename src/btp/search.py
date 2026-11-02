import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from .artifact import Predictor, logits
from .common import dump, sha256, versions
from .metrics import best_threshold, scores
from .schema import FEATURES
from .splits import load_partition
from .training import compare, rank, build_pipeline, assert_development_open


SPACES = {
    'logistic': {'C':[0.01,0.05,0.1,0.5,1,5,10], 'class_weight':[None,'balanced']},
    'forest': {'max_depth':[3,5,8,None], 'min_samples_leaf':[2,5,10],
        'max_features':['sqrt',0.7], 'class_weight':[None,'balanced']},
    'xgboost': {'n_estimators':[120,240], 'max_depth':[2,3,4],
        'learning_rate':[0.03,0.07], 'subsample':[0.8,1.0], 'colsample_bytree':[0.8,1.0],
        'reg_lambda':[1,10], 'min_child_weight':[3,8], 'scale_pos_weight':[1.0,2.0]},
    'catboost': {'iterations':[150,300], 'depth':[3,4,5], 'learning_rate':[0.03,0.07],
        'l2_leaf_reg':[3,10], 'auto_class_weights':[None,'Balanced']},
}


def train(cfg):
    assert_development_open(cfg)
    iterations = cfg['search_iterations']
    if not 1<=iterations<=min(30,cfg['search_cap']):
        raise ValueError('Search iterations must be in 1..30')
    specs=[]
    for model,space in SPACES.items():
        parameters=list(ParameterSampler(space,n_iter=iterations,random_state=cfg['seed']))
        for panel in ['clinical','core','expanded']:
            specs.extend(dict(model=model,panel=panel,params=p) for p in parameters)
    records=compare(cfg,specs,'search')
    strongest=rank(records)[:3]
    # Prespecified alternatives only for the three strongest development candidates.
    alternatives=[]
    for candidate in strongest:
        spec=candidate['spec']
        alternatives.append(dict(spec,imputation='knn'))
        if spec['panel']!='clinical':
            alternatives.append(dict(spec,normalize_mri=True))
    records=compare(cfg,alternatives,'preprocessing_ablation')
    selected=rank(records)[0]
    dump(Path(cfg['local_dir'])/'selection.json',dict(selected=selected,
        rule='mean five-fold F1; ties recall then simpler model; inner holdout threshold per fold',
        search_iterations_per_model_panel=iterations, experiments=len(records)))
    finalize(cfg,selected)


def finalize(cfg,selected):
    assert_development_open(cfg)
    local=Path(cfg['local_dir'])
    dev=load_partition(cfg,'development')
    cal=load_partition(cfg,'calibration')
    pipeline=build_pipeline(selected['spec'],cfg['seed']).fit(dev,dev.Label.astype(int))
    raw=pipeline.predict_proba(cal)[:,1]
    calibrator=LogisticRegression(C=1e6,solver='lbfgs',random_state=cfg['seed']).fit(logits(raw),cal.Label.astype(int))
    if calibrator.coef_[0,0]<=0:
        raise RuntimeError('Calibration reverses risk ranking; investigate before opening final validation')
    probabilities=calibrator.predict_proba(logits(raw))[:,1]
    threshold=best_threshold(cal.Label.astype(int),probabilities)
    predictor=Predictor(pipeline,calibrator,threshold,'0.1.0',selected['spec'])
    target=local/'model.joblib'
    joblib.dump(predictor,target)
    restored=joblib.load(target)
    np.testing.assert_allclose(restored.predict_proba(cal),probabilities,rtol=0,atol=0)
    manifest=dict(model_sha256=sha256(target),master_sha256=sha256(local/'master_baseline.csv'),
        split_sha256=sha256(local/'splits.json'),selection_sha256=sha256(local/'selection.json'),
        config=cfg,versions=versions(),selected=selected,threshold=threshold,
        fitted_features=pipeline.named_steps['panel'].columns_,
        calibration=scores(cal.Label,probabilities,threshold),
        calibration_note='Calibration and threshold use the same reserved partition; its metrics are fitted, not independent evaluation.',
        raw_predictors=FEATURES,protocol='Any incident dementia within five calendar years; original baseline only')
    dump(local/'frozen_manifest.json',manifest)
    print('Frozen model:',selected['spec'],'threshold=',round(threshold,4),flush=True)
