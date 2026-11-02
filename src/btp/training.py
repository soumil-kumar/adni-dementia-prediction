"""Development-only comparison; nested holdout thresholding within each CV fold."""
import json
import time
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from .common import dump, sha256
from .metrics import best_threshold, scores
from .preprocessing import PanelFeatures, preprocessing
from .schema import FEATURES
from .splits import load_partition, ensure_splits


def build_pipeline(spec, seed=42):
    kind, params = spec['model'], spec.get('params', {})
    if kind == 'dummy':
        model = DummyClassifier(strategy='most_frequent')
    elif kind == 'logistic':
        model = LogisticRegression(max_iter=3000, random_state=seed, **params)
    elif kind == 'forest':
        model = RandomForestClassifier(n_estimators=160, random_state=seed, n_jobs=4, **params)
    elif kind == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(random_state=seed, n_jobs=4, tree_method='hist',
            eval_metric='logloss', **params)
    elif kind == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(random_seed=seed, thread_count=4, verbose=False,
            allow_writing_files=False, **params)
    else:
        raise ValueError(kind)
    return Pipeline([('panel', PanelFeatures(spec['panel'], spec.get('normalize_mri',False))),
        ('preprocessing', preprocessing(spec.get('imputation','median'))), ('model',model)])


def assert_development_open(cfg):
    local = Path(cfg['local_dir'])
    if (local / 'validation_opened.json').exists() or (local / 'validation_results.json').exists():
        raise RuntimeError('Final validation has been opened; training is frozen. Use a new local_dir for exploratory work.')


def spec_key(spec):
    return json.dumps(spec, sort_keys=True)


def compare(cfg, specifications, label):
    assert_development_open(cfg)
    local = Path(cfg['local_dir'])
    splits = ensure_splits(cfg)
    dev = load_partition(cfg,'development')
    X, y = dev.drop(columns=['Label']), dev.Label.astype(int).to_numpy()
    cv = list(StratifiedKFold(n_splits=cfg['folds'],shuffle=True,random_state=cfg['seed']).split(X,y))
    dump(local / 'development_folds.json', [dict(train_RIDs=dev.iloc[tr].RID.tolist(),
        heldout_RIDs=dev.iloc[va].RID.tolist()) for tr,va in cv])
    path = local / 'experiments.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['split_sha256'] != sha256(local / 'splits.json'):
            raise ValueError('Experiment split mismatch')
        records = previous['records']
    else:
        records = []
    completed = {spec_key(r['spec']) for r in records}
    for number,spec in enumerate(specifications,1):
        if spec_key(spec) in completed:
            continue
        started = time.time()
        pipeline = build_pipeline(spec,cfg['seed'])
        fold_metrics = []
        for fold,(tr,va) in enumerate(cv):
            if spec['model'] == 'dummy':
                threshold = 0.5
            else:
                fit, threshold_set = train_test_split(tr,test_size=0.25,
                    stratify=y[tr],random_state=cfg['seed']+fold)
                inner = clone(pipeline).fit(X.iloc[fit],y[fit])
                threshold = best_threshold(y[threshold_set],inner.predict_proba(X.iloc[threshold_set])[:,1])
            fitted = clone(pipeline).fit(X.iloc[tr],y[tr])
            result = scores(y[va],fitted.predict_proba(X.iloc[va])[:,1],threshold)
            result['threshold'] = threshold
            fold_metrics.append(result)
        mean = {k:float(np.mean([m[k] for m in fold_metrics])) for k in
            ['accuracy','precision','recall','f1','roc_auc','average_precision','balanced_accuracy']}
        std = {k:float(np.std([m[k] for m in fold_metrics],ddof=1)) for k in mean}
        record = dict(spec=spec,mean=mean,std=std,folds=fold_metrics,seconds=round(time.time()-started,2),stage=label)
        records.append(record)
        completed.add(spec_key(spec))
        dump(path,dict(split_sha256=sha256(local/'splits.json'),records=records))
        print(f'{label} {number}/{len(specifications)} {spec["model"]}/{spec["panel"]}'
            f' F1={mean["f1"]:.3f} precision={mean["precision"]:.3f} recall={mean["recall"]:.3f}'
            f' ({record["seconds"]:.1f}s)',flush=True)
    return records


COMPLEXITY = {'dummy':0,'logistic':1,'forest':2,'xgboost':3,'catboost':4}


def rank(records):
    return sorted([r for r in records if r['spec']['model']!='dummy'],
        key=lambda r:(-r['mean']['f1'],-r['mean']['recall'],COMPLEXITY[r['spec']['model']],
            len(FEATURES) if r['spec']['panel']=='expanded' else 28 if r['spec']['panel']=='core' else 14))


def baseline(cfg):
    specs = [dict(model='dummy',panel='clinical',params={})]
    for model,params in [('logistic',{'C':1.0,'class_weight':'balanced'}),
            ('forest',{'max_depth':5,'min_samples_leaf':5,'class_weight':'balanced'})]:
        for panel in ['clinical','core','expanded']:
            specs.append(dict(model=model,panel=panel,params=params))
    return compare(cfg,specs,'baseline')
