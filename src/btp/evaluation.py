"""One-time held-out evaluation; only aggregate summaries may be published."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve
from .common import dump, sha256
from .metrics import scores
from .splits import load_partition, ensure_splits


def intervals(y, p, threshold, iterations=2000, seed=42):
    rng=np.random.default_rng(seed)
    y=np.asarray(y,dtype=int)
    groups=[np.flatnonzero(y==c) for c in [0,1]]
    values={k:[] for k in ['accuracy','precision','recall','f1','specificity','balanced_accuracy',
        'roc_auc','average_precision','brier']}
    for _ in range(iterations):
        ids=np.concatenate([rng.choice(group,size=len(group),replace=True) for group in groups])
        metric=scores(y[ids],p[ids],threshold)
        for key in values:
            values[key].append(metric[key])
    return {key:np.quantile(value,[0.025,0.975]).tolist() for key,value in values.items()}


def evaluate(cfg):
    local=Path(cfg['local_dir'])
    manifest=json.loads((local/'frozen_manifest.json').read_text())
    for name,key in [('model.joblib','model_sha256'),('master_baseline.csv','master_sha256'),
            ('splits.json','split_sha256'),('selection.json','selection_sha256')]:
        if sha256(local/name)!=manifest[key]:
            raise ValueError(f'Frozen artifact mismatch: {name}')
    path=local/'validation_results.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous['model_sha256']!=manifest['model_sha256']:
            raise RuntimeError('Cannot replace an existing validation result')
        print(json.dumps(previous['primary'],indent=2),flush=True)
        return previous
    dump(local/'validation_opened.json',{'model_sha256':manifest['model_sha256'],
        'note':'Training is permanently frozen for this experiment directory.'})
    predictor=joblib.load(local/'model.joblib')
    data=load_partition(cfg,'validation')
    y=data.Label.astype(int).to_numpy()
    p=predictor.predict_proba(data)
    primary=scores(y,p,predictor.threshold)
    uncertainty=intervals(y,p,predictor.threshold,cfg['bootstrap_iterations'],cfg['seed'])
    subgroup={}
    mappings={'baseline_group':data.baseline_diagnosis.map(
        {'CN':'CN/SMC','SMC':'CN/SMC','EMCI':'MCI','LMCI':'MCI'}), 'phase':data.phase}
    for kind,groups in mappings.items():
        for group in groups.dropna().unique():
            mask=groups.eq(group).to_numpy()
            subgroup[f'{kind}:{group}']=scores(y[mask],p[mask],predictor.threshold)
    # Baseline trained only on development: never infer majority from validation labels.
    dev=load_partition(cfg,'development')
    majority=int(dev.Label.mean()>=0.5)
    result=dict(model_sha256=manifest['model_sha256'],threshold=predictor.threshold,
        selected=manifest['selected']['spec'],primary=primary,confidence_intervals_95=uncertainty,
        at_threshold_0_5=scores(y,p,0.5),majority_baseline=scores(y,np.full(len(y),majority)),
        subgroups=subgroup,split_counts=ensure_splits(cfg)['counts'],
        bootstrap=dict(iterations=cfg['bootstrap_iterations'],seed=cfg['seed'],
            method='stratified subject bootstrap, fixed model and threshold; conditional on held-out class counts'),
        caveats=['Any dementia in ADNI, not specifically Alzheimer-attributed dementia.',
            'Small held-out sample; subgroup metrics can be unstable.',
            'Selection and incomplete-follow-up exclusions limit generalizability.',
            'Probabilities are research estimates, not a validated clinical service.'])
    dump(path,result)
    pd.DataFrame({'RID':data.RID,'Label':y,'probability':p,
        'predicted_positive':p>=predictor.threshold}).to_csv(local/'validation_predictions.csv',index=False)
    plots(local,data,p,predictor.threshold)
    print(json.dumps(primary,indent=2),flush=True)
    return result


def plots(local,data,p,threshold):
    y=data.Label.astype(int).to_numpy()
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    fpr,tpr,_=roc_curve(y,p)
    axes[0,0].plot(fpr,tpr,label='Frozen model')
    axes[0,0].plot([0,1],[0,1],'--',color='gray')
    axes[0,0].set(xlabel='False positive rate',ylabel='Recall',title='Final validation ROC')
    precision,recall,_=precision_recall_curve(y,p)
    axes[0,1].plot(recall,precision)
    axes[0,1].axhline(y.mean(),linestyle='--',color='gray')
    axes[0,1].set(xlabel='Recall',ylabel='Precision',title='Precision-recall')
    observed,predicted=calibration_curve(y,p,n_bins=7,strategy='quantile')
    axes[1,0].plot(predicted,observed,'o-')
    axes[1,0].plot([0,1],[0,1],'--',color='gray')
    axes[1,0].set(xlabel='Predicted probability',ylabel='Observed fraction',title='Calibration')
    for value,label in [(0,'Nonconverter'),(1,'Converter')]:
        axes[1,1].hist(p[y==value],bins=np.linspace(0,1,16),alpha=0.6,label=label)
    axes[1,1].axvline(threshold,color='black',linestyle='--',label='Frozen threshold')
    axes[1,1].set(xlabel='Probability',ylabel='Subjects',title='Risk distributions')
    axes[1,1].legend()
    fig.tight_layout()
    fig.savefig(local/'validation_plots.png',dpi=160)
    plt.close(fig)
    audit=json.loads((local/'data_audit.json').read_text())
    missing=pd.Series(audit['eligible_missing_fraction']).sort_values(ascending=False)
    fig,ax=plt.subplots(figsize=(10,6))
    missing.head(25).sort_values().plot.barh(ax=ax)
    ax.set(xlabel='Missing fraction in eligible cohort',title='25 least complete candidate features')
    fig.tight_layout()
    fig.savefig(local/'missingness.png',dpi=160)
    plt.close(fig)


def publish_aggregates(cfg):
    local=Path(cfg['local_dir'])
    target=Path(cfg['config_path']).parent/'results'
    target.mkdir(exist_ok=True)
    for name in ['data_audit.json','feature_registry.json','validation_results.json','selection.json']:
        dump(target/name,json.loads((local/name).read_text()))
    experiments=json.loads((local/'experiments.json').read_text())
    dump(target/'development_comparison.json',{'records':experiments['records']})
    import shutil
    for name in ['validation_plots.png','missingness.png']:
        shutil.copyfile(local/name,target/name)
