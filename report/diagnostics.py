"""Descriptive report diagnostics; never fit or change the frozen predictor."""
from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, fbeta_score
from sklearn.calibration import calibration_curve
from btp.splits import load_partition
from btp.common import read_config

root=Path(__file__).resolve().parent
cfg=read_config(root.parent/'config.json')
model=joblib.load(root.parent/'local/model.joblib')
d=load_partition(cfg,'validation'); y=d.Label.astype(int).to_numpy()
p=model.predict_proba(d); raw=model.pipeline.predict_proba(d)[:,1]
def ece(p):
    bins=np.minimum((p*10).astype(int),9)
    return sum(np.mean(bins==i)*abs(y[bins==i].mean()-p[bins==i].mean()) for i in range(10) if (bins==i).any())
r={'before':{'brier':brier_score_loss(y,raw),'ece':ece(raw)},
   'after':{'brier':brier_score_loss(y,p),'ece':ece(p)},'f2':fbeta_score(y,p>=model.threshold,beta=2),
   'f05':fbeta_score(y,p>=model.threshold,beta=.5)}
rng=np.random.default_rng(42); groups=[np.flatnonzero(y==c) for c in [0,1]]
boot={b:[] for b in [.5,2]}
for _ in range(2000):
    ids=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
    for b in boot:boot[b].append(fbeta_score(y[ids],p[ids]>=model.threshold,beta=b))
for b,v in boot.items():r['f'+str(b)+'_ci']=np.quantile(v,[.025,.975]).tolist()
fig,ax=plt.subplots(figsize=(6.5,3.7))
for name,values in [('Before calibration',raw),('After calibration',p)]:
    observed,predicted=calibration_curve(y,values,n_bins=10,strategy='quantile')
    ax.plot(predicted,observed,'o-',label=name)
ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='Mean predicted probability',ylabel='Observed dementia fraction')
ax.legend();fig.tight_layout();fig.savefig(root/'figures/calibration.png',dpi=170);plt.close(fig)
pre=model.pipeline.named_steps['preprocessing']; names=pre.get_feature_names_out()
coef=model.pipeline.named_steps['model'].coef_[0]
top=np.argsort(abs(coef))[-12:]
fig,ax=plt.subplots(figsize=(7,4))
ax.barh(np.array(names)[top],coef[top],color=['#aa5533' if x<0 else '#23678c' for x in coef[top]])
ax.set(xlabel='Standardized-model log-odds coefficient',title='Largest absolute coefficients (not SHAP)')
fig.tight_layout();fig.savefig(root/'figures/coefficients.png',dpi=170);plt.close(fig)
r['top_coefficients']=[{'term':str(names[i]),'coefficient':float(coef[i])} for i in top[::-1]]
r['errors']={}
for name,mask in [('false_positive',(y==0)&(p>=model.threshold)),('false_negative',(y==1)&(p<model.threshold))]:
    x=d.loc[mask]
    r['errors'][name]={'n':len(x),'baseline_diagnoses':x.baseline_diagnosis.value_counts().to_dict(),
       'median_age':float(x.AGE.median()),'median_MMSE':float(x.MMSE_bl.median()),
       'median_CDRSB':float(x.CDRSB_bl.median())}
for name in ['validation_plots.png','missingness.png']:shutil.copyfile(root.parent/'results'/name,root/'figures'/name)
(root/'diagnostics.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
