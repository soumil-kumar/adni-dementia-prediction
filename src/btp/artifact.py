"""Frozen predictor, calibration, and validated baseline JSON input contract."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .prepare import parse_lab
from .schema import FEATURES, CATEGORICAL, CSF


def logits(probabilities):
    p = np.clip(np.asarray(probabilities,dtype=float),1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)


@dataclass
class Predictor:
    pipeline: object
    calibrator: object
    threshold: float
    version: str
    spec: dict

    def predict_proba(self, frame):
        raw = self.pipeline.predict_proba(frame)[:,1]
        return self.calibrator.predict_proba(logits(raw))[:,1]

    def predict(self, frame):
        return self.predict_proba(frame) >= self.threshold


def input_frame(records):
    if isinstance(records,dict):
        records=[records]
    if not isinstance(records,list) or not records:
        raise ValueError('Expected a baseline object or nonempty list of objects')
    rows=[]
    for record in records:
        if not isinstance(record,dict):
            raise ValueError('Each record must be a JSON object')
        unknown=set(record)-set(FEATURES)-{'baseline_diagnosis'}
        if unknown:
            raise ValueError(f'Unsupported fields: {sorted(unknown)}')
        if record.get('baseline_diagnosis') not in ['CN','SMC','EMCI','LMCI']:
            raise ValueError('baseline_diagnosis must confirm CN, SMC, EMCI, or LMCI')
        row={c:record.get(c,np.nan) for c in FEATURES}
        for col in FEATURES:
            if row[col] is None:
                row[col]=np.nan
            if col in CSF:
                row[col],row[col+'_bound']=parse_lab(row[col])
            elif col in CATEGORICAL:
                if pd.notna(row[col]) and not isinstance(row[col],str):
                    raise ValueError(f'{col} must be a category string or null')
            elif pd.notna(row[col]):
                row[col]=float(row[col])
                if not np.isfinite(row[col]):
                    raise ValueError(f'{col} must be finite')
        for col,lo,hi in [('AGE',40,120),('PTEDUCAT',0,40),('MMSE_bl',0,30),
                ('MOCA_bl',0,30),('CDRSB_bl',0,18),('FAQ_bl',0,30)]:
            if pd.notna(row[col]) and not lo<=row[col]<=hi:
                raise ValueError(f'{col} must be between {lo} and {hi}')
        if pd.notna(row['APOE4']) and row['APOE4'] not in [0,1,2]:
            raise ValueError('APOE4 must be 0, 1, 2, or null')
        if sum(pd.notna(row[c]) for c in FEATURES)<3:
            raise ValueError('Supply at least three measured baseline features')
        rows.append(row)
    return pd.DataFrame(rows)
