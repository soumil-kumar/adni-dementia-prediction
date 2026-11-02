"""Conservative, auditable incident dementia labels from the harmonized DXSUM."""
import numpy as np
import pandas as pd


def build_labels(baseline, diagnoses):
    dx = diagnoses.copy()
    dx['date'] = pd.to_datetime(dx.EXAMDATE, format='%Y-%m-%d', errors='coerce')
    dx['DIAGNOSIS'] = pd.to_numeric(dx.DIAGNOSIS, errors='coerce')
    groups = dict(tuple(dx.groupby('RID')))
    rows = []
    for s in baseline.itertuples(index=False):
        index = s.index_date
        end = index + pd.DateOffset(years=5)
        v = groups.get(s.RID, dx.iloc[:0]).copy()
        valid = v[v.DIAGNOSIS.isin([1, 2, 3]) & v.date.notna()]
        conflicts = valid.groupby('date').DIAGNOSIS.nunique().gt(1).any()
        prior = valid[valid.date.le(index) & valid.DIAGNOSIS.eq(3)]
        events = valid[valid.date.gt(index) & valid.DIAGNOSIS.eq(3)].sort_values('date')
        event = events.date.min()
        last = valid.date.max()
        label, reason = np.nan, 'insufficient_followup'
        if s.DX_bl == 'AD':
            reason = 'baseline_dementia'
        elif pd.isna(s.DX_bl) or s.DX_bl not in ['CN', 'SMC', 'EMCI', 'LMCI']:
            reason = 'unknown_baseline_diagnosis'
        elif len(prior):
            reason = 'dementia_at_or_before_baseline'
        elif conflicts:
            reason = 'conflicting_same_day_diagnosis'
        elif (v.DIAGNOSIS.eq(3) & v.date.isna()).any():
            reason = 'undated_dementia'
        elif pd.notna(event) and event <= end:
            label, reason = 1, 'incident_dementia'
        elif pd.notna(last) and last >= end:
            label, reason = 0, 'observed_five_year_nonconverter'
        rows.append(dict(RID=s.RID, index_date=index, horizon_date=end,
            baseline_diagnosis=s.DX_bl, phase=s.COLPROT, first_dementia_date=event,
            last_diagnosis_date=last, Label=label, reason=reason,
            undated_diagnosis_rows=int(v.date.isna().sum())))
    return pd.DataFrame(rows)


def validate_handoffs(shared, a, b, expected_a, expected_b):
    for name, frame, expected in [('shared', shared, ['EXAMDATE']),
            ('person_A', a, expected_a), ('person_B', b, expected_b)]:
        if frame.RID.isna().any() or not frame.RID.is_unique:
            raise ValueError(f'{name}: missing or duplicate RID')
        if set(frame.columns) != {'RID', *expected}:
            raise ValueError(f'{name}: unexpected feature contract')
        if set(frame.RID) != set(shared.RID):
            raise ValueError(f'{name}: RID set differs from shared list')
