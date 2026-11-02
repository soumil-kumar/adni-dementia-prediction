"""Read-only ingestion of original handoffs; corrected outputs are always local."""
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .cohort import build_labels, validate_handoffs
from .common import dump, sha256, versions
from .schema import A_FEATURES, B_FEATURES, FEATURES, CATEGORICAL, CSF, NEURO, HISTORY, registry


def parse_lab(value):
    if pd.isna(value):
        return np.nan, 0
    match = re.fullmatch(r'\s*([<>]?)\s*(\d+(?:\.\d+)?)\s*', str(value))
    if not match:
        if str(value).strip() in ['-1', '-4', '-9', 'NA', '']:
            return np.nan, 0
        raise ValueError(f'Unrecognized laboratory value: {value!r}')
    return float(match[2]), {'': 0, '<': -1, '>': 1}[match[1]]


def history_at_baseline(history, baseline):
    records = history.merge(baseline[['RID', 'index_date']], on='RID', validate='many_to_one')
    records['date'] = pd.to_datetime(records.VISDATE, format='%Y-%m-%d', errors='coerce')
    records = records[records.date.le(records.index_date)]
    values = records[HISTORY].apply(pd.to_numeric, errors='coerce')
    values = values.where(values.isin([0, 1]))
    grouped = pd.concat([records[['RID']], values], axis=1).groupby('RID')[HISTORY].max()
    result = baseline[['RID']].merge(grouped, on='RID', how='left', validate='one_to_one')
    provenance = records[['RID', 'date']].assign(source='MEDHIST')
    return result, provenance


def neuro_at_baseline(neuro, baseline, lookback=90):
    records = neuro.merge(baseline[['RID', 'index_date']], on='RID', validate='many_to_one')
    records['date'] = pd.to_datetime(records.VISDATE, format='%Y-%m-%d', errors='coerce')
    records['days_before'] = (records.index_date - records.date).dt.days
    visit = records.VISCODE2.fillna(records.VISCODE)
    records['visit'] = visit
    records = records[visit.isin(['sc', 'bl']) & records.days_before.between(0, lookback)].copy()
    rows, provenance = [], []
    groups = dict(tuple(records.groupby('RID')))
    for rid in baseline.RID:
        record = {'RID': rid}
        available = groups.get(rid, records.iloc[:0])
        for col in NEURO:
            values = pd.to_numeric(available[col], errors='coerce')
            candidates = available[values.ge(0)].copy()
            candidates[col] = values[values.ge(0)]
            preferred = 'sc' if col in ['LIMMTOTAL', 'LDELTOTAL'] else 'bl'
            candidates['priority'] = candidates.visit.ne(preferred).astype(int)
            candidates = candidates.sort_values(['priority', 'date'], ascending=[True, False])
            if candidates.empty:
                record[col] = np.nan
                continue
            best = candidates.iloc[0]
            tied = candidates[candidates.priority.eq(best.priority) & candidates.date.eq(best.date)]
            if tied[col].nunique() > 1:
                record[col] = np.nan
                provenance.append(dict(RID=rid, feature=col, source='NEUROBAT',
                    date=best.date, status='conflicting_same_day_values'))
            else:
                record[col] = float(best[col])
                provenance.append(dict(RID=rid, feature=col, source='NEUROBAT',
                    date=best.date, status='selected'))
        rows.append(record)
    return pd.DataFrame(rows), pd.DataFrame(provenance)


def prepare(cfg):
    source, local = Path(cfg['data_dir']), Path(cfg['local_dir'])
    if (local / 'validation_results.json').exists():
        raise RuntimeError('Validation has already been opened. Use a new local_dir for a new experiment.')
    local.mkdir(parents=True, exist_ok=True)
    paths = {
        'a': source / 'person_A_features.csv', 'b': source / 'person_B_features.csv',
        'shared': source / 'shared_subject_list.csv',
        'adni': source / 'tadpole_challenge_201911210/tadpole_challenge/ADNIMERGE.csv',
        'dx': source / 'DXSUM_05Sep2026.csv', 'neuro': source / 'NEUROBAT_06Sep2026.csv',
        'history': source / 'MEDHIST_06Sep2026.csv', 'old_labels': source / 'outcome_labels.csv',
    }
    tables = {k: pd.read_csv(p, low_memory=False) for k, p in paths.items()}
    a, b, shared = [tables[x] for x in ['a', 'b', 'shared']]
    validate_handoffs(shared, a, b, A_FEATURES, B_FEATURES)
    baseline = tables['adni'].loc[tables['adni'].VISCODE.eq('bl')].copy()
    if not baseline.RID.is_unique or set(baseline.RID) != set(shared.RID):
        raise ValueError('ADNIMERGE baseline does not match shared subjects')
    baseline['index_date'] = pd.to_datetime(baseline.EXAMDATE, format='%d-%m-%Y', errors='raise')
    dates = baseline[['RID', 'index_date']].merge(shared, on='RID', validate='one_to_one')
    if not dates.index_date.eq(pd.to_datetime(dates.EXAMDATE, format='%Y-%m-%d')).all():
        raise ValueError('Shared and original baseline dates disagree')
    labels = build_labels(baseline, tables['dx'])
    labels.to_csv(local / 'outcome_labels_corrected.csv', index=False)
    new_neuro, npv = neuro_at_baseline(tables['neuro'], baseline, cfg['screening_lookback_days'])
    new_history, hpv = history_at_baseline(tables['history'], baseline)
    corrected_b = (b[['RID'] + [c for c in B_FEATURES if c not in NEURO + HISTORY]]
        .merge(new_neuro, on='RID', validate='one_to_one')
        .merge(new_history, on='RID', validate='one_to_one'))
    changes = {}
    old = b.set_index('RID').sort_index()
    new = corrected_b.set_index('RID').sort_index()
    for col in NEURO + HISTORY:
        changes[col] = int((~(old[col].eq(new[col]) | (old[col].isna() & new[col].isna()))).sum())
    corrected_b[['RID'] + B_FEATURES].to_csv(local / 'person_B_features_corrected.csv', index=False)
    a.to_csv(local / 'person_A_features_preserved.csv', index=False)
    master = a.merge(corrected_b, on='RID', validate='one_to_one')
    assert len(master) == len(shared) and len(master.columns) == 61
    for col in CSF:
        parsed = master[col].map(parse_lab)
        master[col] = parsed.map(lambda v: v[0])
        master[col + '_bound'] = parsed.map(lambda v: v[1])
    for col in set(FEATURES) - set(CATEGORICAL + CSF):
        master[col] = pd.to_numeric(master[col], errors='raise')
        # RAVLT learning and forgetting can legitimately be negative.
        if not col.startswith('RAVLT_'):
            master[col] = master[col].mask(master[col].isin([-1, -4, -9]))
    master = master.merge(labels, on='RID', validate='one_to_one')
    master.to_csv(local / 'master_baseline.csv', index=False)
    pd.concat([npv, hpv], ignore_index=True).to_csv(local / 'feature_provenance.csv', index=False)
    labels.merge(tables['old_labels'], on='RID', suffixes=('_corrected', '_old'),
        validate='one_to_one').to_csv(local / 'label_reconciliation.csv', index=False)
    eligible = master[master.Label.notna()]
    missing = eligible[FEATURES].isna().mean().to_dict()
    report = dict(total_subjects=len(master), labeled_subjects=len(eligible),
        cohort_flow=labels.reason.value_counts().to_dict(),
        labels=eligible.Label.astype(int).value_counts().sort_index().to_dict(),
        source_features=60, bound_indicators=3, corrected_b_cells=changes,
        eligible_missing_fraction=missing,
        diagnosis_missing_dates=int(tables['dx'].EXAMDATE.isna().sum()),
        diagnosis_duplicate_subject_dates=int(tables['dx'].duplicated(['RID', 'EXAMDATE']).sum()),
        limitations=['Baseline ADNIMERGE _bl values use merged-table baseline provenance; individual acquisition dates are not all supplied.',
            'Five-year negatives are based on observed diagnoses, not continuous surveillance.',
            'Short-follow-up exclusion can introduce selection bias.',
            'MH3HEAD covers head/eyes/ears/nose/throat; it is not a specific traumatic brain injury measure.'],
        audit_reconciliation={'original_total_fields':3066, 'tier1':34, 'tier2':33,
            'arithmetic_remainder':2999, 'pdf_reported_tier3':2943,
            'primary_source_features':60, 'deferred_plasma_features':7})
    dump(local / 'data_audit.json', report)
    dump(local / 'feature_registry.json', registry())
    dump(local / 'prepare_manifest.json', dict(inputs={k:dict(filename=p.name, sha256=sha256(p))
        for k,p in paths.items()}, master_sha256=sha256(local / 'master_baseline.csv'),
        config=cfg, versions=versions()))
    print(f'Prepared {len(master)} baseline rows; {len(eligible)} known outcomes.', flush=True)
    print(labels.reason.value_counts().to_string(), flush=True)
    return report
