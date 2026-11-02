import pandas as pd
import pytest
from btp.cohort import build_labels, validate_handoffs


def label(records, diagnosis='CN'):
    baseline = pd.DataFrame([dict(RID=1, index_date=pd.Timestamp('2020-02-29'),
        DX_bl=diagnosis, COLPROT='ADNI2')])
    dx = pd.DataFrame([dict(RID=1, EXAMDATE=date, DIAGNOSIS=value) for date, value in records])
    return build_labels(baseline, dx).iloc[0]


@pytest.mark.parametrize('date,expected', [('2025-02-28', 1), ('2025-03-01', 0)])
def test_calendar_boundary(date, expected):
    assert label([(date, 3)]).Label == expected


def test_baseline_dementia_and_short_followup():
    assert label([('2021-01-01', 3)], 'AD').reason == 'baseline_dementia'
    assert label([('2020-02-29', 3)]).reason == 'dementia_at_or_before_baseline'
    assert pd.isna(label([('2024-01-01', 1)]).Label)


def test_duplicates_conflicts_undated_and_reversion():
    assert label([('2021-01-01', 3), ('2021-01-01', 3)]).Label == 1
    assert label([('2021-01-01', 3), ('2021-01-01', 2)]).reason.startswith('conflicting')
    assert label([(None, 3)]).reason == 'undated_dementia'
    assert label([('2021-01-01', 3), ('2026-01-01', 1)]).Label == 1


def test_handoff_rejects_duplicate_and_different_subjects():
    s = pd.DataFrame({'RID': [1], 'EXAMDATE': ['2020-01-01']})
    a = pd.DataFrame({'RID': [1], 'a': [2]})
    b = pd.DataFrame({'RID': [1], 'b': [3]})
    validate_handoffs(s, a, b, ['a'], ['b'])
    with pytest.raises(ValueError):
        validate_handoffs(s, pd.concat([a, a]), b, ['a'], ['b'])
    with pytest.raises(ValueError):
        validate_handoffs(s, a, b.assign(RID=2), ['a'], ['b'])
