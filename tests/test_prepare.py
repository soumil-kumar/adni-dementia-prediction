import numpy as np
import pandas as pd
import pytest
from btp.prepare import parse_lab, history_at_baseline, neuro_at_baseline
from btp.schema import HISTORY, NEURO


def test_laboratory_bounds_and_invalid_strings():
    assert parse_lab('>1700') == (1700, 1)
    assert parse_lab('<80') == (80, -1)
    assert parse_lab('12.5') == (12.5, 0)
    assert np.isnan(parse_lab(np.nan)[0])
    with pytest.raises(ValueError):
        parse_lab('bad')


def test_absent_history_is_not_negative_and_future_is_excluded():
    baseline = pd.DataFrame({'RID': [1,2,3], 'index_date': pd.to_datetime(['2020-01-01']*3)})
    records = pd.DataFrame([dict(RID=rid,VISDATE=date,**dict.fromkeys(HISTORY,value))
        for rid,date,value in [(1,'2019-12-01',0),(1,'2020-02-01',1),(3,'2019-12-01',1)]])
    result,_ = history_at_baseline(records, baseline)
    assert result.loc[0,HISTORY[0]] == 0
    assert pd.isna(result.loc[1,HISTORY[0]])
    assert result.loc[2,HISTORY[0]] == 1


def test_neuro_date_filter_preference_and_conflict():
    baseline = pd.DataFrame({'RID': [1], 'index_date': [pd.Timestamp('2020-01-01')]})
    records = pd.DataFrame([dict(RID=1,VISDATE=date,VISCODE=visit,VISCODE2=visit,
        **dict.fromkeys(NEURO,value)) for date,visit,value in
        [('2019-12-01','sc',5),('2020-01-02','bl',99),('2018-01-01','bl',98)]])
    result,_ = neuro_at_baseline(records, baseline)
    assert result.loc[0,'TRABSCOR'] == 5
    conflict = pd.concat([records, records.iloc[[0]].assign(TRABSCOR=8)])
    result,pv = neuro_at_baseline(conflict, baseline)
    assert pd.isna(result.loc[0,'TRABSCOR'])
    assert 'conflicting_same_day_values' in pv.status.values
