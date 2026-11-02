import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from btp.preprocessing import PanelFeatures, preprocessing
from btp.metrics import best_threshold, scores


def test_validation_cannot_change_preprocessing_and_unseen_category_is_supported():
    train = pd.DataFrame({'AGE':[60.,70.,np.nan,80.], 'PTGENDER':['Male','Female','Male','Male'],
        'PTEDUCAT':[12,14,16,18], 'APOE4':[0,1,0,1]})
    p = Pipeline([('panel',PanelFeatures('clinical')),('preprocess',preprocessing()),
        ('model',DummyClassifier())]).fit(train,[0,1,0,1])
    before = p.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute'].statistics_.copy()
    p.predict_proba(train.assign(AGE=10000, PTGENDER='unseen'))
    after = p.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute'].statistics_
    np.testing.assert_array_equal(before,after)
    assert before[0] == 70


def test_missingness_filter_is_training_only_and_knn_handles_missing():
    train = pd.DataFrame({'AGE':[60.,70.,80.,90.,np.nan], 'MMSE_bl':[np.nan]*5})
    panel = PanelFeatures('clinical').fit(train)
    assert 'MMSE_bl' not in panel.columns_
    assert list(panel.transform(train.fillna(20)).columns) == ['AGE']
    values = preprocessing('knn').fit_transform(panel.transform(train))
    assert np.isfinite(values).all()


def test_threshold_tradeoff_and_majority_baseline():
    y = [0,0,1,1]
    p = [0.1,0.2,0.3,0.4]
    assert best_threshold(y,p) == 0.3
    assert scores(y,p,0.3)['f1'] == 1
    assert scores(y,[0]*4)['recall'] == 0
