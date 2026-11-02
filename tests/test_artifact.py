import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from btp.artifact import Predictor, input_frame, logits
from btp.training import build_pipeline


def test_saved_predictor_and_json_contract(tmp_path):
    rng=np.random.default_rng(42)
    frame=pd.DataFrame({'AGE':rng.uniform(60,85,50),'MMSE_bl':rng.uniform(15,30,50)})
    y=(frame.MMSE_bl<23).astype(int)
    spec={'model':'logistic','panel':'clinical','params':{'C':1.0}}
    pipeline=build_pipeline(spec).fit(frame,y)
    calibrator=LogisticRegression().fit(logits(pipeline.predict_proba(frame)[:,1]),y)
    predictor=Predictor(pipeline,calibrator,0.4,'test',spec)
    path=tmp_path/'model.joblib'
    joblib.dump(predictor,path)
    np.testing.assert_array_equal(joblib.load(path).predict_proba(frame),predictor.predict_proba(frame))
    parsed=input_frame({'baseline_diagnosis':'CN','AGE':70,'MMSE_bl':29,'ABETA_bl':'>1700'})
    assert parsed.ABETA_bl_bound.iloc[0]==1
    for record in [{'baseline_diagnosis':'AD','AGE':70},
            {'baseline_diagnosis':'CN','AGE':200,'MMSE_bl':28,'PTEDUCAT':10},
            {'baseline_diagnosis':'CN','RID':1}, {'AGE':70}]:
        with pytest.raises(ValueError):
            input_frame(record)
