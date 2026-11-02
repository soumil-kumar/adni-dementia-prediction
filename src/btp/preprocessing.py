"""All learned feature decisions live inside the fitted sklearn pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .schema import PANELS, CATEGORICAL, CSF, MRI


class PanelFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, panel='expanded', normalize_mri=False):
        self.panel = panel
        self.normalize_mri = normalize_mri

    def _derive(self, frame):
        cols = list(PANELS[self.panel])
        cols += [c + '_bound' for c in CSF if c in cols]
        result = frame.reindex(columns=cols).copy()
        if self.normalize_mri and 'ICV_bl' in result:
            icv = pd.to_numeric(result['ICV_bl'], errors='coerce').where(lambda x: x > 0)
            for col in MRI:
                if col != 'ICV_bl' and col in result:
                    result[col] = pd.to_numeric(result[col], errors='coerce') / icv
        return result

    def fit(self, X, y=None):
        frame = self._derive(X)
        self.columns_ = [c for c in frame if frame[c].isna().mean() <= 0.8
            and frame[c].nunique(dropna=True) > 1]
        if not self.columns_:
            raise ValueError('No usable training features')
        return self

    def transform(self, X):
        return self._derive(X)[self.columns_]


def categorical_columns(frame):
    return [c for c in frame if c in CATEGORICAL]


def numeric_columns(frame):
    return [c for c in frame if c not in CATEGORICAL]


def preprocessing(imputation='median'):
    if imputation == 'median':
        numeric = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True,
            keep_empty_features=True)), ('scale', StandardScaler())])
    elif imputation == 'knn':
        numeric = FeatureUnion([
            ('values', Pipeline([('scale', StandardScaler()),
                ('impute', KNNImputer(n_neighbors=5, weights='distance', keep_empty_features=True))])),
            ('missing', MissingIndicator(features='all'))])
    else:
        raise ValueError(f'Unknown imputation: {imputation}')
    categorical = Pipeline([('missing', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('numeric', numeric, numeric_columns),
        ('categorical', categorical, categorical_columns)], sparse_threshold=0)
