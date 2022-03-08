import logging
import config as config
import numpy as np
# import pandas as pd
# import category_encoders as ce

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class DataPreprocessor:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.non_feat = config.NON_FEATURE_COLUMNS
        self.cat_feat = config.CATEGORICAL_FEATURES
    
    def preprocess(self, data):
        """
        data: pd.DataFrame
        """
        logger = self.logger
        
        cols = data.columns.difference(self.non_feat).to_list()
        cat_cols = list(set(cols).intersection(self.cat_feat))
        num_cols = list(set(cols).difference(cat_cols))
        logger.info(f'{len(num_cols)} numeric, {len(cat_cols)} categorical')

        col_median = config.COLUMN_TO_IMPUTE_MEDIAN
        col_const = list(set(num_cols).difference(col_median))

        ppl_median = Pipeline([
            ('median', SimpleImputer(strategy='median')),
            ('power', PowerTransformer(method='yeo-johnson', standardize=True))
        ])
        ppl_const = Pipeline([
            ('constant', SimpleImputer(strategy='constant', fill_value=0.0)),
            ('power', PowerTransformer(method='yeo-johnson', standardize=True))
        ])
        transformer = ColumnTransformer(transformers=[
            ('ne1', ppl_median, col_median),
            ('ne2', ppl_const, col_const),
            ('ce', catEncoder(), cat_cols)
            # ('ce', ce.OneHotEncoder(), cat_cols)
        ], remainder='drop', n_jobs=-1)

        cleaned_data = transformer.fit_transform(data)
        feat_cols = col_median + col_const + cat_cols
        logger.info('impute and scale data')

        return cleaned_data, feat_cols


class catEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([[] for _ in range(len(X))])

