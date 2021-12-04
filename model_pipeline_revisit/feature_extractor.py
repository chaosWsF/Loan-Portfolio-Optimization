import pandas as pd
import numpy as np
import logging

from tqdm import tqdm
from time import time
from joblib import Parallel, delayed
from pathlib import Path
from typing import List, Dict

import config as config


class FeatureExtractor:
    """
    Extract features for the loan loss training data.
    """
    id_columns: List[str] = config.ID_COLUMNS
    base_label_column: str = config.BASE_LABEL_COLUMN
    naics_mapping: Dict[int, str] = config.NAICS_MAPPING
    time_lags: List[int] = config.TIME_LAGS
    last_obs_features: List[str] = config.LAST_OBS_COLUMN_FEATURES
    categorical_features: List[str] = config.CATEGORICAL_FEATURES
    columns_to_take_percent_change: List[str] = config.COLUMNS_TO_TAKE_PERCENT_CHANGE
    columns_to_take_diff: List[str] = config.COLUMNS_TO_TAKE_DIFF
    columns_to_take_mean: List[str] = config.COLUMNS_TO_TAKE_MEAN
    columns_to_take_max: List[str] = config.COLUMNS_TO_TAKE_MAX

    def __init__(self, raw_data: pd.DataFrame):
        """
        :param raw_data: DataFrame of the raw data
        """

        self.logger = logging.getLogger(__name__)

        raw_data.columns = [col for col in raw_data.columns] # removed the lower temporarily
        self.original_columns = raw_data.columns
        self.training_data = raw_data.reset_index(drop=True)

        self.feature_columns = []
        self.feature_columns.extend(self.id_columns)
        self.feature_columns.append(self.base_label_column)

    def process_naics(self):
        self.training_data['naics_code'] = self.training_data['naics_id'].fillna(0).astype(str).str[:2]

        # taking the least granular naics classification
        self.training_data['naics_code'] = np.where(self.training_data['naics_code'] == '0.',
                                                    '0',
                                                    self.training_data['naics_code']).astype(int)

        self.training_data['naics_name'] = self.training_data['naics_code'].map(self.naics_mapping)

        self.logger.info('NAICS Processes')

    def get_features(self):

        tic = time()
        self.process_naics()
        self.training_data[self.categorical_features] = self.training_data[self.categorical_features].astype('category')
        self.feature_columns.extend(self.categorical_features)

        self.training_data = self.training_data.sort_values('cal_day')
        grouped = self.training_data.groupby('bus_ptnr_group')
        

        for lag in self.time_lags:
            self.logger.info(f'Adding features for lag {lag}')
            self.add_features_for_function(grouped,
                                           f'{lag}_diff',
                                           lambda x: x.diff(lag),
                                           self.columns_to_take_diff)
            self.add_features_for_function(grouped,
                                           f'{lag}_pct_change',
                                           lambda x: x.pct_change(lag),
                                           self.columns_to_take_percent_change)
            if lag > 1:
                self.add_features_for_function(grouped,
                                               f'{lag}_rolling_mean',
                                               lambda x: x.rolling(lag).mean(),
                                               self.columns_to_take_mean)
                self.add_features_for_function(grouped,
                                               f'{lag}_rolling_max',
                                               lambda x: x.rolling(lag).max(),
                                               self.columns_to_take_max)

        # Dedupe columns before saving
        # self.training_data = self.training_data.loc[:, ~self.training_data.columns.duplicated()].copy()
        last_obs_feature_names = {col: f'{col}_last_obs' for col in self.last_obs_features}
        self.training_data = self.training_data.rename(columns=last_obs_feature_names)
        self.feature_columns.extend(last_obs_feature_names.values())

        self.logger.info(f'Training Data Generated in {round((time() - tic) / 60, 2)} minutes')

        return self.training_data[self.feature_columns]

    def add_features_for_function(self, grouped_data, function_name, function, columns):

        feat_dfs = Parallel(n_jobs=-1)(delayed(function)(grouped_data[col]) for col in tqdm(columns))
        feat_df = (pd.concat(feat_dfs, axis=1)
                   .rename(columns=self.create_new_column_name_dict(columns, function_name))
                   .reset_index(drop=True))

        self.training_data = pd.concat([self.training_data, feat_df], axis=1)

    def create_new_column_name_dict(self, columns: List[str], function_name: str) -> Dict[str, str]:
        """
        Create a dictionary of column names for renaming calculated features
        :param columns: list of columns to generate new names for
        :param function_name: name of the feature function that is being applied to the column
        :return: dictionary with keys being the old column names and values the new column names
        """
        new_cols = {col: f'{col}_{function_name}' for col in columns}
        self.feature_columns.extend(new_cols.values())

        return new_cols


if __name__ == '__main__':
    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s - %(name)s - %(funcName)s: l%(lineno)d: %(message)s',
        handlers=[logging.StreamHandler()
                  ]
    )
    raw_data_path = config.PATH_RAW_DATA
    raw_data = pd.read_parquet(raw_data_path)
    
    #parse date + remove errors
    raw_data['cal_day'] = pd.to_datetime(raw_data['cal_day'], errors = 'coerce')
    raw_data = raw_data[raw_data['cal_day'].notnull()].copy()
    

    fe = FeatureExtractor(raw_data)
    fe.get_features()
    
    fe.training_data.to_parquet('feature_engineered_dataset.pq')
