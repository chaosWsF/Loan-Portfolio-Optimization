import pandas as pd
import numpy as np
import multiprocessing
import logging

from joblib import Parallel, delayed
from tqdm.auto import tqdm, trange
from pathlib import Path

import config as config


class DataPipeline:
    """
    Class to manage processing of the data.
    """

    def __init__(self,
                 base_label_column = config.BASE_LABEL_COLUMN,
                 label_column=config.LABEL_COLUMN,
                 non_feature_columns=config.NON_FEATURE_COLUMNS
                 ):
        self.base_label_column = base_label_column
        self.label_column = label_column
        self.non_feature_columns = non_feature_columns

        self.logger = logging.getLogger(__name__)

    def create_label(self, data: pd.DataFrame):
        self.logger.info('Applying labels')

        def helper(df):
            df[self.label_column] = df[self.base_label_column].rolling(3).max().shift(-3)
            df['evaluation_date'] = df['prediction_date'].shift(-3)

            return df

        # create features
        data = pd.concat(Parallel(n_jobs=multiprocessing.cpu_count() - 1)(delayed(helper)
                                                                          (df_grouped) for _, df_grouped in
                                                                          tqdm(data.groupby(['bus_ptnr_group']))))
        data = data.loc[~data[self.label_column].isna()]

        return data

    def log_dataset(self, data: pd.DataFrame, experiment_path: Path):
        data_statistics = {}
        data_statistics['Min Date:'] = data['prediction_date'].min()
        data_statistics['Max Date:'] = data['prediction_date'].max()
        data_statistics['Number of Observations'] = data.index.max()
        data_statistics['Number of BPs'] = data['bus_ptnr_group'].nunique()
        target_counts = data[self.label_column].value_counts()
        data_statistics['Number of Class 0:'] = target_counts[0]
        data_statistics['Number of Class 1:'] = target_counts[1]
        data_statistics['Positive Balance'] = target_counts[1] / target_counts[0]

        for measure, value in data_statistics.items():
            self.logger.info(f'{measure} {value}')

        pd.DataFrame.from_records([data_statistics]).to_csv(experiment_path / 'data_statistics.csv')

    def clean_data(self, data: pd.DataFrame):
        data = data.rename(columns={'cal_day': 'prediction_date'})
        data = self.create_label(data)
        data = data.loc[data[self.base_label_column] == 0]
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.sort_values(['prediction_date'])

        return data

    def train_test_split(self, data: pd.DataFrame):
        self.logger.info('Start Train-Test Split')
        data_train_period = data['prediction_date'].between(config.TRAIN_PERIOD[0], config.TRAIN_PERIOD[1])
        data_test_period = data['prediction_date'].between(config.TEST_PERIOD[0], config.TEST_PERIOD[1])
        train = data.loc[data_train_period]
        test = data.loc[data_test_period]
        self.logger.info('Train Test Split Finished')

        return train, test
