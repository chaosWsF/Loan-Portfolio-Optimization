import logging
import config as config
import numpy as np
import pandas as pd

from pathlib import Path
from time_series_split import generate_datasets
from data_preprocessing import DataPreprocessor
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import brier_score_loss


def mth_diff(date1: str, date2: str) -> int:
    """
    date1, date2: YYYY-MM-DD
    """
    date1 = np.datetime64(date1[:-3])
    date2 = np.datetime64(date2[:-3])
    return ((date2 - date1).astype(int) + 1)


if __name__ == '__main__':
    working_dir = config.PATH_WORKING_DIR
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s - %(name)s - %(funcName)s: l%(lineno)d: %(message)s',
        handlers=[
            logging.FileHandler(working_dir / Path('kmeans_rolling.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    date_id = config.DATE_COLUMN
    train_period = config.TRAIN_PERIOD
    test_period = config.TEST_PERIOD
    start_date = train_period[0]
    end_date = test_period[1]

    loans_data = pd.read_parquet(config.PATH_RAW_DATA)
    loans_data = loans_data[(loans_data[date_id] >= start_date) & (loans_data[date_id] <= end_date)]
    logger.info('load loans data')

    Preprocessor = DataPreprocessor()
    data, features = Preprocessor.transform(loans_data)

    training_months = mth_diff(start_date, train_period[1])
    testing_months = mth_diff(test_period[0], end_date)

    datasets = generate_datasets(data, 'cal_day', start_date, end_date, training_months, testing_months)

    # TODO: add rolling kmeans by fixing centers. try to recursively use previous center but PCA needs an online version
    train, test = datasets[0]
    data = train.append(test).reset_index(drop=True)

    models = Pipeline([
        ('PCA', PCA(**config.PCA_PARAM)),
        # ('scaling', StandardScaler()),
        ('KMeans', KMeans(**config.KMEANS_PARAM)),
    ])

    models.fit(train[features].to_numpy())
    labels = models.predict(data[features].to_numpy())
    
    result_id = config.KMEANS_PARAM['n_clusters']
    loans_data[config.LABEL_COLUMN] = labels
    loans_data.to_csv(working_dir / f'cluster_labels_{result_id}.csv')
