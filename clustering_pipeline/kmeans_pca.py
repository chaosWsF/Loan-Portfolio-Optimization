import logging
import config as config
import numpy as np
import pandas as pd

from pathlib import Path
from data_preprocessing import DataPreprocessor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class Model:
    def __init__(self, parameters) -> None:
        """
        parameters: dict('PCA': dict(...), 'KMeans': dict(...), ...)
        """
        self.logger = logging.getLogger(__name__)
        self.param = parameters
    
    def construct(self):
        pca_param = self.param['PCA']
        kmeans_param = self.param['KMeans']

        self.ppl = Pipeline([
            ('PCA', PCA(**pca_param)),
            # ('scaling', StandardScaler()),
            ('KMeans', KMeans(**kmeans_param)),
        ])
        self.logger.info('initialize the model pipeline: PCA and KMeans')

    def fit(self, X):
        """
        X: array-like
        """
        self.construct()
        self.ppl.fit(X)
        self.logger.info('fitting')
    
    def predict(self, X):
        """"
        fitting before predicting
        """
        labels = self.ppl.predict(X)
        self.logger.info('getting labels')
        return labels


def generate_labels(data, train_period, test_period):
    """
    data: pd.DataFrame; train_period, test_period: tuple(str, str)
    """
    date_id = config.DATE_COLUMN

    Preprocessor = DataPreprocessor()
    df, features = Preprocessor.transform(data)
    
    train_idx = df[date_id].between(*train_period, inclusive='both')
    train = df.loc[train_idx, features].to_numpy()
    test_idx = df[date_id].between(*test_period, inclusive='both')
    test = df.loc[test_idx, features].to_numpy()
    logging.getLogger(__name__).info(f'split train and test with the size of {len(train)} and {len(test)}')

    model = Model(config.MODEL_PARAM)
    model.fit(train)

    X = np.append(train, test, axis=0)
    labels = model.predict(X)
    return labels


def mk_env(start_date, end_date, logging_file):
    working_dir = config.PATH_WORKING_DIR
    working_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s - %(name)s - %(funcName)s: l%(lineno)d: %(message)s',
        handlers=[
            logging.FileHandler(working_dir / Path(logging_file)),
            logging.StreamHandler()
        ]
    )

    loans_data = pd.read_parquet(config.PATH_RAW_DATA)
    selected_dates = loans_data[config.DATE_COLUMN].between(start_date, end_date, inclusive='both')
    loans_data = loans_data[selected_dates]

    return loans_data


if __name__ == '__main__':
    train_period = config.TRAIN_PERIOD
    test_period = config.TEST_PERIOD

    loans_data = mk_env(train_period[0], test_period[1], 'kmeans_pca.log')

    logger = logging.getLogger(__name__)

    labels = generate_labels(loans_data, train_period, test_period)
    loans_data[config.KMEANS_LABEL_COLUMN] = labels
    loans_data.to_csv(config.PATH_KMEANS_RESULT, index=False)
    logger.info('store results')
