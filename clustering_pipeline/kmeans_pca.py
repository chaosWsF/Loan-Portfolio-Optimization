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
        self.logger.info('initialize the model pipeline')

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
        self.logger('getting labels')
        return labels


def generate_labels(data, train_period=config.TRAIN_PERIOD, test_period=config.TEST_PERIOD):
    """
    data: pd.DataFrame; train_period, test_period: tuple(str, str)
    """
    working_dir = config.PATH_WORKING_DIR
    working_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s - %(name)s - %(funcName)s: l%(lineno)d: %(message)s',
        handlers=[
            logging.FileHandler(working_dir / Path('kmeans_pca.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    date_id = config.DATE_COLUMN

    start_date = train_period[0]
    end_date = test_period[1]
    data = data[(data[date_id] >= start_date) & (data[date_id] <= end_date)]

    Preprocessor = DataPreprocessor()
    df, features = Preprocessor.transform(data)
    
    train_idx = df[date_id].between(*train_period, inclusive='both')
    train = df.loc[train_idx, features].to_numpy()
    test_idx = df[date_id].between(*test_period, inclusive='both')
    test = df.loc[test_idx, features].to_numpy()
    logger.info(f'split train and test with the size of {len(train)} and {len(test)}')

    model = Model(config.MODEL_PARAM)
    model.fit(train)

    X = np.append(train, test, axis=0)
    labels = model.predict(X)

    return labels


if __name__ == '__main__':
    loans_data = pd.read_parquet(config.PATH_RAW_DATA)
    labels = generate_labels(loans_data)
    loans_data[config.LABEL_COLUMN] = labels
    loans_data.to_csv(config.PATH_KMEANS_RESULT)
