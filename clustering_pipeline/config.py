from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sqlalchemy import DATE


# paths
PATH_WORKING_DIR = Path('working_dir')
PATH_RAW_DATA = PATH_WORKING_DIR / Path('loans_data.pq')

# features
NON_FEATURE_COLUMNS = [
    'cal_day',
    'kmeans_label',
    'bus_ptnr_group',
    'naics_id',
    'impaired',
    'defaults_3_months',
    'defaults_6_months',
    'defaults_9_months',
    'defaults_12_months',
    'has_loan'
]

CATEGORICAL_FEATURES = ['trend_code', 'dunning_level', 'syndicated_loan']

DATE_COLUMN = 'cal_day'

COLUMN_TO_IMPUTE_MEDIAN = ['days_in_arrears', 'dunning_level_code']

ID_COLUMNS = ['cal_day', 'bus_ptnr_group']

BASE_LABEL_COLUMN = 'impaired'
LABEL_COLUMN = 'kmeans_label'

# experiment settings
TRAIN_PERIOD = ('2016-01-31', '2020-12-31')
TEST_PERIOD = ('2021-01-31', '2021-12-31')

INITIAL_TRAIN_PERIOD = ('2016-01-31', '2018-12-31')    # for backtesing or walk-forward validation
TEST_PERIOD_LENGTH = 12    # months

MODEL_PARAM = {
    'PCA': {'n_components': 0.95},
    'KMeans': {'n_clusters': 10}
}

KMEANS_ID = MODEL_PARAM['KMeans']['n_clusters']
PATH_KMEANS_RESULT = PATH_WORKING_DIR / Path(f'kmeans_with_{KMEANS_ID}_centers.csv')
