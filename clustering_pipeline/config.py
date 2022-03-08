from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


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

COLUMN_TO_IMPUTE_MEDIAN = ['days_in_arrears', 'dunning_level_code']

ID_COLUMNS = ['cal_day', 'bus_ptnr_group']

BASE_LABEL_COLUMN = 'impaired'
LABEL_COLUMN = 'kmeans_label'

# Experiment
EXPERIMENT_NAME = 'train_with_pipelines'

INITIAL_TRAIN_PERIOD = ('2013-03-31', '2015-03-01')

TRAIN_PERIOD = ('2013-04-01', '2018-12-31')
TEST_PERIOD = ('2019-04-01', '2019-12-31')

# Models
pca = {
    'ESTIMATOR': PCA(),
    'PARAMS': {
        'n_components': 0.95
    }
}

kmeans = {
    'ESTIMATOR': KMeans(),
    'PARAMS': {
        'n_clusters': 10
    }
}

