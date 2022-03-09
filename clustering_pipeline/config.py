from pathlib import Path


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
KMEANS_LABEL_COLUMN = 'kmeans_label'
PAM_LABEL_COLUMN = 'kmedoids_label'

# experiment settings
TRAIN_PERIOD = ('2016-01-31', '2020-12-31')
TEST_PERIOD = ('2021-01-31', '2021-12-31')

INITIAL_TRAIN_PERIOD = ('2016-01-31', '2017-12-31')    # for backtesing or walk-forward validation
TEST_PERIOD_LENGTH = 12    # months

MODEL_PARAM = {
    'PCA': {'n_components': 0.95},
    'KMeans': {'n_clusters': 20},
    'KMedoids': {'n_clusters': 20}
}

PATH_KMEANS_RESULT = PATH_WORKING_DIR / Path(f"kmeans_with_{MODEL_PARAM['KMeans']['n_clusters']}_centers.csv")
PATH_PAM_RESULT = PATH_WORKING_DIR / Path(f"kmedoids_with_{MODEL_PARAM['KMedoids']['n_clusters']}_medoids.csv")
