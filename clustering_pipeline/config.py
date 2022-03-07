from pathlib import Path
from scipy.stats import uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# paths
PATH_WORKING_DIR = Path('working_dir')
PATH_RAW_DATA = PATH_WORKING_DIR / Path('loans_data.pq')

# Feature Engineering
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

ID_COLUMNS = ['cal_day', 'bus_ptnr_group']

BASE_LABEL_COLUMN = 'impaired'
LABEL_COLUMN = 'kmeans_label'

# Experiment
EXPERIMENT_NAME = 'train_with_pipelines'

INITIAL_TRAIN_PERIOD = ('2013-03-31', '2015-03-01')

# train and test periods
TRAIN_PERIOD = ('2013-04-01', '2018-12-31')
TEST_PERIOD = ('2019-04-01', '2019-12-31')

TIME_LAGS = [1, 3, 6, 12]

# Models
DECISION_TREE_PARAMS = {
    'max_depth': range(2, 3),
    'criterion': ['entropy', 'gini'],
    'class_weight': ['balanced']
}

DECISION_TREE = {
    'ESTIMATOR': DecisionTreeClassifier(),
    'PARAMS': DECISION_TREE_PARAMS
}

RANDOM_FOREST_PARAMS = {
    'criterion': ['entropy', 'gini'],
    'bootstrap': [False],
    'n_estimators': [200],
    'max_depth': range(8, 15),
    # 'min_samples_split': [200, 500, 800, 1000],
    # 'min_samples_leaf': Random(Discrete([100, 200, 400, 700]))
    'max_features': uniform(0.1, 0.4),
    #     'min_weight_fraction_leaf': uniform(0, 0.05),
    'n_jobs': [-1],
    'class_weight': ['balanced'],
    'verbose': [1]
}
RANDOM_FOREST = {
    'ESTIMATOR': RandomForestClassifier(),
    'PARAMS': RANDOM_FOREST_PARAMS
}

MODELS = [('decision_tree', DECISION_TREE)]

# CV
NUM_INNER_SPLITS = 5
NUM_OUTER_SPLITS = 5
NUM_ITER = 1
CV_SCORE = 'f1_macro'
