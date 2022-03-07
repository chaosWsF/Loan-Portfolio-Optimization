from pathlib import Path
from scipy.stats import uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from custom_xgboost import SampleWeightedXGBoost

CONFIG = {}

# Paths
PATH_WORKING_DIR = Path('working_dir')
PATH_RAW_DATA = PATH_WORKING_DIR / Path('loans_data.pq')
PATH_PROCESSED_DATA = PATH_WORKING_DIR / Path('training_set_processed_borrower_level.pkl')

# train and test
TRAIN_PERIOD = ('2013-04-01', '2018-12-31')
TEST_PERIOD = ('2019-04-01', '2019-12-31')

# TRAIN_PERIOD = ('2013-04-01', '2015-12-31')
# TEST_PERIOD = ('2017-01-01', '2017-12-31')

# CV
NUM_INNER_SPLITS = 5
NUM_OUTER_SPLITS = 5
NUM_ITER = 1
CV_SCORE = 'f1_macro'

# Feature Engineering
ID_COLUMNS = ['cal_day', 'bus_ptnr_group']

TIME_LAGS = [1, 3, 6, 12]

CATEGORICAL_FEATURES = ['naics_name', 'dunning_level']

COLUMNS_TO_TAKE_PERCENT_CHANGE = ['BRR',
        'BEACON',
        'dunning_level_code',
        'days_in_arrears',
        'Oustanding_principle_on_posting_date',
        'percentage_rate',
        'transactions',
        'abs_transactions',
        'transactions_db',
        'transactions_cr',
        'n_transactions',
        'SUB_SYSTEM_DP',
        'SUB_SYSTEM_FD',
        'SUB_SYSTEM_IN',
        'SUB_SYSTEM_LN',
        'SUB_SYSTEM_RF',
        'SUB_SYSTEM_RP',
        'SUB_SYSTEM_SP',
        'SUB_SYSTEM_TF',
        'transaction_type_Loan_Disbursement',
        'transaction_type_Payment_Distrib_Loan',
        'transaction_type_Loan_Payment',
        'transaction_type_Bank_Trsf_Deposit_Acct',
        'transaction_type_Installment_Payment',
        'transaction_type_Transfer',
        'transaction_type_Direct_Deposit',
        'transaction_type_Cheque',
        'transaction_type_Deposit_Cheque',
        'transaction_type_Incoming_Wire',
        'transaction_type_Auto_LOC_Repayment',
        'transaction_type_Outgoing_Wire',
        'transaction_type_EFT_Settlement',
        'transaction_type_Direct_Debit',
        'transaction_type_Overdraft_Transfer',
        'transaction_type_Customer_Transfer',
        'transaction_type_LOC_Disburse_RealTime_Adv',
        'transaction_type_Loan_Transfer',
        'transaction_type_EOD_ODP_Trf_Funded_Acc',
        'transaction_type_misc',
                                  ]

COLUMNS_TO_TAKE_DIFF = ['BRR',
        'BEACON',
        'dunning_level_code',
        'days_in_arrears',
        'Oustanding_principle_on_posting_date',
        'percentage_rate',
        'transactions',
        'abs_transactions',
        'transactions_db',
        'transactions_cr',
        'n_transactions',
        'SUB_SYSTEM_DP',
        'SUB_SYSTEM_FD',
        'SUB_SYSTEM_IN',
        'SUB_SYSTEM_LN',
        'SUB_SYSTEM_RF',
        'SUB_SYSTEM_RP',
        'SUB_SYSTEM_SP',
        'SUB_SYSTEM_TF',
        'transaction_type_Loan_Disbursement',
        'transaction_type_Payment_Distrib_Loan',
        'transaction_type_Loan_Payment',
        'transaction_type_Bank_Trsf_Deposit_Acct',
        'transaction_type_Installment_Payment',
        'transaction_type_Transfer',
        'transaction_type_Direct_Deposit',
        'transaction_type_Cheque',
        'transaction_type_Deposit_Cheque',
        'transaction_type_Incoming_Wire',
        'transaction_type_Auto_LOC_Repayment',
        'transaction_type_Outgoing_Wire',
        'transaction_type_EFT_Settlement',
        'transaction_type_Direct_Debit',
        'transaction_type_Overdraft_Transfer',
        'transaction_type_Customer_Transfer',
        'transaction_type_LOC_Disburse_RealTime_Adv',
        'transaction_type_Loan_Transfer',
        'transaction_type_EOD_ODP_Trf_Funded_Acc',
        'transaction_type_misc',
                        ]

COLUMNS_TO_TAKE_MEAN = COLUMNS_TO_TAKE_PERCENT_CHANGE
COLUMNS_TO_TAKE_MAX = ['BRR', 'dunning_level_code', 'Oustanding_principle_on_posting_date', 'percentage_rate']

# Last OBS Features to use in the model
LAST_OBS_COLUMN_FEATURES = ['mth_since_brr_update'
                            ]

# Data Clean
NON_FEATURE_COLUMNS = ['prediction_date',
                       'evaluation_date',
                       'label',
                       'bus_ptnr_group',
                       'naics_id',
                       'impaired',
                       'defaults_3_months',
                         'defaults_6_months',
                         'defaults_9_months',
                         'defaults_12_months',
                         'has_loan'
                       ]

NAICS_MAPPING = {0: 'missing',
                 11: 'agriculture_forestry_fishing_and_hunting',
                 21: 'mining',
                 22: 'utilities',
                 23: 'construction',
                 31: 'manufacturing',
                 32: 'manufacturing',
                 33: 'manufacturing',
                 42: 'wholesale_trade',
                 44: 'retail_trade',
                 45: 'retail_trade',
                 48: 'transportation_and_warehousing',
                 49: 'transportation_and_warehousing',
                 51: 'information',
                 52: 'finance_and_insurance',
                 53: 'real_estate_rental_and_leasing',
                 54: 'professional,_scientific,_and_technical_services',
                 55: 'management_of_companies_and_enterprises',
                 56: 'administrative_and_support_and_waste_management_and_remediation_services',
                 61: 'educational_services',
                 62: 'health_care_and_social_assistance',
                 71: 'arts_entertainment,_and_recreation',
                 72: 'accommodation_and_food_services',
                 81: 'other_services_(except_public_administration)',
                 92: 'public_administration'}

# Labelling
BASE_LABEL_COLUMN = 'impaired'
LABEL_COLUMN = 'label'

# Experiment
EXPERIMENT_NAME = 'train_with_pipelines'

INITIAL_TRAIN_PERIOD = ('2013-03-31', '2015-03-01')

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


# XGBOOST
XGBOOST_PARAMS = {
    "n_estimators": [50, 100, 150, 200, 300],
    "max_depth": [2, 7],
    'tree_method': ['gpu_hist'],
    "max_leaves": range(3, 126),
    "subsample": [0.3, 0.5, 0.7, 0.9],
    "gamma": [0, 0.5, 0.9],
    "colsample_bytree": [0.3, 0.5, 0.7, 0.9],
    'lambda': range(1, 5, 10),
    'alpha': range(1, 5, 10)
}

XGBOOST = {
    'ESTIMATOR': SampleWeightedXGBoost(),
    'PARAMS': XGBOOST_PARAMS
}

MODELS = [('sample_weighted_xgboost', XGBOOST)]
# MODELS = [('decision_tree', DECISION_TREE)]
