import pandas as pd
import numpy as np
import dask.dataframe as dd

from pathlib import Path
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import model_pipeline_revisit.config as config


data_path = Path('/home/c44406/datasets/atb_uofa/loan_portfolio/') / 'datasetv7*.csv'
dtype = {
    'dunning_level': 'object',
    'SUB_SYSTEM_FD': 'float64',
    'SUB_SYSTEM_IN': 'float64',
    'SUB_SYSTEM_RF': 'float64',
    'SUB_SYSTEM_RP': 'float64',
    'SUB_SYSTEM_SP': 'float64',
    'SUB_SYSTEM_TF': 'float64',
    'n_transactions': 'float64',
    'transaction_type_Bank_Trsf_Deposit_Acct': 'float64',
    'transaction_type_Customer_Transfer': 'float64',
    'transaction_type_Incoming_Wire': 'float64',
    'transaction_type_Loan_Disbursement': 'float64',
    'transaction_type_Outgoing_Wire': 'float64',
}
ddf = dd.read_csv(data_path, dtype=dtype)
data = ddf.compute()    # compile to pd.DataFrame


start_date = '2013-01-31'
end_date = '2021-08-31'

data['cal_day'] = pd.to_datetime(data['cal_day'], errors='coerce')
data = data[data['cal_day'].notnull()].copy()
data = data[(data['cal_day'] >= start_date) & (data['cal_day'] <= end_date)].copy()
data = data.sort_values('cal_day')
data = data.reset_index(drop=True)


num_cols = data.columns.difference(
    config.ID_COLUMNS + config.CATEGORICAL_FEATURES + config.NON_FEATURE_COLUMNS
).to_list()

num_transformer = Pipeline(steps=[
    ('imputing', SimpleImputer(strategy='median')),
    ('scaling', PowerTransformer(method='yeo-johnson', standardize=True)),
    ('masking', SimpleImputer(strategy='constant', fill_value=0.)),
])
preprocessor = ColumnTransformer(transformers=[
    ('num_col', num_transformer, used_cols),
])

data[num_cols] = preprocessor.fit_transform(data)


data.to_parquet(config.PATH_RAW_DATA)
