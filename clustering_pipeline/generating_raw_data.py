import pandas as pd
import numpy as np
import dask.dataframe as dd

from pathlib import Path
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import config as config


data_path = config.PATH_WORKING_DIR / 'datasetv10*.csv'
dtypes = {
    'AGRI_gross_farming_inc': 'float64',
    'AGRI_net_farm_inc': 'float64',
    'ALL_cp_ltd': 'float64',
    'CORP_net_cash_income': 'float64',
    'CORP_net_income_growth': 'float64',
    'dunning_level': 'object',
    'syndicated_loan': 'object',
    'BEACON': 'object',
}
ddf = dd.read_csv(data_path, dtype=dtypes)
data = ddf.compute()    # compile to pd.DataFrame

start_date = '2013-01-31'
end_date = '2021-08-31'

data['BEACON'] = pd.to_numeric(data['BEACON'], errors='coerce')    # 'X' in the column
data['cal_day'] = pd.to_datetime(data['cal_day'], errors='coerce')
data = data[data['cal_day'].notnull()].copy()
data = data[(data['cal_day'] >= start_date) & (data['cal_day'] <= end_date)].copy()
loans_data = data[data['has_loan'] == 1]
loans_data = loans_data.sort_values('cal_day')
loans_data = loans_data.reset_index(drop=True)

config.PATH_WORKING_DIR.mkdir(exist_ok=True)
loans_data.to_parquet(config.PATH_RAW_DATA)
