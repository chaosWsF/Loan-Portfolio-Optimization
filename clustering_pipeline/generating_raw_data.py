import logging
import pandas as pd
import dask.dataframe as dd
import config as config

from pathlib import Path


working_dir = config.PATH_WORKING_DIR
working_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(funcName)s: l%(lineno)d: %(message)s',
    handlers=[
        logging.FileHandler(working_dir / Path('raw_loans_data.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

data_path = working_dir / 'datasetv12_*.csv'
dtypes = {
    'BRR': 'float64',
    'defaults_3_months': 'float64',
    'defaults_6_months': 'float64',
    'defaults_9_months': 'float64',
    'defaults_12_months': 'float64',
    'dunning_level_code': 'float64',
    'impaired': 'float64',
    'naics_id': 'float64',
    'days_in_arrears': 'float64',
    'mth_since_brr_update': 'float64',
    'transaction_type_Customer_Transfer': 'float64',
    'transaction_type_Incoming_Wire': 'float64',
    'transaction_type_Outgoing_Wire': 'float64',
    'BEACON': 'object',    # 'X' in the column
    'dunning_level': 'object',
    'syndicated_loan': 'object'
}
ddf = dd.read_csv(data_path, dtype=dtypes)
data = ddf.compute()    # compile to pd.DataFrame
logger.info('read csv files')

data.drop(columns=['BEACON', 'dunning_level'], inplace=True)
logger.info(f'drop columns with 90% missing values')

start_date = '2013-01-31'
end_date = '2021-12-31'
data['cal_day'] = pd.to_datetime(data['cal_day'], errors='coerce')
data = data[data['cal_day'].notnull()]
data = data[(data['cal_day'] >= start_date) & (data['cal_day'] <= end_date)]
logger.info(f'select data from {start_date} to {end_date}')

loans_data = data[data['has_loan'] == 1]
logger.info('get observations having loan')

loans_data = loans_data.sort_values('cal_day').reset_index(drop=True)
loans_data.to_parquet(config.PATH_RAW_DATA)
logger.info('save raw data')
