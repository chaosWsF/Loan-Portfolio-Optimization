import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler, OrdinalEncoder
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score, davies_bouldin_score

import config as config


def generate_datasets(data, col, start_date, end_date, train_size, test_size):
    selected_dates = data[col].between(start_date, end_date, inclusive='both')
    data = data[selected_dates]
    
    datasets = []
    training_split_dates = pd.date_range(start_date, end_date, freq=f'{train_size}M')[1:]
    testing_split_dates = training_split_dates.shift(test_size, freq='M')
    for i in range(len(training_split_dates)):
        date1 = training_split_dates[i]
        train_data = data[data[col] < date1]
        
        date2 = testing_split_dates[i]
        test_selected_dates = data[col].between(date1, date2, inclusive='left')
        test_data = data[test_selected_dates]

        datasets.append((train_data, test_data))
    
    return datasets


loans_data = pd.read_parquet(config.PATH_RAW_DATA)
# loans_data['cal_day'] = pd.to_datetime(loans_data['cal_day'])


start_date = '2016-01-31'
end_date = '2020-12-31'
training_months = 48
testing_months = 12

datasets = generate_datasets(loans_data, 'cal_day', start_date, end_date, training_months, testing_months)
train, test = datasets[0]
data = train.append(test).reset_index(drop=True)


numeric_cols = data.select_dtypes(include='number').columns.difference(
    config.ID_COLUMNS + config.CATEGORICAL_FEATURES + config.NON_FEATURE_COLUMNS
).to_list()
categorical_cols = ['trend_code', 'dunning_level', 'syndicated_loan']
cols = numeric_cols + categorical_cols

ct = ColumnTransformer([
    ('cat_encoding', OrdinalEncoder(), categorical_cols),
])
train[categorical_cols] = ct.fit_transform(train)
clean_data = data.copy()
clean_data[categorical_cols] = ct.fit_transform(clean_data)

scaler = Pipeline([
    ('median_imputer', SimpleImputer(strategy='median')),
    ('power_scaler', PowerTransformer(method='yeo-johnson', standardize=True)),
])
train[numeric_cols] = scaler.fit_transform(train[numeric_cols].to_numpy())
train.replace(to_replace=[np.inf, -np.inf], value=np.nan, inplace=True)
train.fillna(0.0, inplace=True)
clean_data[numeric_cols] = scaler.fit_transform(clean_data[numeric_cols].to_numpy())
clean_data.replace(to_replace=[np.inf, -np.inf], value=np.nan, inplace=True)
clean_data.fillna(0.0, inplace=True)

n_clusters = 20

ppl = Pipeline([
    ('PCA', PCA(n_components=0.95)),
    ('scaler', StandardScaler()),
    ('KMeans', KMeans(n_clusters=n_clusters)),
])

ppl.fit(train[cols].to_numpy())
labels = ppl.predict(clean_data[cols].to_numpy())
data['kmeans_label'] = labels
data.to_parquet(config.PATH_WORKING_DIR / f'cluster_labels_{n_clusters}.pq')
