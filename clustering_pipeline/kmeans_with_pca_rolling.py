import pandas as pd
import logging
from feature_extractor import FeatureExtractor

import config as config


class cluster:
    def __init__(self):
        return 
    

    def fit(self, X, y):
        


regen_features = True

if regen_features:
    base_data = pd.read_parquet(config.PATH_RAW_DATA)
    feature_extractor = FeatureExtractor(base_data)
    data = feature_extractor.get_features()
    data.to_pickle(config.PATH_PROCESSED_DATA)
else:
    data = pd.read_pickle(config.PATH_PROCESSED_DATA)
    data['cal_day'] = pd.to_datetime(data['cal_day'])


feat_cols = [col for col in data.columns if col not in (config.NON_FEATURE_COLUMNS + config.ID_COLUMNS)]

pca = PCA(n_components=40, svd_solver='randomized', iterated_power='auto')
reduced_train_data = pca.fit_transform(train[used_cols].to_numpy())
reduced_test_data = pca.fit_transform(test[used_cols].to_numpy())
reduced_datasets.append((reduced_train_data, reduced_test_data))

models = []
for train, _ in reduced_datasets:
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(train)
    models.append(kmeans)
