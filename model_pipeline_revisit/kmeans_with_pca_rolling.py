import pandas as pd
import logging

from feature_extractor import FeatureExtractor

import config as config


regen_features = True

if regen_features:
    base_data = pd.read_parquet(config.PATH_RAW_DATA)
    feature_extractor = FeatureExtractor(base_data)
    data = feature_extractor.get_features()
    data.to_pickle(config.PATH_PROCESSED_DATA)
else:
    data = pd.read_pickle(config.PATH_PROCESSED_DATA)
    data['cal_day'] = pd.to_datetime(data['cal_day'])
