import logging
import pandas as pd
import numpy as np

from sklearn.metrics import brier_score_loss


def generate_datasets(data, col, start_date, end_date, train_size, test_size):
    # TODO: add initial train period
    # TODO: add loggings
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


def mth_diff(date1: str, date2: str) -> int:
    """
    date1, date2: YYYY-MM-DD
    """
    date1 = np.datetime64(date1[:-3])
    date2 = np.datetime64(date2[:-3])
    return ((date2 - date1).astype(int) + 1)
