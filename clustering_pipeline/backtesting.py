import logging
import pandas as pd
import numpy as np


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
