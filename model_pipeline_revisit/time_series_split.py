import logging
import pandas as pd
import numpy as np


def generate_datasets(data, col, start_date, end_date, train_size, test_size):
    data = data[data[col] >= start_date]
    data = data[data[col] <= end_date]
    
    datasets = []
    training_split_dates = pd.date_range(start_date, end_date, freq=f'{train_size}M')[1:]
    testing_split_dates = training_split_dates.shift(test_size, freq='M')
    for i in range(len(training_split_dates)):
        date1 = training_split_dates[i]
        date2 = testing_split_dates[i]
        
        train_data = data[data[col] < date1].copy()
        test_data = data[data[col] >= date1].copy()
        test_data = test_data[test_data[col]< date2]

        datasets.append((train_data, test_data))
    
    return datasets


if __name__ == '__main__':
    start_date = '2016-01-31'
    end_date = '2020-12-31'
    training_months = 19
    testing_months = 3

    datasets = generate_datasets(loan_data, 'cal_day', start_date, end_date, training_months, testing_months)
