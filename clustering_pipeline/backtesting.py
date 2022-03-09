import logging
import config as config
import numpy as np

from kmeans_pca import generate_labels, mk_env
from sklearn.metrics import brier_score_loss


def helper(cur, mth):
    """
    cur: str, 'YYYY-MM-DD'; mth: int
    """
    next_date = np.datetime64(cur[:-3]) + np.timedelta64(mth, 'M')
    next_date = np.datetime64(str(next_date + 1) + '-01') - 1
    return str(next_date)


def get_test_period(train_period):
    return (helper(train_period[1], 1), helper(train_period[1], config.TEST_PERIOD_LENGTH))


def next_periods(train_period, test_period):
    next_train_period = (train_period[0], test_period[1])
    next_test_period = get_test_period(next_train_period)
    return next_train_period, next_test_period


def model_eval(data, train_period, test_period):
    date_id = config.DATE_COLUMN
    target = config.KMEANS_LABEL_COLUMN
    base_label = config.BASE_LABEL_COLUMN

    train_start, train_end = train_period
    test_start, test_end = test_period

    labels = generate_labels(data, train_period, test_period)
    data = data[data[date_id].between(train_start, test_end, inclusive='both')].copy()
    data[target] = labels

    in_sample = data[data[date_id].between(train_start, train_end, inclusive='both')]
    out_of_sample = data[data[date_id].between(test_start, test_end, inclusive='both')]
    y_pred = in_sample.groupby(target)[base_label].mean()
    y_out = out_of_sample.groupby(target)[base_label].mean()
    y_out[y_out >= 0.5] = 1
    y_out[y_out < 0.5] = 0
    shared_idx = y_pred.index.intersection(y_out.index)
    y_pred = y_pred[shared_idx].to_numpy()
    y_out = y_out[shared_idx].to_numpy()

    return brier_score_loss(y_out, y_pred)


if __name__ == '__main__':
    train_period = config.INITIAL_TRAIN_PERIOD
    test_period = get_test_period(train_period)
    start_date = train_period[0]
    end_date = config.TEST_PERIOD[1]

    loans_data = mk_env(start_date, end_date, 'backtesting.log')
    
    logger = logging.getLogger(__name__)
    
    scores = []
    
    while not any([np.datetime64(d) > np.datetime64(end_date) for d in (train_period + test_period)]):
        score = model_eval(loans_data, train_period, test_period)

        scores.append(score)

        logger.info(
            f'train on {train_period}; test on {test_period}; score = {score}'
        )

        train_period, test_period = next_periods(train_period, test_period)

    average_score = np.mean(scores)
    logger.info(f'Average Brier Score is {average_score}.')
