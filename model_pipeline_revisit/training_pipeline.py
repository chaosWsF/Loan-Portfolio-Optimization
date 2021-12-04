import logging
import json
import pandas as pd
import numpy as np
import sklearn.model_selection
import joblib

import model_pipeline_revisit.config as config
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, TimeSeriesSplit
from typing import Dict, Any, List
from model_eval import ModelEvaluator


class TrainingPipeline:
    """
    Training pipeline class for default prediction.
    """

    def __init__(self,
                 model: Dict[str, Any],
                 feature_columns: List[str],
                 cv_scoring_metric: str = config.CV_SCORE,
                 label_column_name: str = config.LABEL_COLUMN,
                 num_iterations: int = config.NUM_ITER):
        """
        :param feature_columns: columns to be used as features
        :param cv_scoring_metric:
        :param label_column_name:
        """

        self.logger = logging.getLogger(__name__)

        self.cv_scoring = cv_scoring_metric
        self.label_column = label_column_name
        self.feature_columns = feature_columns
        self.num_iterations = num_iterations
        self.estimator = model['ESTIMATOR']
        self.parameter_space = model['PARAMS']

    def process_dataset(self, data):
        x = data[self.feature_columns]
        y = data[self.label_column].ravel()

        return x, y

    def hyperparameter_tuning(self,
                              training_data: pd.DataFrame,
                              n_splits: int = config.NUM_OUTER_SPLITS,
                              cv_scoring: str = config.CV_SCORE):
        """
        Run tuning for training and test
        :return:
        """

        self.logger.info('Model tuning: Started')

        outer_cv = TimeSeriesSplit(n_splits=n_splits)
        clf = RandomizedSearchCV(self.estimator,
                                 self.parameter_space,
                                 n_iter=self.num_iterations,
                                 cv=outer_cv,
                                 scoring=cv_scoring,
                                 verbose=2)

        train_x = training_data[self.feature_columns]
        train_y = training_data[self.label_column].ravel()

        clf.fit(train_x, train_y)
        for param, param_value in clf.best_params_.items():
            self.logger.info(f'{param} - {param_value}')

        self.estimator = clf.best_estimator_
        self.logger.info('Model tuning: Finished')

    def evaluate_trained_model(self, train: pd.DataFrame, test: pd.DataFrame, model_path: Path):

        model_evaluator = ModelEvaluator(save_path=model_path)

        self.logger.info('Begin logging metrics')

        train_x, train_y = self.process_dataset(train)
        test_x, test_y = self.process_dataset(test)

        model_evaluator.confusion_matrix(self.estimator, train_x, train_y, phase='train')
        model_evaluator.confusion_matrix(self.estimator, test_x, test_y, phase='test')
        model_evaluator.balance_plot(train_y, test_y)
        model_evaluator.feature_importance_plot(self.estimator)

        train_predictions = self.estimator.predict(train_x)
        train_probabilities = self.estimator.predict_proba(train_x)

        test_predictions = self.estimator.predict(test_x)
        test_probabilities = self.estimator.predict_proba(test_x)

        model_evaluator.roc_plot(self.estimator, test_x, test_y)

        model_evaluator.eval_metrics(train_y, train_predictions, train_probabilities, 'train')
        model_evaluator.eval_metrics(test_y, test_predictions, test_probabilities, 'test')
        model_evaluator.classification_report_table(train_y, train_predictions, 'train')
        model_evaluator.classification_report_table(test_y, test_predictions, 'test')

        self.logger.info('Finished logging metrics')

    def run_nested_cv_experiment(self,
                                 training_data,
                                 model_path: Path,
                                 num_iterations: int = config.NUM_ITER,
                                 num_inner_splits: int = config.NUM_INNER_SPLITS,
                                 num_outer_splits: int = config.NUM_OUTER_SPLITS
                                 ):
        """
        Run nested cv for a better estimate of model performance and save the plot comparing nested and non_nested_scores
        """

        self.logger.info('Nested CV Experiment Begin')

        nested_scores = np.zeros(num_iterations)
        non_nested_scores = np.zeros(num_iterations)
        self.logger.info(f'Experiment Begin with {num_iterations} Iterations')

        for i in range(num_iterations):
            self.logger.info(f'Iteration {i}/{num_iterations} Started')
            inner_cv = TimeSeriesSplit(n_splits=num_inner_splits)
            outer_cv = TimeSeriesSplit(n_splits=num_outer_splits)
            
            """
            clf = RandomizedSearchCV(self.estimator, self.parameter_space,
                                     cv=inner_cv, scoring=self.cv_scoring, verbose=2)
            train_x = training_data[self.feature_columns]
            train_y = training_data[self.label_column].ravel()
            clf.fit(train_x, train_y)
            non_nested_scores[i] = clf.best_score_
            """
            
            clf = self.estimator
            
            
            nested_score = cross_val_score(clf, X=train_x, y=train_y, cv=outer_cv)
            nested_scores[i] = nested_score.mean()
            self.logger.info(f'Iteration {i + 1}/{num_iterations} Finished')

        ModelEvaluator(model_path).cv_score_plot(nested_scores, non_nested_scores, num_iterations)

        self.logger.info('Nested CV Experiment: Finished')

    def generate_historical_predictions(self, training_data, model_path):
        """
        Train model for an initial train period, then predict next quarter, calculate expected losses and store them.
        Then increase training period by one quarter and retrain
        """

        initial_train_data = training_data.loc[training_data['prediction_date'] < config.INITIAL_TRAIN_PERIOD[1]]

        self.hyperparameter_tuning(initial_train_data)

        dates = training_data['prediction_date'].unique()

        historical_predictions = []

        for date in dates:
            self.logger.info(f'Prediction Process for {date}')
            data_for_training = training_data.loc[training_data['evaluation_date'] <= date]
            self.estimator.fit(data_for_training[self.feature_columns], data_for_training[self.label_column].ravel())

            data_for_prediction = training_data.loc[training_data['prediction_date'] == date]
            predictions = self.estimator.predict_proba(data_for_prediction[self.feature_columns])

            predictions = pd.DataFrame(predictions, columns=['non_default', 'default'])
            predictions['prediction_date'] = data_for_prediction['prediction_date']
            predictions['evaluation_date'] = data_for_training['evaluation_date']
            predictions['bus_ptnr_group'] = data_for_prediction['bus_ptnr_group']

            historical_predictions.append(predictions)

        self.logger.info('Historical predictions generated')
        historical_predictions = pd.concat(historical_predictions)
        historical_predictions.to_csv(model_path / 'historical_predictions.csv')

        self.logger.info('Dollar Performance Numbers Saved')
