# from model_pipeline_revisit.training_pipeline import TrainingPipeline
import pandas as pd
import logging

from datetime import datetime
from pathlib import Path

from model_pipeline_revisit.training_pipeline import TrainingPipeline
from model_pipeline_revisit.data_pipeline import DataPipeline
from model_pipeline_revisit.feature_extractor import FeatureExtractor
from model_pipeline_revisit.model_pipeline import build_pipeline
import model_pipeline_revisit.config as config


def create_experiment_directory():
    # working dir
    config.PATH_WORKING_DIR.mkdir(exist_ok=True)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    model_output_path = config.PATH_WORKING_DIR / Path(f'{config.EXPERIMENT_NAME}_{current_time}')
    model_output_path.mkdir(exist_ok=True)

    return model_output_path


def run_experiment(experiment_type: str = 'ml_eval', regen_features: bool = True):
    """
    Run loan loss experiment.
    :param experiment_type: Type of experiment either backtest or ml_eval
    :param regen_features: Wether to regen_features or read in old ones
    """
    new_experiment_path = create_experiment_directory()

    logging.root.handlers = []
    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s - %(name)s - %(funcName)s: l%(lineno)d: %(message)s',
        handlers=[logging.FileHandler(new_experiment_path / Path(f'model_pipeline_log.log')),
                  logging.StreamHandler()
                  ]
    )

    # read data
    if regen_features:
        base_data = pd.read_csv(config.PATH_RAW_DATA)
        feature_extractor = FeatureExtractor(base_data)
        data = feature_extractor.get_features()
        print(len([col for col in data.columns if 'naics_name' in col]))
        data.to_pickle(config.PATH_PROCESSED_DATA)
    else:
        data = pd.read_pickle(config.PATH_PROCESSED_DATA)
        data['cal_day'] = pd.to_datetime(data['cal_day'])

    data_pipeline = DataPipeline()
    data = data_pipeline.clean_data(data)
    data_pipeline.log_dataset(data, new_experiment_path)
    feature_columns = [col for col in data.columns if col not in config.NON_FEATURE_COLUMNS]

    if experiment_type == 'ml_eval':
        train, test = data_pipeline.train_test_split(data)

        for model_name, model in config.MODELS:
            model_path = new_experiment_path / Path(model_name)
            model_path.mkdir(exist_ok=True)
            model_pipeline = build_pipeline(model)
            training_pipeline = TrainingPipeline(model_pipeline, feature_columns)
            training_pipeline.hyperparameter_tuning(train, n_splits=config.NUM_OUTER_SPLITS)
            training_pipeline.evaluate_trained_model(train, test, model_path)

    elif experiment_type == 'periodic_predictions':
        for model_name, model in config.MODELS:
            model_path = new_experiment_path / Path(model_name)
            model_path.mkdir(exist_ok=True)
            model_pipeline = build_pipeline(model)
            training_pipeline = TrainingPipeline(model_pipeline, feature_columns)
            training_pipeline.generate_historical_predictions(data, model_path)


if __name__ == '__main__':
    # run_experiment('ml_eval')
    run_experiment('ml_eval', regen_features=False)
