import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector, ColumnTransformer


def build_pipeline(model):

    numerical_preprocessing = Pipeline(steps=[('num_imputation', SimpleImputer(strategy='median'))])
    categorical_preprocessing = Pipeline(
        steps=[('cat_imputation', SimpleImputer(strategy='constant', fill_value='missing')),
               ('encoder', OneHotEncoder(drop=None, handle_unknown='ignore'))
               ])

    preprocessing_pipeline = ColumnTransformer([
        ('categorical_preprocessor',
         categorical_preprocessing,
         make_column_selector(dtype_include='category')),
        ('numerical_preprocessor',
         numerical_preprocessing,
         make_column_selector(dtype_include='number'))
    ])

    pipeline_estimator = Pipeline([
        ('preprocessor', preprocessing_pipeline),
        ('classifier', model['ESTIMATOR'])
    ])

    new_param_distribution = {}
    for parameter in model['PARAMS']:
        new_param_distribution[f'classifier__{parameter}'] = model['PARAMS'][parameter]

    model_with_pipeline = {'ESTIMATOR': pipeline_estimator,
                           'PARAMS': new_param_distribution
                           }

    return model_with_pipeline
