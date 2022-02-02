import xgboost as xgb
import catboost
import numpy as np
from sklearn.utils import class_weight


class SampleWeightedXGBoost(xgb.XGBClassifier):
    """
    Wrapper for XGBClassifier that is able to compute sample weights for each train period.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_sample_weights(self, y):
        unique_labels = np.unique(y)
        class_weights = class_weight.compute_class_weight('balanced', classes=unique_labels, y=y)
        label_to_class_weight = {unique_labels[i]: class_weights[i] for i in range(len(unique_labels))}
        sample_weights = [label_to_class_weight[y[i]] for i in range(len(y))]

        return sample_weights

    def fit(self, X, y, use_balanced_sample_weighting=True, **kwargs):
        # calculate sample sample_weights
        if use_balanced_sample_weighting:
            sample_weights = self.get_sample_weights(y)
        else:
            sample_weights = None

        return super().fit(X, y, sample_weight=sample_weights)
    
    
class SampleWeightedCatboost(catboost.CatBoostClassifier):
    """
    Wrapper for CatBoostClassifier that is able to compute sample weights for each train period.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_sample_weights(self, y):
        unique_labels = np.unique(y)
        class_weights = class_weight.compute_class_weight('balanced', classes=unique_labels, y=y)
        label_to_class_weight = {unique_labels[i]: class_weights[i] for i in range(len(unique_labels))}
        sample_weights = [label_to_class_weight[y[i]] for i in range(len(y))]

        return sample_weights

    def fit(self, X, y, use_balanced_sample_weighting=True, **kwargs):
        # calculate sample sample_weights
        if use_balanced_sample_weighting:
            sample_weights = self.get_sample_weights(y)
        else:
            sample_weights = None

        return super().fit(X, y, sample_weight=sample_weights)