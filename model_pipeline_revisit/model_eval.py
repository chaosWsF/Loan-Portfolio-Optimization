from pathlib import Path
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, plot_roc_curve, plot_confusion_matrix, \
    roc_auc_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class ModelEvaluator:
    def __init__(self, save_path):
        self.save_path = save_path

    def eval_metrics(self, actual, pred, prob, phase):
        metric_dict = {
            f'{phase}_f1_score': f1_score(actual, pred, average='binary'),
            f'{phase}_precision_score': precision_score(actual, pred, average='binary'),
            f'{phase}_recall_score': recall_score(actual, pred, average='binary'),
            f'{phase}_auc_score': roc_auc_score(actual, prob[:, 1])
        }

        df = pd.DataFrame(metric_dict, index=[0])
        df.to_csv(self.save_path / Path(f'model_{phase}_ml_metrics_summary.csv'))

        return metric_dict

    def cv_score_plot(self, nested_scores, non_nested_scores, num_iterations):
        score_difference = non_nested_scores - nested_scores

        fig = plt.figure()
        plt.subplot(211)
        non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
        nested_line, = plt.plot(nested_scores, color='b')
        plt.ylabel("score", fontsize="14")
        plt.legend([non_nested_scores_line, nested_line],
                   ["Non-Nested CV", "Nested CV"],
                   bbox_to_anchor=(0, .4, .5, 0))
        plt.title("Non-Nested and Nested Cross Validation",
                  x=.5, y=1.1, fontsize="15")

        # Plot bar chart of the difference.
        plt.subplot(212)
        difference_plot = plt.bar(range(num_iterations), score_difference)
        plt.xlabel("Individual Trial #")
        plt.legend([difference_plot],
                   ["Non-Nested CV - Nested CV Score"],
                   bbox_to_anchor=(0, 1, .8, 0))
        fig.savefig(self.save_path / Path('cv_scores.png'))

    def confusion_matrix(self, classifier, x_test, y_test, phase='test'):
        normalization_options = [('Normalized', 'true'),
                                 ('Not Normalized', None)
                                 ]
        for title, normalize in normalization_options:
            fig, ax = plt.subplots(figsize=(10, 10))
            disp = plot_confusion_matrix(classifier, x_test, y_test, normalize=normalize)
            plt.title(f'{title} Confusion Matrix ({phase})')
            plt.savefig(self.save_path / Path(f'{title.lower()}_{phase}_confusion_matrix.png'))

    def balance_plot(self, train_data_labels, test_data_labels):
        train_label = pd.DataFrame(train_data_labels, columns=['label'])
        train_label['phase'] = 'train'
        test_label = pd.DataFrame(test_data_labels, columns=['label'])
        test_label['phase'] = 'test'
        data = pd.concat([train_label, test_label])
        g = sns.catplot(x='label', col='phase', data=data, kind='count')
        g.savefig(self.save_path / Path('total_balance_plot.png'))

    def roc_plot(self, estimator, x_test, y_test):

        fig, ax = plt.subplots()
        plot_roc_curve(estimator, x_test, y_test)
        plt.title('ROC Curve')
        plt.savefig(self.save_path / Path('ROC_curve.png'), bbox_inches='tight')

    def feature_importance_plot(self, estimator):
        feature_names = list(estimator['preprocessor'].transformers_[0][1][1].get_feature_names())
        feature_names.extend(estimator['preprocessor'].transformers_[1][2])
        importances = estimator['classifier'].feature_importances_
        indices = np.argsort(importances)
        indices = indices[-15:]

        fig, ax = plt.subplots()
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importances')
        plt.tight_layout()
        plt.savefig(self.save_path / Path('feature_importances.png'), bbox_inches='tight')

    def classification_report_table(self, y_true, y_pred, phase):
        clf_report = classification_report(y_true, y_pred, output_dict=True)
        pd.DataFrame(clf_report).to_csv(self.save_path / Path(f'{phase}_classification_report.csv'))
