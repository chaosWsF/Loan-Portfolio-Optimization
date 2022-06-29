from banditpam import KMedoids
from sklearn.base import BaseEstimator, ClusterMixin


class BanditPAM:
    def __init__(
        self, n_medoids: int = 0, algorithm: str = 'BanditPAM', max_iter: int = 1000, 
        build_confidence: int = 1000, swap_confidence: int = 10000
    ) -> None:
        self.params = {
            'n_medoids': n_medoids,
            'algorithm': algorithm,
            'max_iter': max_iter,
            'build_confidence': build_confidence,
            'swap_confidence': swap_confidence
        }

    def fit(self, X, y=None):
        """
        X: numpy.ndarray[numpy.float32]
        """
        self.cluster = KMedoids(**self.params)
        self.cluster.fit(X, 'L2')

    def predict(self, X):    # get fitted model's predictions
        # self.fit(X)
        labels = self.cluster.labels
        return labels
