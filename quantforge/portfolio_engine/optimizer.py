import numpy as np


class MinimumVarianceOptimizer:

    def __init__(
        self,
        max_weight=0.20,
        min_weight=0.00,
    ):
        self.max_weight = max_weight
        self.min_weight = min_weight

    def optimize(
        self,
        covariance,
    ):

        covariance = np.asarray(covariance)

        n = covariance.shape[0]

        inv = np.linalg.pinv(covariance)

        ones = np.ones(n)

        weights = inv @ ones

        weights = weights / weights.sum()

        weights = np.clip(
            weights,
            self.min_weight,
            self.max_weight,
        )

        weights = weights / weights.sum()

        return weights
