import numpy as np
from scipy.optimize import minimize


class EqualRiskContributionOptimizer:

    def __init__(
        self,
        max_weight=0.20,
        min_weight=0.02,
    ):
        self.max_weight = max_weight
        self.min_weight = min_weight

    def _risk_contribution(
        self,
        weights,
        covariance,
    ):

        portfolio_var = (
            weights.T
            @ covariance
            @ weights
        )

        portfolio_vol = np.sqrt(
            max(portfolio_var, 1e-12)
        )

        marginal = (
            covariance @ weights
        ) / portfolio_vol

        contribution = (
            weights * marginal
        )

        return contribution

    def _objective(
        self,
        weights,
        covariance,
    ):

        rc = self._risk_contribution(
            weights,
            covariance,
        )

        target = rc.mean()

        return np.sum(
            (rc - target) ** 2
        )

    def optimize(
        self,
        covariance,
    ):

        covariance = np.asarray(
            covariance,
            dtype=float,
        )

        n = covariance.shape[0]

        x0 = np.repeat(
            1.0 / n,
            n,
        )

        bounds = [
            (
                self.min_weight,
                self.max_weight,
            )
            for _ in range(n)
        ]

        constraints = [
            {
                "type": "eq",
                "fun": lambda w: w.sum() - 1.0,
            }
        ]

        result = minimize(
            self._objective,
            x0,
            args=(covariance,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "ftol": 1e-10,
                "maxiter": 500,
                "disp": False,
            },
        )

        if not result.success:

            return x0

        weights = result.x

        weights = np.clip(
            weights,
            self.min_weight,
            self.max_weight,
        )

        weights /= weights.sum()

        return weights


# Backward compatibility
MinimumVarianceOptimizer = EqualRiskContributionOptimizer
