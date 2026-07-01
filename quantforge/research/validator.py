class ExperimentValidator:
    """
    Reject experiments that do not meet the
    minimum quality requirements.
    """

    def __init__(self, config=None):

        config = config or {}

        self.min_sharpe = config.get(
            "min_sharpe",
            1.00,
        )

        self.max_drawdown = config.get(
            "max_drawdown",
            -0.40,
        )

        self.min_cagr = config.get(
            "min_cagr",
            0.00,
        )

        self.min_win_rate = config.get(
            "min_win_rate",
            0.45,
        )

    def validate(self, metrics):

        reasons = []

        if metrics["Sharpe"] < self.min_sharpe:
            reasons.append(
                f"Sharpe < {self.min_sharpe}"
            )

        if metrics["Max Drawdown"] < self.max_drawdown:
            reasons.append(
                f"Drawdown < {self.max_drawdown}"
            )

        if metrics["CAGR"] < self.min_cagr:
            reasons.append(
                f"CAGR < {self.min_cagr}"
            )

        if metrics["Win Rate"] < self.min_win_rate:
            reasons.append(
                f"WinRate < {self.min_win_rate}"
            )

        return (

            len(reasons) == 0,

            reasons,

        )
