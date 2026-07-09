import math


def portfolio_objective(metrics):
    """
    Portfolio optimization objective.

    Higher is better.
    """

    sharpe = metrics["Sharpe"]
    cagr = metrics["CAGR"]
    max_dd = abs(metrics["Max Drawdown"])

    turnover = metrics.get(
        "Average Turnover",
        0.0,
    )

    score = (

        2.0 * sharpe

        + 0.5 * cagr

        - 1.0 * max_dd

        - 0.20 * turnover

    )

    if math.isnan(score):

        return -1e9

    return score