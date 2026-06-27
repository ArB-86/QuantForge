import numpy as np


def evaluate(
    portfolio,
    holding_days=5
):

    years = (
        len(portfolio)
        * holding_days
    ) / 252

    final_equity = (
        portfolio["Equity"]
        .iloc[-1]
    )

    cagr = (
        final_equity
        **
        (1 / years)
        - 1
    )

    volatility = (
        portfolio["Return"]
        .std()
        *
        np.sqrt(
            252 / holding_days
        )
    )

    sharpe = (
        portfolio["Return"]
        .mean()
        /
        portfolio["Return"]
        .std()
        *
        np.sqrt(
            252 / holding_days
        )
    )

    rolling_max = (
        portfolio["Equity"]
        .cummax()
    )

    drawdown = (
        portfolio["Equity"]
        /
        rolling_max
        - 1
    )

    max_dd = drawdown.min()

    win_rate = (
        portfolio["Return"] > 0
    ).mean()

    return {

        "Final Equity": final_equity,

        "CAGR": cagr,

        "Sharpe": sharpe,

        "Volatility": volatility,

        "Max Drawdown": max_dd,

        "Win Rate": win_rate

    }
