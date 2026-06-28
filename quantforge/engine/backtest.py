from quantforge.backtest.backtest_engine import BacktestEngine


def backtest(config):
    """
    Run portfolio backtest.

    Parameters
    ----------
    config : dict

    Returns
    -------
    (portfolio, metrics)
    """

    print("=" * 80)
    print("BACKTEST")
    print("=" * 80)

    engine = BacktestEngine(config)

    return engine.run()