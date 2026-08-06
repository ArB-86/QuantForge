from quantforge.backtest.backtest_engine import BacktestEngine
from quantforge.core.config.config import Config


def backtest(config):
    """
    Run portfolio backtest.
    """

    if isinstance(config, str):
        config = Config(config).dict()

    print("=" * 80)
    print("BACKTEST")
    print("=" * 80)

    engine = BacktestEngine(config)

    return engine.run()
