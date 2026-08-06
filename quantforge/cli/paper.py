import argparse

from quantforge.paper_trading.engine import PaperTradingEngine
from quantforge.paper_trading.types import OrderSide


def main():
    parser = argparse.ArgumentParser("QuantForge Paper Trading")

    parser.add_argument(
        "--capital",
        type=float,
        default=1_000_000,
    )

    args = parser.parse_args()

    engine = PaperTradingEngine(args.capital)

    engine.submit(
        ticker="RELIANCE.NS",
        side=OrderSide.BUY,
        quantity=100,
        price=1500,
    )

    engine.mark({
        "RELIANCE.NS": 1525,
    })

    engine.snapshot("DEMO")

    print("=" * 70)
    print("Cash      :", engine.cash())
    print("Equity    :", engine.portfolio_value(engine.market.snapshot()))
    print("Positions :", engine.positions())
    print("=" * 70)

    engine.save_report()


if __name__ == "__main__":
    main()
