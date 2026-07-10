import argparse

from quantforge.paper_trading.engine import PaperTradingEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=1_000_000)

    args = parser.parse_args()

    engine = PaperTradingEngine(capital=args.capital)

    engine.buy("RELIANCE.NS", 100, 1500)
    engine.buy("TCS.NS", 50, 3500)

    engine.mark({
        "RELIANCE.NS": 1540,
        "TCS.NS": 3620,
    })

    engine.snapshot("DAY1")
    engine.save_report()

    print("Cash:", engine.cash())
    print("Equity:", engine.portfolio_value(engine.market.snapshot()))
    print("Report generated.")


if __name__ == "__main__":
    main()
