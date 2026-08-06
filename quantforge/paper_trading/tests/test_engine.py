from quantforge.paper_trading.engine import PaperTradingEngine


def main():
    engine = PaperTradingEngine(capital=500000)

    engine.buy("RELIANCE.NS", 100, 1500)
    engine.buy("TCS.NS", 20, 4000)
    engine.sell("RELIANCE.NS", 40, 1600)

    prices = {
        "RELIANCE.NS": 1625,
        "TCS.NS": 4150,
    }

    print("=" * 70)
    print("Cash:", engine.cash())
    print("Equity:", engine.portfolio_value(prices))
    print("=" * 70)

    for ticker, pos in engine.positions().items():
        print(
            f"{ticker:15} Qty={pos.quantity:8.2f} Avg={pos.avg_price:10.2f}"
        )


if __name__ == "__main__":
    main()
