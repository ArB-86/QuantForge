from quantforge.paper_trading.engine import PaperTradingEngine


def main():
    engine = PaperTradingEngine(capital=1_000_000)

    engine.buy("RELIANCE.NS", 100, 1500)
    engine.buy("TCS.NS", 50, 3500)
    engine.buy("INFY.NS", 200, 1200)

    prices = {
        "RELIANCE.NS": 1550,
        "TCS.NS": 3650,
        "INFY.NS": 1180,
    }

    print("=" * 60)
    print("Cash:", engine.cash())
    print("Portfolio Value:", engine.portfolio_value(prices))
    print("=" * 60)

    for ticker, pos in engine.positions().items():
        print(
            ticker,
            pos.quantity,
            pos.avg_price,
            prices[ticker],
        )

    engine.sell("INFY.NS", 50, 1180)

    print("\nAfter Sell")
    print("Cash:", engine.cash())
    print("Portfolio Value:", engine.portfolio_value(prices))


if __name__ == "__main__":
    main()
