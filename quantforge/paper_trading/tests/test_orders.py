from quantforge.paper_trading.engine import PaperTradingEngine


def main():
    engine = PaperTradingEngine(capital=100000)

    engine.buy("RELIANCE.NS", 10, 1000)
    engine.buy("TCS.NS", 5, 4000)

    print("Cash:", engine.cash())

    print("Value:",
          engine.portfolio_value({
              "RELIANCE.NS": 1050,
              "TCS.NS": 4200,
          }))

    engine.sell("RELIANCE.NS", 5, 1100)

    print("Cash After Sell:", engine.cash())

    print("Positions:")
    for t, p in engine.positions().items():
        print(t, p)


if __name__ == "__main__":
    main()
