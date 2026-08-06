from quantforge.paper_trading.engine import PaperTradingEngine
from quantforge.paper_trading.order import Order
from quantforge.paper_trading.types import OrderSide


def main():
    engine = PaperTradingEngine(capital=100000)

    order = Order(
        ticker="RELIANCE.NS",
        side=OrderSide.BUY,
        quantity=10,
        price=1000,
    )

    engine.execution.execute([order])

    equity = engine.account.portfolio.equity(
        {"RELIANCE.NS": 1050}
    )

    print("Cash :", engine.account.portfolio.cash)
    print("Equity :", equity)
    print("Positions :", engine.account.portfolio.positions)


if __name__ == "__main__":
    main()
