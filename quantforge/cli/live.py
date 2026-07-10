import argparse

from quantforge.live_trading.engine import LiveTradingEngine
from quantforge.live_trading.factory import create_broker
from quantforge.live_trading.order import (
    LiveOrder,
    OrderSide,
    OrderType,
)


def main():
    parser = argparse.ArgumentParser("QuantForge Live Trading")

    parser.add_argument(
        "--broker",
        default="paper",
        choices=["paper", "kite"],
    )

    args = parser.parse_args()

    broker = create_broker(args.broker)
    engine = LiveTradingEngine(broker)

    print("=" * 70)
    print("Broker :", args.broker)
    print("Funds  :", engine.funds())
    print("=" * 70)

    if args.broker == "paper":
        result = engine.submit(
            LiveOrder(
                ticker="RELIANCE.NS",
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.MARKET,
            )
        )

        print("Order :", result)
        print("Orders:", engine.orders())


if __name__ == "__main__":
    main()
