from quantforge.live_trading.router import OrderRouter
from quantforge.live_trading.order import (
    LiveOrder,
    OrderSide,
    OrderType,
)

router = OrderRouter("paper")

router.submit(
    LiveOrder(
        ticker="RELIANCE.NS",
        side=OrderSide.BUY,
        quantity=5,
        order_type=OrderType.MARKET,
    )
)

print(router.orders())
print("PASS")
