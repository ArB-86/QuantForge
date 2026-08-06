from quantforge.live_trading.pretrade import PreTradeRisk
from quantforge.live_trading.order import LiveOrder, OrderSide, OrderType

risk = PreTradeRisk()

risk.validate(
    LiveOrder(
        ticker="RELIANCE.NS",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
        price=1500,
    )
)

print("PASS")
