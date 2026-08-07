from quantforge.paper_trading.account import PaperTradingAccount
from quantforge.paper_trading.broker import PaperBroker
from quantforge.paper_trading.order import Order
from quantforge.paper_trading.types import OrderSide, OrderStatus

def test_buy_order_is_filled_and_recorded():
    account = PaperTradingAccount(capital=100_000)
    broker = PaperBroker(account)
    order = Order(
        ticker="RELIANCE.NS",
        side=OrderSide.BUY,
        quantity=10,
        price=2500,
    )
    result = broker.submit(order)
    assert result.status == OrderStatus.FILLED
    assert len(broker.orders) == 1
    assert broker.orders[0] is result
    position = account.portfolio.positions["RELIANCE.NS"]
    assert position.quantity == 10
    assert account.portfolio.cash == 75_000

def test_sell_order_is_filled_and_updates_portfolio():
    account = PaperTradingAccount(capital=100_000)
    broker = PaperBroker(account)
    broker.submit(
        Order(
            ticker="RELIANCE.NS",
            side=OrderSide.BUY,
            quantity=10,
            price=2500,
        )
    )
    sell = Order(
        ticker="RELIANCE.NS",
        side=OrderSide.SELL,
        quantity=4,
        price=2600,
    )
    result = broker.submit(sell)
    assert result.status == OrderStatus.FILLED
    assert len(broker.orders) == 2
    position = account.portfolio.positions["RELIANCE.NS"]
    assert position.quantity == 6
    assert position.avg_price == 2500
    assert account.portfolio.cash == 85_400
