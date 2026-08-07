import pytest
from quantforge.paper_trading.portfolio import Portfolio

def test_buy_reduces_cash_and_creates_position():
    portfolio = Portfolio(initial_cash=100_000)
    portfolio.buy("RELIANCE.NS", 10, 2500)
    assert portfolio.cash == 75_000
    assert "RELIANCE.NS" in portfolio.positions
    position = portfolio.positions["RELIANCE.NS"]
    assert position.quantity == 10
    assert position.avg_price == 2500

def test_sell_reduces_position_and_increases_cash():
    portfolio = Portfolio(initial_cash=100_000)
    portfolio.buy("RELIANCE.NS", 10, 2500)
    portfolio.sell("RELIANCE.NS", 4, 2600)
    position = portfolio.positions["RELIANCE.NS"]
    assert position.quantity == 6
    assert position.avg_price == 2500
    assert portfolio.cash == 85400

def test_multiple_buys_update_weighted_average_price():
    portfolio = Portfolio(initial_cash=100_000)
    portfolio.buy("RELIANCE.NS", 10, 2000)
    portfolio.buy("RELIANCE.NS", 20, 2500)
    position = portfolio.positions["RELIANCE.NS"]
    assert position.quantity == 30
    assert position.avg_price == pytest.approx(2333.3333333333335)

def test_buy_with_insufficient_cash_raises_error():
    portfolio = Portfolio(initial_cash=10_000)
    with pytest.raises(ValueError, match="Insufficient cash"):
        portfolio.buy("RELIANCE.NS", 10, 2000)
    assert portfolio.cash == 10_000
    assert portfolio.positions == {}

def test_sell_more_than_owned_raises_error():
    portfolio = Portfolio(initial_cash=100_000)
    portfolio.buy("RELIANCE.NS", 10, 2000)
    with pytest.raises(ValueError, match="Insufficient quantity"):
        portfolio.sell("RELIANCE.NS", 11, 2100)
    position = portfolio.positions["RELIANCE.NS"]
    assert position.quantity == 10
    assert position.avg_price == 2000
    assert portfolio.cash == 80_000
