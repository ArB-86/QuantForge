from quantforge.live_trading.orderbook import OrderBook

book = OrderBook()

book.update(
    bids=[
        (100.0,100),
        (99.9,200),
    ],
    asks=[
        (100.1,150),
        (100.2,250),
    ],
)

assert book.best_bid.price == 100.0
assert book.best_ask.price == 100.1
assert abs(book.spread - 0.1) < 1e-9

print(book.best_bid)
print(book.best_ask)
print(book.spread)

print("PASS")
