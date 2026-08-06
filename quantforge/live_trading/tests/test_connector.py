from quantforge.live_trading.connectors.paper import PaperConnector

c = PaperConnector()

assert c.connect()

print("Orders:", c.orders())
print("Positions:", c.positions())
print("Holdings:", c.holdings())

print("PASS")
