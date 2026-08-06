from quantforge.live_trading.connectors.factory import create_connector

connector = create_connector("paper")

print(type(connector).__name__)
print(connector.orders())

print("PASS")
