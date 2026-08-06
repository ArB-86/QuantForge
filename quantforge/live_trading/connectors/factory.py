from quantforge.live_trading.connectors.paper import PaperConnector
from quantforge.live_trading.connectors.kite import KiteConnector
from quantforge.live_trading.config import KiteConfig


def create_connector(name="paper"):
    name = name.lower()

    if name == "paper":
        c = PaperConnector()
        c.connect()
        return c

    if name == "kite":
        cfg = KiteConfig.from_env()
        c = KiteConnector(
            cfg.api_key,
            cfg.access_token,
        )
        c.connect()
        return c

    raise ValueError(f"Unknown connector: {name}")
