from quantforge.live_trading.paper_broker import PaperBroker
from quantforge.live_trading.kite_broker import KiteBroker
from quantforge.live_trading.config import KiteConfig


def create_broker(mode="paper"):
    mode = mode.lower()

    if mode == "paper":
        return PaperBroker()

    if mode == "kite":
        cfg = KiteConfig.from_env()
        broker = KiteBroker(
            cfg.api_key,
            cfg.access_token,
        )
        broker.login()
        return broker

    raise ValueError(f"Unknown broker mode: {mode}")
