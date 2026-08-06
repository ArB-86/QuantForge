from quantforge.core.config.loader import load
from quantforge.walkforward.engine import WalkForwardEngine

def run(config_path):

    config = load(config_path)

    engine = WalkForwardEngine(config)

    return engine
