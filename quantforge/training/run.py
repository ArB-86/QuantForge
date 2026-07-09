from quantforge.config.loader import load
from quantforge.walkforward.engine_engine import WalkForwardEngine

def run(config_path):

    config = load(config_path)

    engine = WalkForwardEngine(config)

    return engine
