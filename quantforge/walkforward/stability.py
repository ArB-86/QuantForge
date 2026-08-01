import numpy as np

class StabilityAnalyzer:
    @staticmethod
    def score(values):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            return None
        mean = np.mean(values)
        std  = np.std(values)
        if mean == 0:
            return None
        return float(1 - std / abs(mean))
