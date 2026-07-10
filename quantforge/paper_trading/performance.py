class PerformanceTracker:
    def __init__(self):
        self.history = []

    def record(self, timestamp, equity, cash):
        self.history.append(
            {
                "timestamp": timestamp,
                "equity": equity,
                "cash": cash,
            }
        )

    def latest(self):
        if not self.history:
            return None
        return self.history[-1]

    def returns(self):
        if len(self.history) < 2:
            return []

        r = []
        for i in range(1, len(self.history)):
            prev = self.history[i-1]["equity"]
            cur = self.history[i]["equity"]
            r.append((cur-prev)/prev)
        return r
