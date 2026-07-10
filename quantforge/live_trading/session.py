from datetime import datetime, time


class TradingSession:

    def __init__(
        self,
        start=time(9, 15),
        end=time(15, 30),
    ):
        self.start = start
        self.end = end

    def is_open(self, now=None):
        now = now or datetime.now().time()
        return self.start <= now <= self.end

    def ensure_open(self):
        if not self.is_open():
            raise RuntimeError("Market is closed")
