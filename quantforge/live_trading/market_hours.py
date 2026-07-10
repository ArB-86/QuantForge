from datetime import datetime, time


class MarketHours:

    OPEN = time(9, 15)
    CLOSE = time(15, 30)

    @classmethod
    def is_open(cls, dt=None):
        dt = dt or datetime.now()

        if dt.weekday() >= 5:
            return False

        t = dt.time()

        return cls.OPEN <= t <= cls.CLOSE

    @classmethod
    def ensure_open(cls):
        if not cls.is_open():
            raise RuntimeError("Market is closed")
