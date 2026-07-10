class AllocationEngine:

    def __init__(
        self,
        max_position=0.10,
        max_portfolio=1.0,
    ):
        self.max_position = max_position
        self.max_portfolio = max_portfolio

    def allocate(self, signals):

        buys = [s for s in signals if s.action == "BUY"]

        if not buys:
            return signals

        total = sum(max(s.score, 0.0) for s in buys)

        if total == 0:
            w = min(
                self.max_position,
                self.max_portfolio / len(buys),
            )

            for s in buys:
                s.allocation = w

            return signals

        allocated = 0.0

        for s in sorted(buys, key=lambda x: x.score, reverse=True):

            w = min(
                self.max_position,
                self.max_portfolio * s.score / total,
            )

            s.allocation = w
            allocated += w

        if allocated > self.max_portfolio:
            scale = self.max_portfolio / allocated

            for s in buys:
                s.allocation *= scale

        return signals
