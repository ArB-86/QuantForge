from dataclasses import dataclass


@dataclass
class TradeSignal:
    ticker: str
    action: str
    score: float
    confidence: float
    entry: float
    stop_loss: float
    target: float
    expected_return: float
    allocation: float


class SignalEngine:

    def __init__(
        self,
        buy_threshold=0.60,
        sell_threshold=0.40,
        risk_reward=2.0,
        stop_pct=0.03,
    ):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.risk_reward = risk_reward
        self.stop_pct = stop_pct

    def generate(self, predictions):

        signals = []

        for row in predictions.itertuples():

            price = row.close
            prob = row.probability_up

            if prob >= self.buy_threshold:

                stop = price * (1 - self.stop_pct)
                target = price + (price - stop) * self.risk_reward

                signals.append(
                    TradeSignal(
                        ticker=row.ticker,
                        action="BUY",
                        score=row.score,
                        confidence=prob,
                        entry=price,
                        stop_loss=stop,
                        target=target,
                        expected_return=row.expected_return,
                        allocation=0.0,
                    )
                )

            elif prob <= self.sell_threshold:

                stop = price * (1 + self.stop_pct)
                target = price - (stop - price) * self.risk_reward

                signals.append(
                    TradeSignal(
                        ticker=row.ticker,
                        action="SELL",
                        score=row.score,
                        confidence=1 - prob,
                        entry=price,
                        stop_loss=stop,
                        target=target,
                        expected_return=row.expected_return,
                        allocation=0.0,
                    )
                )

        return signals
