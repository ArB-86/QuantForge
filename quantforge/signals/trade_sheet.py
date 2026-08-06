import pandas as pd


class TradeSheetGenerator:

    def generate(self, signals):

        rows = []

        for s in signals:

            if s.action != "BUY":
                continue

            rr = (
                (s.target - s.entry)
                /
                (s.entry - s.stop_loss)
            )

            rows.append({

                "Ticker": s.ticker,

                "Action": s.action,

                "Entry": round(s.entry,2),

                "Stop": round(s.stop_loss,2),

                "Target": round(s.target,2),

                "Confidence": round(
                    s.confidence*100,
                    2,
                ),

                "ExpectedReturn": round(
                    s.expected_return*100,
                    2,
                ),

                "Allocation": round(
                    s.allocation*100,
                    2,
                ),

                "Quantity": s.quantity,

                "Capital": round(
                    s.position_value,
                    2,
                ),

                "RiskReward": round(rr,2),

            })

        df = pd.DataFrame(rows)

        if len(df):

            df = df.sort_values(

                "Confidence",

                ascending=False,

            )

        return df
