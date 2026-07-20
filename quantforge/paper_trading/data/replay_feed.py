import pandas as pd


class ReplayFeed:
    def __init__(self, dataframe):
        self.df = dataframe.sort_values(["Date", "Ticker"])

    @classmethod
    def from_csv(cls, path):
        return cls(pd.read_csv(path, low_memory=False))

    def __iter__(self):
        for date, frame in self.df.groupby("Date"):
            prices = {}

            for _, row in frame.iterrows():
                prices[row["Ticker"]] = float(row["Close"])

            yield {
                "date": str(date),
                "prices": prices,
            }
