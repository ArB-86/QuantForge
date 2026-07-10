from pathlib import Path
import pandas as pd


class CSVDataFeed:

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

        if "Date" in self.df.columns:
            self.df["Date"] = pd.to_datetime(self.df["Date"])

    def __iter__(self):
        for _, row in self.df.iterrows():
            yield row.to_dict()
