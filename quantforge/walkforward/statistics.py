import pandas as pd

class WalkForwardStatistics:
    @staticmethod
    def build(df):
        numeric = df.select_dtypes("number")
        stats = pd.DataFrame()
        stats["Mean"]   = numeric.mean()
        stats["Median"] = numeric.median()
        stats["Std"]    = numeric.std()
        stats["Min"]    = numeric.min()
        stats["Max"]    = numeric.max()
        stats["Q25"]    = numeric.quantile(0.25)
        stats["Q75"]    = numeric.quantile(0.75)
        stats["IQR"]    = stats["Q75"] - stats["Q25"]
        return stats
