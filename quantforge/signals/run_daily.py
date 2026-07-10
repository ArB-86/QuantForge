import pandas as pd

from quantforge.signals.predict import PredictionEngine
from quantforge.signals.daily_pipeline import DailyTradingPipeline


def run(
    feature_file,
    model_file,
    capital=1_000_000,
):

    features = pd.read_parquet(feature_file)

    predictions = PredictionEngine(
        model_file
    ).predict(features)

    trades = DailyTradingPipeline(
        capital=capital,
    ).run(predictions)

    return trades


if __name__ == "__main__":

    trades = run(

        feature_file="data/latest_features.parquet",

        model_file="models/latest_model.pkl",

    )

    print(trades)

    trades.to_csv(
        "results/daily_trades.csv",
        index=False,
    )

    print()

    print("="*80)
    print("DAILY TRADE SHEET GENERATED")
    print("="*80)
