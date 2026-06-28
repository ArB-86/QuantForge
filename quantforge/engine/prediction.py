from quantforge.analysis.prediction_engine import PredictionEngine


def prediction(predictions):
    """
    Run prediction analysis.

    Parameters
    ----------
    predictions : pandas.DataFrame

    Returns
    -------
    PredictionEngine
    """

    print("=" * 80)
    print("PREDICTION")
    print("=" * 80)

    return PredictionEngine(predictions)