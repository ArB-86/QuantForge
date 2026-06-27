from collections.abc import Mapping


def _set_turnover(previous, current):
    if len(previous) == 0:
        return 1.0

    overlap = len(previous & current)

    return (len(current) - overlap) / max(len(current), 1)


def _weight_turnover(previous, current):
    """
    Portfolio turnover using portfolio weights.

    previous/current:
        {
            "RELIANCE": 0.12,
            "TCS": 0.08,
            ...
        }

    Returns
    -------
    Fraction of portfolio traded.
    """

    tickers = set(previous) | set(current)

    traded = 0.0

    for ticker in tickers:
        old = previous.get(ticker, 0.0)
        new = current.get(ticker, 0.0)

        traded += abs(new - old)

    return 0.5 * traded


def calculate_turnover(previous, current):
    """
    Backward compatible turnover calculator.

    Supports:

    - set[str]
    - dict[str, weight]
    """

    if len(previous) == 0:
        return 1.0

    if isinstance(previous, Mapping) and isinstance(current, Mapping):
        return _weight_turnover(previous, current)

    return _set_turnover(previous, current)