def apply_transaction_cost(
    gross_return,
    turnover,
    round_trip_cost=0.002
):
    """
    Apply transaction cost.

    Parameters
    ----------
    gross_return : float

    turnover : float

    round_trip_cost : float

    Returns
    -------
    net_return
    """

    transaction_cost = (
        turnover
        * round_trip_cost
    )

    return (
        gross_return
        - transaction_cost
    )
