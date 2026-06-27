from quantforge.portfolio.equal_weight import (
    build_equal_weight_portfolio,
)

from quantforge.portfolio.inverse_volatility import (
    build_inverse_volatility_portfolio,
)


def build_portfolio(
    df,
    method="inverse_volatility",
    **kwargs,
):
    """
    Unified portfolio construction interface.

    Parameters
    ----------
    method : str
        equal_weight
        inverse_volatility
    """

    method = method.lower()

    if method == "equal_weight":
        return build_equal_weight_portfolio(
            df,
            **kwargs,
        )

    if method == "inverse_volatility":
        return build_inverse_volatility_portfolio(
            df,
            **kwargs,
        )

    raise ValueError(
        f"Unknown portfolio method: {method}"
    )
