from quantforge.portfolio.equal_weight import (
    build_equal_weight_portfolio,
)

from quantforge.portfolio.inverse_volatility import (
    build_inverse_volatility_portfolio,
)

from quantforge.portfolio.score_weight import (
    build_score_weight_portfolio,
)

ALLOCATORS = {

    "equal_weight":
        build_equal_weight_portfolio,

    "inverse_volatility":
        build_inverse_volatility_portfolio,

    "score_weight":
        build_score_weight_portfolio,

}


def build_portfolio(
    df,
    method="equal_weight",
    **kwargs,
):

    try:

        return ALLOCATORS[
            method.lower()
        ](
            df,
            **kwargs,
        )

    except KeyError:

        raise ValueError(
            f"Unknown portfolio method: {method}"
        )