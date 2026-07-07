from quantforge.portfolio.equal_weight import (
    build_equal_weight_portfolio,
)

from quantforge.portfolio.inverse_volatility import (
    build_inverse_volatility_portfolio,
)

from quantforge.portfolio.score_weight import (
    build_score_weight_portfolio,
)

from quantforge.portfolio.constraints import (
    PortfolioConstraints,
)
from quantforge.risk.exposure import ExposureManager

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

    # Remove max_stock_weight from kwargs before passing to allocator
    max_stock_weight = kwargs.pop(
        "max_stock_weight",
        1.0,
    )

    try:

        portfolio = ALLOCATORS[
            method.lower()
        ](
            df,
            **kwargs,
        )

    except KeyError:

        raise ValueError(
            f"Unknown portfolio method: {method}"
        )

    # Apply constraint after portfolio is built
    portfolio = PortfolioConstraints(
        max_weight=max_stock_weight,
    ).apply(portfolio)

    # Apply exposure management (ensure weights sum to 100%)
    portfolio = ExposureManager().apply(
        portfolio
    )

    return portfolio