from dataclasses import dataclass


@dataclass(slots=True)
class PortfolioConfig:

    method: str = "inverse_volatility"

    top_n: int = 10

    max_weight: float = 0.20

    min_volatility: float = 1e-3

    confidence_weighting: bool = False

    transaction_cost: float = 0.002

    holding_days: int = 5
