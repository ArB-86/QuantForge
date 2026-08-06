FEATURES_V3 = [
    "RETURN_1D",
    "RETURN_5D",
    "RETURN_20D",
    "RETURN_60D",
    "RETURN_120D",
    "RETURN_250D",
    "LOG_RETURN",
    "RETURN_5D_RANK",
    "RETURN_20D_RANK",
    "RETURN_60D_RANK",
    "RETURN_120D_RANK",
    "RETURN_250D_RANK",
    "Volume_RANK",
    "ATR14_RANK",
    "SIZE_RANK",
    "VOL_20D",
    "ATR14",
    "ATR_PCT",
    "EMA20",
    "EMA50",
    "EMA200",
    "EMA20_OVER_EMA200",
    "RSI14",
    "RSI14_RANK",
    "MACD",
    "MACD_SIGNAL",
    "MACD_HIST",
    "Volume",
    "DOLLAR_VOLUME",
    "LOG_DOLLAR_VOLUME",
    "VOLUME_RATIO_20D",
    "PRICE_TO_52W_HIGH",
    "BB_UPPER",
    "BB_LOWER",
    "VWAP",
    "MARKET_RET_20D",
    "RETURN_20D_MINUS_5D",
    "RETURN_120D_MINUS_20D",
    "BULL_REGIME",
    "HIGH_VOL_REGIME",
]

FEATURES_V4 = [
    # Returns
    "RETURN_1D",
    "RETURN_5D",
    "RETURN_20D",
    "RETURN_60D",
    "RETURN_120D",
    "RETURN_250D",
    "LOG_RETURN",

    # Return ranks
    "RETURN_5D_RANK",
    "RETURN_20D_RANK",
    "RETURN_60D_RANK",
    "RETURN_120D_RANK",
    "RETURN_250D_RANK",

    # Momentum
    "MOM_20_5",
    "MOM_60_20",
    "MOM_120_60",
    "MOM_250_120",
    "ACCELERATION",
    "LONG_ACCELERATION",

    # Trend
    "EMA20",
    "EMA50",
    "EMA200",
    "EMA20_OVER_EMA200",
    "EMA20_DISTANCE",
    "EMA50_DISTANCE",
    "EMA200_DISTANCE",
    "EMA20_50_SPREAD",
    "EMA50_200_SPREAD",
    "PRICE_VS_VWAP",

    # RSI / MACD
    "RSI14",
    "RSI14_RANK",
    "MACD",
    "MACD_SIGNAL",
    "MACD_HIST",

    # Volatility
    "VOL_20D",
    "ATR14",
    "ATR_PCT",
    "ATR_NORMALIZED",
    "TRUE_RANGE",
    "PARKINSON_VOL",
    "GARMAN_KLASS_VOL",
    "BB_WIDTH",

    # Liquidity
    "Volume",
    "Volume_RANK",
    "DOLLAR_VOLUME",
    "LOG_DOLLAR_VOLUME",
    "SIZE_RANK",
    "VOLUME_RATIO_20D",
    "ILLIQUIDITY",
    "VOLUME_SHOCK",
    "DOLLAR_VOLUME_LOG_DIFF",

    # Statistical
    "ROLL_STD20",
    "ROLL_SKEW20",
    "ROLL_KURT20",
    "ZSCORE_20D",

    # Market
    "MARKET_RET_20D",
    "RELATIVE_60D",
    "ATR_VOL_RATIO",
    "VOL_REGIME",
    "MOM_PERSISTENCE",

    # Existing
    "PRICE_TO_52W_HIGH",
    "BB_UPPER",
    "BB_LOWER",
    "VWAP",
    "RETURN_120D_MINUS_20D",
]

# Copy V4 to V5 and remove the eight features
FEATURES_V5 = [
    f for f in FEATURES_V4
    if f not in {
        "MOM_20_5",
        "ACCELERATION",
        "PARKINSON_VOL",
        "GARMAN_KLASS_VOL",
        "TRUE_RANGE",
        "BB_WIDTH",
        "ILLIQUIDITY",
        "MOM_PERSISTENCE",
    }
]

REGISTRY = {
    "v3": FEATURES_V3,
    "v4": FEATURES_V4,
    "v5": FEATURES_V5,
}


def get_features(name):
    return REGISTRY[name]
