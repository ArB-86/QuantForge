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


REGISTRY = {

    "v3": FEATURES_V3,

}


def get_features(name):

    return REGISTRY[name]
