EXPERIMENTS = {

    "baseline": {},

    "no_vwap": {
        "drop_features": [
            "VWAP",
            "PRICE_VS_VWAP",
        ]
    },

    "no_volatility": {
        "drop_prefixes": [
            "ATR",
            "VOL",
            "ROLL_",
            "PARKINSON",
            "GARMAN",
            "BB_",
        ]
    },

    "no_momentum": {
        "drop_prefixes": [
            "RETURN_",
            "MOM_",
            "RELATIVE",
            "ZSCORE",
        ]
    },

    "trend_only": {
        "keep_prefixes": [
            "EMA",
            "VWAP",
            "PRICE_VS",
        ]
    },

    "momentum_only": {
        "keep_prefixes": [
            "RETURN",
            "MOM",
            "RELATIVE",
            "ZSCORE",
        ]
    },

}
