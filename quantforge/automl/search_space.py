SEARCH_SPACE = {
    "lightgbm": {
        "learning_rate": (0.02, 0.04),
        "num_leaves": (48, 96),
        "max_depth": (4, 7),
        "subsample": (0.75, 0.90),
        "colsample_bytree": (0.75, 0.90),
        "min_child_samples": (10, 60),
        "reg_alpha": (0.0, 1.0),
        "reg_lambda": (0.0, 1.0),
    },
    "catboost": {
        "iterations": (300, 1200),
        "learning_rate": (0.01, 0.08),
        "depth": (4, 10),
        "l2_leaf_reg": (1.0, 15.0),
        "bagging_temperature": (0.0, 1.0),
        "random_strength": (0.0, 5.0),
        "border_count": (32, 255),
    },
}
