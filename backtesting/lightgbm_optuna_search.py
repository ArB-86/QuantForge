import optuna
import joblib
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

print("Loading...")

df = pd.read_csv(
    "data/training/master_v9.csv"
)

df["Date"] = pd.to_datetime(df["Date"])

FEATURES = [
    "EMA20","EMA50","EMA200",
    "RSI14","MACD","MACD_SIGNAL","MACD_HIST",
    "ATR14","BB_UPPER","BB_LOWER","VWAP",
    "RETURN_1D","RETURN_5D","LOG_RETURN",
    "RSI14_RANK","RETURN_5D_RANK",
    "Volume_RANK","ATR14_RANK",
    "VOL_20D",
    "BULL_REGIME",
    "HIGH_VOL_REGIME",

    "RETURN_20D",
    "RETURN_60D",
    "RETURN_120D",

    "RETURN_20D_RANK",
    "RETURN_60D_RANK",
    "RETURN_120D_RANK",

    "ATR_PCT",
    "EMA20_OVER_EMA200",
    "VOLUME_RATIO_20D",

    "RETURN_250D",
    "RETURN_250D_RANK",

    "RETURN_20D_MINUS_5D",
    "RETURN_120D_MINUS_20D",

    "PRICE_TO_52W_HIGH"
]

TARGET = "TARGET_5D_RETURN"

df = df.dropna(
    subset=FEATURES+[TARGET]
)

train = df[
    df["Date"] < "2019-01-01"
]

valid = df[
    (df["Date"] >= "2019-01-01")
    &
    (df["Date"] < "2020-01-01")
]

X_train = train[FEATURES]
y_train = train[TARGET]

X_valid = valid[FEATURES]
y_valid = valid[TARGET]


def objective(trial):

    params = {

        "device":"gpu",

        "objective":"regression",

        "metric":"rmse",

        "learning_rate":
        trial.suggest_float(
            "learning_rate",
            0.01,
            0.10
        ),

        "num_leaves":
        trial.suggest_int(
            "num_leaves",
            31,
            255
        ),

        "max_depth":
        trial.suggest_int(
            "max_depth",
            4,
            12
        ),

        "feature_fraction":
        trial.suggest_float(
            "feature_fraction",
            0.6,
            1.0
        ),

        "bagging_fraction":
        trial.suggest_float(
            "bagging_fraction",
            0.6,
            1.0
        ),

        "bagging_freq":1,

        "min_child_samples":
        trial.suggest_int(
            "min_child_samples",
            20,
            200
        ),

        "lambda_l1":
        trial.suggest_float(
            "lambda_l1",
            0,
            5
        ),

        "lambda_l2":
        trial.suggest_float(
            "lambda_l2",
            0,
            5
        ),

        "n_estimators":500,

        "force_col_wise":True,

        "n_jobs":8,

        "random_state":42,

        "verbose":-1
    }

    model = LGBMRegressor(
        **params
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_valid
    )

    rmse = (
        mean_squared_error(
            y_valid,
            pred
        ) ** 0.5
    )

    return rmse


study = optuna.create_study(
    direction="minimize"
)

study.optimize(
    objective,
    n_trials=100
)

print("\n========================")
print("BEST PARAMS")
print("========================")
print(study.best_params)

print("\nBest RMSE:", study.best_value)
