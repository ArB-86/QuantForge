from lightgbm import (
    LGBMRegressor,
    LGBMRanker,
)


def build_lgbm_regressor(
    random_state=42,
    n_jobs=8,
    **params,
):

    defaults = dict(

        device="gpu",

        objective="regression",

        metric="rmse",

        force_col_wise=True,

        verbose=-1,

        random_state=random_state,

        n_jobs=n_jobs,

    )

    defaults.update(params)

    return LGBMRegressor(
        **defaults
    )


def build_lgbm_ranker(
    random_state=42,
    n_jobs=8,
    **params,
):

    defaults = dict(

        device="gpu",

        objective="lambdarank",

        metric="ndcg",

        force_col_wise=True,

        verbose=-1,

        random_state=random_state,

        n_jobs=n_jobs,

    )

    defaults.update(params)

    return LGBMRanker(
        **defaults
    )
