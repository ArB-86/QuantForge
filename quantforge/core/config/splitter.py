MODEL_KEYS = {

    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "subsample",
    "colsample_bytree",
    "random_state",
    "objective",
    "n_jobs",
    "min_child_samples",
    "min_child_weight",
    "reg_alpha",
    "reg_lambda",
    "subsample_freq",
    "boosting_type"

}

def split(config):

    model = {}

    experiment = {}

    for k, v in config.items():

        if k in MODEL_KEYS:

            model[k] = v

        else:

            experiment[k] = v

    return model, experiment
