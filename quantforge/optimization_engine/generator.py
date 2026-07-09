import itertools
import json
from pathlib import Path

SPACE = {

    "learning_rate":[0.01,0.03,0.05],

    "num_leaves":[31,64,128],

    "top_n":[10,15,20],

    "holding_days":[5,10,20]

}

OUT = Path("configs/generated")
OUT.mkdir(parents=True,exist_ok=True)

i=1

for lr,nl,top,hold in itertools.product(

    SPACE["learning_rate"],
    SPACE["num_leaves"],
    SPACE["top_n"],
    SPACE["holding_days"]

):

    cfg={

        "name":f"EXP_{i:04d}",

        "model":"lightgbm",

        "target":"TARGET_20D_RETURN",

        "learning_rate":lr,

        "num_leaves":nl,

        "n_estimators":500,

        "subsample":0.8,

        "colsample_bytree":0.8,

        "random_state":42,

        "n_jobs":-1,

        "top_n":top,

        "holding_days":hold,

        "transaction_cost":0.003,

        "feature_store":"v3"

    }

    with open(

        OUT/f"exp_{i:04d}.json",

        "w"

    ) as fp:

        json.dump(

            cfg,

            fp,

            indent=4

        )

    i+=1

print(i-1,"configs generated")
