import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def feature_importance(model_file, features):

    model = joblib.load(model_file)

    imp = pd.DataFrame({

        "Feature": features,

        "Importance": model.feature_importances_,

    })

    imp = imp.sort_values(

        "Importance",

        ascending=False,

    )

    print()

    print(imp)

    Path("results").mkdir(

        exist_ok=True,

    )

    imp.to_csv(

        "results/feature_importance.csv",

        index=False,

    )

    plt.figure(

        figsize=(10,12)

    )

    plt.barh(

        imp.Feature,

        imp.Importance,

    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(

        "results/feature_importance.png",

        dpi=200,

    )

    print()

    print("Saved results/")
