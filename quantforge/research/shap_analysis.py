from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from quantforge.core.config.config import Config
from quantforge.features.registry import get_features


def shap_analysis(config):

    if isinstance(config, str):
        config = Config(config).dict()

    model = joblib.load(
        config["model_file"]
    )

    df = pd.read_csv(
        config["prediction_file"]
    )

    # Handle both string registry name and direct list
    feature_spec = config["features"]
    if isinstance(feature_spec, str):
        features = get_features(feature_spec)
    else:
        features = feature_spec

    # Ensure features exist in the dataframe
    missing = [
        f
        for f in features
        if f not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    # Sample a subset for SHAP (to keep runtime reasonable)
    sample = (
        df[features]
        .sample(
            n=min(5000, len(df)),
            random_state=42,
        )
        .copy()
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(sample)

    output = Path("results")
    output.mkdir(
        exist_ok=True,
    )

    # Mean absolute SHAP values per feature
    # If shap_values is a list (for multiclass), we take the first class
    if isinstance(shap_values, list):
        shap_vals = shap_values[0]  # binary classification
    else:
        shap_vals = shap_values

    mean_abs = (
        pd.DataFrame(
            {
                "Feature": features,
                "MeanAbsSHAP":
                abs(shap_vals).mean(axis=0),
            }
        )
        .sort_values(
            "MeanAbsSHAP",
            ascending=False,
        )
    )

    mean_abs["Rank"] = range(
        1,
        len(mean_abs) + 1,
    )

    mean_abs.to_csv(
        output / "shap_values.csv",
        index=False,
    )

    plt.figure(
        figsize=(10, 12)
    )

    shap.summary_plot(
        shap_vals,
        sample,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        output / "shap_summary.png",
        dpi=200,
    )

    plt.close()

    plt.figure(
        figsize=(10, 12)
    )

    shap.summary_plot(
        shap_vals,
        sample,
        plot_type="bar",
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        output / "shap_bar.png",
        dpi=200,
    )

    plt.close()

    print()
    print("=" * 80)
    print("SHAP ANALYSIS")
    print("=" * 80)
    print()
    print(mean_abs.head(20))
    print()
    print(
        "Saved:",
        output / "shap_values.csv",
    )
    print(
        "Saved:",
        output / "shap_summary.png",
    )
    print(
        "Saved:",
        output / "shap_bar.png",
    )
