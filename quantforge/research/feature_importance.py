import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib


def feature_importance(model_path, features, output_dir="results"):
    """
    Load a trained model and generate feature importance CSV and plot.

    Parameters
    ----------
    model_path : str or Path
        Path to the saved model (pickle).
    features : list of str
        List of feature names in the same order as the model was trained on.
    output_dir : str, default="results"
        Directory to save the output files.
    """
    model = joblib.load(model_path)

    # Ensure model has feature_importances_
    if not hasattr(model, "feature_importances_"):
        print("Model does not have feature_importances_ attribute.")
        return

    # Fix: handle mismatch between features list and model importances length
    n = min(
        len(features),
        len(model.feature_importances_),
    )

    if len(features) != len(model.feature_importances_):
        print(
            f"WARNING: feature count mismatch "
            f"(registry={len(features)}, "
            f"model={len(model.feature_importances_)})"
        )

    imp = pd.DataFrame({
        "Feature": features[:n],
        "Importance": model.feature_importances_[:n],
    })

    # Sort by importance descending
    imp = imp.sort_values("Importance", ascending=False)

    # Save CSV
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "feature_importance.csv"
    imp.to_csv(csv_path, index=False)
    print(f"Feature importance saved to {csv_path}")

    # Plot top 20 features
    top_n = min(20, len(imp))
    top = imp.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(top["Feature"], top["Importance"])
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    png_path = output_dir / "feature_importance.png"
    plt.savefig(png_path, dpi=150)
    print(f"Feature importance plot saved to {png_path}")
    plt.close()
