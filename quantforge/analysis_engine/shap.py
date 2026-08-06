from quantforge.diagnostics.shap import SHAPEngine


def shap(shap_df):

    print("=" * 80)
    print("SHAP")
    print("=" * 80)

    return SHAPEngine(shap_df)