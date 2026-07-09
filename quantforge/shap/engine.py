from quantforge.analysis.shap_engine import SHAPEngine


def shap(shap_df):

    print("=" * 80)
    print("SHAP")
    print("=" * 80)

    return SHAPEngine(shap_df)