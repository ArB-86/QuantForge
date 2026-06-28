import subprocess

def shap():

    print("="*60)
    print("SHAP")
    print("="*60)

    subprocess.run(
        [
            "python",
            "analysis/shap_walkforward.py"
        ],
        check=True
    )
