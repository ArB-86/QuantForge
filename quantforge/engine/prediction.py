import subprocess

def prediction():

    print("="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)

    subprocess.run(
        [
            "python",
            "analysis/prediction_analysis.py"
        ],
        check=True
    )
