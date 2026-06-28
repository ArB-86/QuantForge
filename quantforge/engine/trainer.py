import subprocess

def train():

    print("="*60)
    print("TRAINING")
    print("="*60)

    subprocess.run(
        [
            "python",
            "backtesting/monthly_walkforward_regression_v15.py"
        ],
        check=True
    )
