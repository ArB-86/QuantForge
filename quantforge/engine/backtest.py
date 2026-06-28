import subprocess

def backtest():

    print("="*60)
    print("BACKTEST")
    print("="*60)

    subprocess.run(
        [
            "python",
            "backtesting/portfolio_backtest_v15.py"
        ],
        check=True
    )
