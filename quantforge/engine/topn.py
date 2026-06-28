import subprocess

def topn():

    print("="*60)
    print("TOP-N SWEEP")
    print("="*60)

    subprocess.run(
        [
            "python",
            "analysis/topn_sweep.py"
        ],
        check=True
    )
