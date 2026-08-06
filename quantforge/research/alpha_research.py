from pathlib import Path

import pandas as pd

from quantforge.alpha_factory import (
    AlphaEvaluator,
)
from quantforge.research.alpha_report import AlphaReport


def alpha_research(
    dataset,
):

    if isinstance(
        dataset,
        str,
    ):

        df = pd.read_csv(dataset, low_memory=False)

    else:

        df = dataset.copy()

    evaluator = AlphaEvaluator()

    df = evaluator.evaluate(df)

    AlphaReport(
        df
    ).generate()

    out = Path("results")

    out.mkdir(
        exist_ok=True,
    )

    df.to_csv(
        out / "alpha_dataset.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("ALPHA RESEARCH COMPLETE")
    print("=" * 80)
