from pathlib import Path

import pandas as pd

from .composite import CompositeAlpha


class AlphaEvaluator:

    def __init__(
        self,
        output_dir="results",
    ):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _daily_ic(
        self,
        df,
        column,
        target,
    ):

        return (
            df.groupby("Date")
            .apply(
                lambda x:
                x[column].corr(
                    x[target],
                    method="spearman",
                )
            )
        )

    def evaluate(
        self,
        df,
        target_column="TARGET_20D_RETURN",
    ):

        df = df.copy()

        composite = CompositeAlpha().compute(df)

        df["COMPOSITE_ALPHA"] = composite

        ic = self._daily_ic(
            df,
            "COMPOSITE_ALPHA",
            target_column,
        )

        report = pd.DataFrame(
            {
                "Metric": [
                    "Mean IC",
                    "Median IC",
                    "Std IC",
                    "Positive IC %",
                ],
                "Value": [
                    ic.mean(),
                    ic.median(),
                    ic.std(),
                    (ic > 0).mean(),
                ],
            }
        )

        report.to_csv(
            self.output_dir /
            "alpha_report.csv",
            index=False,
        )

        alpha_cols = [
            "PRED_RETURN",
            "COMPOSITE_ALPHA",
        ]

        corr = (
            df[alpha_cols]
            .corr(method="spearman")
        )

        corr.to_csv(
            self.output_dir /
            "alpha_correlation.csv"
        )

        # ---- FIX: Add Decile directly to df ----
        df["Decile"] = (
            df.groupby("Date")[
                "COMPOSITE_ALPHA"
            ]
            .transform(
                lambda s:
                pd.qcut(
                    s.rank(
                        method="first"
                    ),
                    10,
                    labels=False,
                    duplicates="drop",
                )
            )
        )

        decile_returns = (
            df.groupby("Decile")[
                target_column
            ]
            .mean()
            .reset_index()
        )

        decile_returns.to_csv(
            self.output_dir /
            "alpha_deciles.csv",
            index=False,
        )

        print()
        print("=" * 80)
        print("COMPOSITE ALPHA")
        print("=" * 80)
        print(report)
        print()
        print(decile_returns)

        return df
