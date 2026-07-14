from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class AlphaReport:

    def __init__(self, df):

        self.df = df.copy()

        self.output = Path("results")
        self.output.mkdir(
            exist_ok=True,
        )

    def monthly_ic(
        self,
        alpha="COMPOSITE_ALPHA",
        target="TARGET_20D_RETURN",
    ):

        self.df["Month"] = (
            pd.to_datetime(
                self.df["Date"]
            )
            .dt.to_period("M")
            .astype(str)
        )

        ic = (
            self.df.groupby("Month")
            .apply(
                lambda x:
                x[alpha].corr(
                    x[target],
                    method="spearman",
                )
            )
            .rename("IC")
            .reset_index()
        )

        ic.to_csv(
            self.output /
            "monthly_ic.csv",
            index=False,
        )

        plt.figure(figsize=(12,4))
        plt.plot(ic["Month"], ic["IC"])
        plt.xticks(rotation=90)
        plt.tight_layout()

        plt.savefig(
            self.output /
            "monthly_ic.png"
        )

        plt.close()

        return ic

    def long_short_spread(
        self,
        target="TARGET_20D_RETURN",
    ):

        # ---- FIX: Create Decile if missing ----
        if "Decile" not in self.df.columns:
            self.df["Decile"] = (
                self.df.groupby("Date")[
                    "COMPOSITE_ALPHA"
                ]
                .transform(
                    lambda s:
                    pd.qcut(
                        s.rank(method="first"),
                        10,
                        labels=False,
                        duplicates="drop",
                    )
                )
            )

        dec = (
            self.df.groupby(
                "Decile"
            )[target]
            .mean()
        )

        spread = (
            dec.loc[9]
            -
            dec.loc[0]
        )

        pd.DataFrame(
            {
                "Long":
                [dec.loc[9]],

                "Short":
                [dec.loc[0]],

                "Spread":
                [spread],
            }
        ).to_csv(
            self.output /
            "long_short.csv",
            index=False,
        )

        return spread

    def rolling_ic(
        self,
        alpha="COMPOSITE_ALPHA",
        target="TARGET_20D_RETURN",
        window=60,
    ):

        daily = (
            self.df.groupby("Date")
            .apply(
                lambda x:
                x[alpha].corr(
                    x[target],
                    method="spearman",
                )
            )
        )

        rolling = (
            daily
            .rolling(window)
            .mean()
        )

        rolling.to_csv(
            self.output /
            "rolling_ic.csv"
        )

        plt.figure(figsize=(12,4))
        plt.plot(rolling.index, rolling.values)
        plt.tight_layout()

        plt.savefig(
            self.output /
            "rolling_ic.png"
        )

        plt.close()

    def correlation_matrix(self):

        cols = [

            "PRED_RETURN",

            "MomentumScore",

            "TrendScore",

            "VolatilityScore",

            "LiquidityScore",

            "MeanReversionScore",

            "COMPOSITE_ALPHA",

        ]

        cols = [
            c
            for c in cols
            if c in self.df.columns
        ]

        corr = (
            self.df[cols]
            .corr(
                method="spearman"
            )
        )

        corr.to_csv(
            self.output /
            "factor_correlation.csv"
        )

        plt.figure(figsize=(8,8))
        plt.imshow(corr)
        plt.colorbar()
        plt.xticks(
            range(len(cols)),
            cols,
            rotation=90,
        )
        plt.yticks(
            range(len(cols)),
            cols,
        )

        plt.tight_layout()

        plt.savefig(
            self.output /
            "factor_correlation.png"
        )

        plt.close()

    def generate(self):

        print()

        print("="*80)
        print("ALPHA REPORT")
        print("="*80)

        self.monthly_ic()

        spread = self.long_short_spread()

        self.rolling_ic()

        self.correlation_matrix()

        print()

        print(
            "Long Short Spread:",
            spread,
        )

        print()

        print(
            "Saved reports to results/"
        )
