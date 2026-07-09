import pandas as pd


class MonthlyLoop:

    def __init__(
        self,
        df,
        features,
        target,
        model_manager,
        checkpoint_manager,
        purge_days=5,
    ):

        self.df = df
        self.features = features
        self.target = target

        self.model_manager = model_manager
        self.checkpoint_manager = checkpoint_manager

        self.purge_days = purge_days

        self.trading_dates = sorted(df["Date"].unique())

    def build_months(
        self,
        start="2016-01-01",
    ):

        return pd.date_range(
            start=pd.Timestamp(start),
            end=self.df["Date"].max(),
            freq="MS",
        )

    def run(self):

        completed = self.checkpoint_manager.completed_months()

        months = self.build_months()

        model = None

        for i in range(
            36,
            len(months) - 1,
        ):

            train_end = months[i]

            test_start = months[i]

            test_end = months[i + 1]

            month_key = test_start.to_period("M").strftime("%Y-%m")

            if month_key in completed:

                print(
                    "Skipping:",
                    month_key,
                )

                continue

            train_dates = [d for d in self.trading_dates if d < train_end]

            if len(train_dates) <= self.purge_days:

                continue

            purge_end = train_dates[-self.purge_days]

            train = self.df[self.df["Date"] < purge_end]

            test = self.df[
                (self.df["Date"] >= test_start) & (self.df["Date"] < test_end)
            ]

            if len(test) == 0:

                continue

            print()

            print("=" * 80)

            print(month_key)

            print(
                len(train),
                "train rows",
            )

            print(
                len(test),
                "test rows",
            )

            model = self.model_manager

            model.fit(
                train[self.features],
                train[self.target],
            )

            pred = model.predict(test[self.features])

            out = test.copy()

            out["PRED_RETURN"] = pred

            self.checkpoint_manager.append_predictions(out)

            print(
                "Saved",
                month_key,
            )

        if model is not None:

            self.checkpoint_manager.save_model(model)
