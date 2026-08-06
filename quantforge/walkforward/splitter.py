from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: object
    train_end: object
    test_start: object
    test_end: object

    def __repr__(self):
        return (
            f"WalkForwardWindow("
            f"train=[{self.train_start} -> {self.train_end}], "
            f"test=[{self.test_start} -> {self.test_end}])"
        )


class WalkForwardSplitter:

    def __init__(self, dates, train_size, test_size):
        self.dates = list(dates)
        self.train_size = int(train_size)
        self.test_size = int(test_size)

    def split(self):
        windows = []

        n = len(self.dates)

        start = 0

        while start + self.train_size + self.test_size <= n:

            train = self.dates[start:start + self.train_size]

            test = self.dates[
                start + self.train_size:
                start + self.train_size + self.test_size
            ]

            windows.append(
                WalkForwardWindow(
                    train_start=train[0],
                    train_end=train[-1],
                    test_start=test[0],
                    test_end=test[-1],
                )
            )

            start += self.test_size

        return windows
