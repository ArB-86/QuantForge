from __future__ import annotations

from typing import Any, Dict, List, Optional

from quantforge.walkforward.optuna_walkforward import (
    WalkForwardStudyManager,
    WalkForwardWindow,
)


class WalkForwardEngine:
    """Compatibility wrapper around the walk-forward Optuna manager."""

    def __init__(
        self,
        base_config: Dict[str, Any],
        runner=None,
        output_dir: str = "results/walkforward_optuna",
    ):
        self.manager = WalkForwardStudyManager(
            base_config=base_config,
            runner=runner,
            output_dir=output_dir,
        )

    def run(
        self,
        n_trials_per_window: int = 50,
        storage: Optional[str] = None,
        load_if_exists: bool = True,
        max_windows: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return self.manager.run(
            n_trials_per_window=n_trials_per_window,
            storage=storage,
            load_if_exists=load_if_exists,
            max_windows=max_windows,
        )

    def build_windows(self, *args, **kwargs):
        return self.manager.build_windows(*args, **kwargs)

    def optimize_window(self, *args, **kwargs):
        return self.manager.optimize_window(*args, **kwargs)
