from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
import pandas as pd

from quantforge.automl.engine import OptunaEngine
from quantforge.research.runner import ExperimentRunner


@dataclass
class WalkForwardWindow:
    index: int
    name: str
    train_start: Optional[str]
    train_end: Optional[str]
    valid_start: Optional[str]
    valid_end: Optional[str]
    test_start: Optional[str]
    test_end: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WalkForwardStudyManager:
    """Manage rolling-window Optuna optimization runs."""

    def __init__(
        self,
        base_config: Dict[str, Any],
        runner: Optional[ExperimentRunner] = None,
        output_dir: str | Path = "results/walkforward_optuna",
    ):
        self.base_config = dict(base_config)
        self.runner = runner or ExperimentRunner()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_dates(self, cfg: Dict[str, Any]) -> pd.Series:
        data_path = Path(cfg["data_path"])
        df = pd.read_parquet(data_path, columns=["Date"])
        dates = pd.to_datetime(df["Date"], errors="coerce").dropna().drop_duplicates().sort_values()
        return dates.reset_index(drop=True)

    def build_windows(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        train_size: int = 252,
        valid_size: int = 20,
        test_size: int = 20,
        step: int = 20,
        max_windows: Optional[int] = None,
    ) -> List[WalkForwardWindow]:
        cfg = dict(cfg or self.base_config)
        dates = self._load_dates(cfg)
        if len(dates) < train_size + valid_size + test_size:
            raise ValueError("Not enough rows to build walk-forward windows")

        windows: List[WalkForwardWindow] = []
        end_idx = len(dates) - test_size
        window_idx = 0

        for anchor in range(train_size + valid_size, end_idx, step):
            if max_windows is not None and len(windows) >= max_windows:
                break

            train_start_idx = max(0, anchor - train_size - valid_size)
            train_end_idx = anchor - valid_size - 1
            valid_start_idx = anchor - valid_size
            valid_end_idx = anchor - 1
            test_start_idx = anchor
            test_end_idx = min(anchor + test_size - 1, len(dates) - 1)

            windows.append(
                WalkForwardWindow(
                    index=window_idx,
                    name=f"window_{window_idx:04d}",
                    train_start=str(dates.iloc[train_start_idx].date()),
                    train_end=str(dates.iloc[train_end_idx].date()),
                    valid_start=str(dates.iloc[valid_start_idx].date()),
                    valid_end=str(dates.iloc[valid_end_idx].date()),
                    test_start=str(dates.iloc[test_start_idx].date()),
                    test_end=str(dates.iloc[test_end_idx].date()),
                )
            )
            window_idx += 1

        if not windows:
            raise ValueError("No walk-forward windows generated")

        return windows

    def _window_trial_storage(self, window: WalkForwardWindow, trial_id: int) -> Dict[str, str]:
        window_dir = self.output_dir / window.name
        trial_dir = window_dir / f"trial_{trial_id:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        return {
            "window_dir": str(window_dir),
            "trial_dir": str(trial_dir),
            "metrics_file": str(trial_dir / "metrics.json"),
            "summary_file": str(trial_dir / "summary.json"),
        }

    def optimize_window(
        self,
        window: WalkForwardWindow,
        n_trials: int = 50,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        load_if_exists: bool = True,
    ) -> optuna.Study:
        cfg = dict(self.base_config)
        cfg["window"] = window.to_dict()
        cfg["window_name"] = window.name
        cfg["window_index"] = window.index

        study_name = study_name or f"{self.base_config.get('name', 'QuantForge')}_{window.name}"

        engine = OptunaEngine(cfg, self._run_single_trial)
        study = engine.optimize(
            n_trials=n_trials,
            storage=storage,
            study_name=study_name,
            load_if_exists=load_if_exists,
        )

        summary = {
            "window": window.to_dict(),
            "study_name": study.study_name,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial_number": study.best_trial.number,
            "storage": storage,
        }
        window_dir = self.output_dir / window.name
        window_dir.mkdir(parents=True, exist_ok=True)
        (window_dir / "window_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return study

    def _run_single_trial(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return self.runner(cfg).metrics

    def run(
        self,
        n_trials_per_window: int = 50,
        storage: Optional[str] = None,
        load_if_exists: bool = True,
        max_windows: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        windows = self.build_windows(max_windows=max_windows)
        results: List[Dict[str, Any]] = []

        for window in windows:
            study = self.optimize_window(
                window=window,
                n_trials=n_trials_per_window,
                storage=storage,
                load_if_exists=load_if_exists,
            )
            results.append(
                {
                    "window": window.to_dict(),
                    "study_name": study.study_name,
                    "best_value": study.best_value,
                    "best_params": study.best_params,
                    "best_trial_number": study.best_trial.number,
                }
            )

        (self.output_dir / "walkforward_summary.json").write_text(json.dumps(results, indent=2, default=str))
        return results
