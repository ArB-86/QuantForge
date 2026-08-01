from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WalkForwardResult:
    window: Any
    experiment_id: str
    metrics: dict
    experiment_dir: str
