from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentContext:

    # ======================================================
    # Identity
    # ======================================================

    experiment_id: str = ""
    name: str = ""

    created_at: datetime = field(default_factory=datetime.utcnow)

    git_commit: str = ""
    git_branch: str = ""

    seed: int = 42

    # ======================================================
    # Configuration
    # ======================================================

    config: Dict[str, Any] = field(default_factory=dict)

    config_hash: str = ""

    # ======================================================
    # Dataset
    # ======================================================

    dataset: Optional[Any] = None

    dataset_path: str = ""

    dataset_rows: int = 0
    dataset_columns: int = 0

    feature_names: List[str] = field(default_factory=list)

    target: str = ""

    # ======================================================
    # Training
    # ======================================================

    model: Optional[Any] = None

    predictions: Optional[Any] = None

    portfolio: Optional[Any] = None

    # ======================================================
    # Results
    # ======================================================

    metrics: Dict[str, Any] = field(default_factory=dict)

    # ======================================================
    # Diagnostics
    # ======================================================

    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ======================================================
    # Artifacts
    # ======================================================

    artifacts: Dict[str, str] = field(default_factory=dict)

    # ======================================================
    # Execution
    # ======================================================

    warnings: List[str] = field(default_factory=list)

    errors: List[str] = field(default_factory=list)

    status: str = "CREATED"

    started_at: Optional[datetime] = None

    finished_at: Optional[datetime] = None
