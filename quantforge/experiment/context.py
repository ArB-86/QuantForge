from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class ExperimentContext:
    # Identity
    experiment_id: str = ""
    name: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    # Dataset
    dataset: Any = None
    dataset_path: str = ""
    feature_names: List[str] = field(default_factory=list)
    target: str = ""

    # Training
    model: Any = None
    predictions: Any = None
    portfolio: Any = None

    # Results
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Diagnostics
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Artifacts
    artifacts: Dict[str, str] = field(default_factory=dict)

    # Execution
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    status: str = "CREATED"
