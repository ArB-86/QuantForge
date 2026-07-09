from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PipelineContext:

    config: Optional[dict] = None

    dataset: Any = None

    features: Any = None

    model: Any = None

    predictions: Any = None

    metrics: dict = field(default_factory=dict)

    artifacts: dict = field(default_factory=dict)

    experiment_id: str = ""

    metadata: dict = field(default_factory=dict)
