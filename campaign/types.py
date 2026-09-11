"""Stable result contracts for experiments 1–4."""
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

@dataclass
class DetectionResult:
    method: str
    flagged_ids: Set[int]
    scores: Dict[int, float] = field(default_factory=dict)
    contamination_budget: Optional[float] = None
    threshold: Optional[float] = None

@dataclass
class CorrectionResult:
    method: str
    removed_ids: Set[int] = field(default_factory=set)
    label_changes: Dict[int, int] = field(default_factory=dict)
    source_detection: str = ""

@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    seed: int = 42

@dataclass
class RunManifest:
    status: str
    config: Dict
    artifacts: Dict = field(default_factory=dict)
    completion_checks: Dict = field(default_factory=dict)
