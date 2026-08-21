"""Backend-neutral inference contract used by the skeleton pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class HandEstimate:
    """One hand represented in the project's MediaPipe 21-point topology."""

    image_joints: np.ndarray
    spatial_joints: np.ndarray | None
    handedness: str | None = None
    handedness_score: float | None = None
    spatial_metadata: dict[str, Any] = field(default_factory=dict)
    backend_details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EstimationResult:
    hands: list[HandEstimate]
    estimator_metadata: dict[str, Any]
    inference_time_ms: float


@runtime_checkable
class Hand3DEstimator(Protocol):
    """Minimal boundary implemented by MediaPipe and HaMeR backends."""

    def estimate(self, rgb_image: np.ndarray) -> EstimationResult: ...

    def close(self) -> None: ...

