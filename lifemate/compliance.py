"""Competition deadline and reproducibility controls shared across backends."""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class SampleDeadlineExceeded(TimeoutError):
    """The complete per-sample wall-clock budget has expired."""


@dataclass(frozen=True)
class SampleDeadline:
    """Monotonic deadline created before any sample-specific processing."""

    timeout_seconds: float
    started_at: float
    expires_at: float
    clock: Callable[[], float] = time.monotonic

    @classmethod
    def start(
        cls,
        timeout_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> SampleDeadline:
        if timeout_seconds <= 0:
            raise ValueError("sample_timeout_seconds must be positive")
        started = clock()
        return cls(timeout_seconds, started, started + timeout_seconds, clock)

    def remaining_seconds(self) -> float:
        return max(0.0, self.expires_at - self.clock())

    def check(self, operation: str = "sample inference") -> None:
        if self.remaining_seconds() <= 0:
            raise SampleDeadlineExceeded(
                f"sample_timeout: {operation} exceeded {self.timeout_seconds:g}s wall-clock budget"
            )

    def timeout_for(self, configured_timeout: float, operation: str) -> float:
        self.check(operation)
        return min(configured_timeout, self.remaining_seconds())

    def require_duration(self, seconds: float, operation: str) -> None:
        self.check(operation)
        if seconds >= self.remaining_seconds():
            raise SampleDeadlineExceeded(
                f"sample_timeout: insufficient time for {operation}; "
                f"remaining={self.remaining_seconds():.3f}s required={seconds:.3f}s"
            )


@dataclass(frozen=True)
class ReproducibilityReport:
    seed: int
    python_seeded: bool
    numpy_seeded: bool
    torch_seeded: bool
    cuda_seeded: bool
    deterministic_torch: bool
    limitations: tuple[str, ...]


def configure_reproducibility(seed: int = 42, *, deterministic: bool = True) -> ReproducibilityReport:
    """Seed available runtimes before model initialization.

    ``PYTHONHASHSEED`` only controls hash randomization when set before process
    startup. It is still exported here for child processes, and the limitation
    is reported explicitly for the already-running interpreter.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)

    limitations = ["PYTHONHASHSEED takes full effect only when set before interpreter startup"]
    numpy_seeded = False
    try:
        import numpy

        numpy.random.seed(seed)
        numpy_seeded = True
    except ImportError:
        limitations.append("NumPy is not installed in this environment")

    torch_seeded = False
    cuda_seeded = False
    deterministic_torch = False
    try:
        import torch

        torch.manual_seed(seed)
        torch_seeded = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            cuda_seeded = True
        if deterministic:
            use_deterministic = getattr(torch, "use_deterministic_algorithms", None)
            if callable(use_deterministic):
                use_deterministic(True, warn_only=True)
                deterministic_torch = True
            cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
            if cudnn is not None:
                cudnn.benchmark = False
                cudnn.deterministic = True
                if hasattr(cudnn, "allow_tf32"):
                    cudnn.allow_tf32 = False
            cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
            matmul = getattr(cuda_backend, "matmul", None)
            if matmul is not None and hasattr(matmul, "allow_tf32"):
                matmul.allow_tf32 = False
    except ImportError:
        limitations.append("PyTorch is not installed in this environment")

    if deterministic and torch_seeded:
        limitations.append("PyTorch may warn or fail when a backend has no deterministic implementation")
    return ReproducibilityReport(
        seed=seed,
        python_seeded=True,
        numpy_seeded=numpy_seeded,
        torch_seeded=torch_seeded,
        cuda_seeded=cuda_seeded,
        deterministic_torch=deterministic_torch,
        limitations=tuple(limitations),
    )


def configure_greedy_generation(model: Any) -> bool:
    """Set deterministic greedy decoding on a Transformers-style model."""

    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return False
    generation_config.do_sample = False
    generation_config.num_beams = 1
    return True
