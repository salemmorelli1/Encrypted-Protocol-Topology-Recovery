"""Typed synthetic event tensors used by the simulation laboratory."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EventBatch:
    """A continuous-time marked event sequence with contiguous node indices."""

    times: torch.Tensor
    sizes: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    node_keys: tuple[str, ...]
    observation_horizon: float | None = None

    def __post_init__(self) -> None:
        tensors = (self.times, self.sizes, self.src, self.dst)
        if any(value.ndim != 1 for value in tensors):
            raise ValueError("event fields must be one-dimensional vectors")
        n = int(self.times.numel())
        if n < 2 or any(int(value.numel()) != n for value in tensors[1:]):
            raise ValueError("an event batch requires aligned vectors and at least two events")
        if not self.times.is_floating_point() or not self.sizes.is_floating_point():
            raise TypeError("times and sizes must use floating-point tensors")
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if self.src.dtype not in integer_dtypes or self.dst.dtype not in integer_dtypes:
            raise TypeError("source and destination indices must use integer tensors")
        if any(value.device != self.times.device for value in tensors[1:]):
            raise ValueError("event tensors must reside on one device")
        if not torch.isfinite(self.times).all() or not torch.isfinite(self.sizes).all():
            raise ValueError("times and sizes must be finite")
        if not torch.all(self.times[1:] >= self.times[:-1]):
            raise ValueError("event times must be nondecreasing")
        if float(self.times[0]) < 0 or torch.any(self.sizes <= 0):
            raise ValueError("times must be nonnegative and packet sizes positive")
        if len(self.node_keys) < 2:
            raise ValueError("at least two synthetic node keys are required")
        if len(set(self.node_keys)) != len(self.node_keys) or any(
            not isinstance(key, str) or not key for key in self.node_keys
        ):
            raise ValueError("node keys must be unique nonempty strings")
        if torch.any(self.src == self.dst):
            raise ValueError("self-links are excluded from this experiment")
        if min(int(self.src.min()), int(self.dst.min())) < 0:
            raise ValueError("node indices must be nonnegative")
        if max(int(self.src.max()), int(self.dst.max())) >= len(self.node_keys):
            raise ValueError("node index exceeds node dictionary")
        if self.observation_horizon is not None:
            if not math.isfinite(self.observation_horizon):
                raise ValueError("observation horizon must be finite")
            if self.observation_horizon <= float(self.times[-1]):
                raise ValueError("observation horizon must be later than the final event")

    @property
    def num_events(self) -> int:
        return int(self.times.numel())

    @property
    def num_nodes(self) -> int:
        return len(self.node_keys)

    @property
    def horizon(self) -> float:
        """Return the declared window end, or a conservative legacy fallback."""

        if self.observation_horizon is not None:
            return self.observation_horizon
        gaps = self.times[1:] - self.times[:-1]
        positive_gaps = gaps[gaps > 0]
        terminal_gap = float(torch.median(positive_gaps)) if positive_gaps.numel() else 1e-3
        return float(self.times[-1]) + max(terminal_gap, 1e-3)

    def to(self, device: torch.device | str) -> EventBatch:
        return EventBatch(
            self.times.to(device),
            self.sizes.to(device),
            self.src.to(device),
            self.dst.to(device),
            self.node_keys,
            self.observation_horizon,
        )

    def prefix(self, proportion: float) -> EventBatch:
        if not 0 < proportion <= 1:
            raise ValueError("proportion must lie in (0, 1]")
        stop = max(2, min(self.num_events, round(self.num_events * proportion)))
        horizon = self.observation_horizon if stop == self.num_events else None
        return EventBatch(
            self.times[:stop],
            self.sizes[:stop],
            self.src[:stop],
            self.dst[:stop],
            self.node_keys,
            horizon,
        )
