"""Typed synthetic event tensors used by the simulation laboratory."""

from __future__ import annotations

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

    def __post_init__(self) -> None:
        n = int(self.times.numel())
        if n < 2 or any(int(x.numel()) != n for x in (self.sizes, self.src, self.dst)):
            raise ValueError("an event batch requires aligned vectors and at least two events")
        if not torch.all(self.times[1:] >= self.times[:-1]):
            raise ValueError("event times must be nondecreasing")
        if float(self.times[0]) < 0 or torch.any(self.sizes <= 0):
            raise ValueError("times must be nonnegative and packet sizes positive")
        if torch.any(self.src == self.dst):
            raise ValueError("self-links are excluded from this experiment")
        if min(int(self.src.min()), int(self.dst.min())) < 0:
            raise ValueError("node indices must be nonnegative")
        if max(int(self.src.max()), int(self.dst.max())) >= len(self.node_keys):
            raise ValueError("node index exceeds node dictionary")

    @property
    def num_events(self) -> int:
        return int(self.times.numel())

    @property
    def num_nodes(self) -> int:
        return len(self.node_keys)

    @property
    def horizon(self) -> float:
        return float(self.times[-1])

    def to(self, device: torch.device | str) -> EventBatch:
        return EventBatch(
            self.times.to(device),
            self.sizes.to(device),
            self.src.to(device),
            self.dst.to(device),
            self.node_keys,
        )

    def prefix(self, proportion: float) -> EventBatch:
        if not 0 < proportion <= 1:
            raise ValueError("proportion must lie in (0, 1]")
        stop = max(2, min(self.num_events, round(self.num_events * proportion)))
        return EventBatch(
            self.times[:stop], self.sizes[:stop], self.src[:stop], self.dst[:stop], self.node_keys
        )
