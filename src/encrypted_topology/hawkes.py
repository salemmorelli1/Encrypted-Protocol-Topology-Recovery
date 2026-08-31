"""Exact exponential-kernel multivariate Hawkes likelihoods."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def exponential_hawkes_log_likelihood(
    times: torch.Tensor,
    marks: torch.Tensor,
    baseline: torch.Tensor,
    excitation_mass: torch.Tensor,
    decay: torch.Tensor,
    horizon: float | torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate the marked point-process log likelihood on ``[0, horizon]``.

    The kernel from mark ``r`` to mark ``s`` is
    ``alpha[r,s] * beta[r,s] * exp(-beta[r,s] * u)``. Consequently,
    ``alpha`` is the integrated branching-mass matrix and the compensator is
    available in closed form.
    """

    if times.ndim != 1 or marks.ndim != 1 or times.numel() != marks.numel():
        raise ValueError("times and marks must be aligned vectors")
    if baseline.ndim != 1:
        raise ValueError("baseline must be a vector")
    m = baseline.numel()
    if excitation_mass.shape != (m, m) or decay.shape != (m, m):
        raise ValueError("Hawkes matrices must be square with one row per mark")
    if times.numel() == 0:
        raise ValueError("at least one event is required")
    if torch.any(times[1:] < times[:-1]) or int(marks.min()) < 0 or int(marks.max()) >= m:
        raise ValueError("times must be ordered and marks in range")
    if torch.any(baseline <= 0) or torch.any(excitation_mass < 0) or torch.any(decay <= 0):
        raise ValueError("invalid Hawkes parameters")

    dtype, device = baseline.dtype, baseline.device
    t = times.to(device=device, dtype=dtype)
    z = marks.to(device=device, dtype=torch.long)
    end = torch.as_tensor(horizon, device=device, dtype=dtype)
    if end <= t[-1]:
        raise ValueError("horizon must be strictly later than the final event")

    log_event = torch.zeros((), device=device, dtype=dtype)
    tiny = torch.finfo(dtype).tiny
    for i in range(t.numel()):
        target = z[i]
        intensity = baseline[target]
        if i:
            source = z[:i]
            beta_values = decay[source, target]
            elapsed = t[i] - t[:i]
            intensity = intensity + torch.sum(
                excitation_mass[source, target]
                * beta_values
                * torch.exp(-beta_values * elapsed)
            )
        log_event = log_event + torch.log(torch.clamp_min(intensity, tiny))

    base_compensator = end * torch.sum(baseline)
    remaining = end - t
    source_rows = excitation_mass[z]
    decay_rows = decay[z]
    excitation_compensator = torch.sum(
        source_rows * (1.0 - torch.exp(-decay_rows * remaining[:, None]))
    )
    return log_event - base_compensator - excitation_compensator


class StableHawkesParameters(nn.Module):
    """Positive Hawkes parameters with spectral-norm subcriticality control."""

    def __init__(self, marks: int, stability_margin: float = 0.85) -> None:
        super().__init__()
        if marks < 1 or not 0 < stability_margin < 1:
            raise ValueError("invalid Hawkes parameter dimensions")
        self.marks = marks
        self.stability_margin = stability_margin
        self.raw_baseline = nn.Parameter(torch.full((marks,), -2.5))
        self.raw_excitation = nn.Parameter(torch.full((marks, marks), -4.0))
        self.raw_decay = nn.Parameter(torch.full((marks, marks), 0.5))

    def constrained(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        baseline = F.softplus(self.raw_baseline) + torch.finfo(self.raw_baseline.dtype).eps
        excitation = F.softplus(self.raw_excitation)
        spectral_norm = torch.linalg.matrix_norm(excitation, ord=2)
        scale = torch.clamp(spectral_norm / self.stability_margin, min=1.0)
        excitation = excitation / scale
        decay = F.softplus(self.raw_decay) + torch.finfo(self.raw_decay.dtype).eps
        return baseline, excitation, decay

