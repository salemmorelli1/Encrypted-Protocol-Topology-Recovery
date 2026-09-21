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
    if not times.is_floating_point():
        raise TypeError("times must use a floating-point tensor")
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if marks.dtype not in integer_dtypes:
        raise TypeError("marks must use an integer tensor")
    if baseline.ndim != 1:
        raise ValueError("baseline must be a vector")
    if not all(value.is_floating_point() for value in (baseline, excitation_mass, decay)):
        raise TypeError("Hawkes parameters must use floating-point tensors")
    m = baseline.numel()
    if excitation_mass.shape != (m, m) or decay.shape != (m, m):
        raise ValueError("Hawkes matrices must be square with one row per mark")
    if times.numel() == 0:
        raise ValueError("at least one event is required")
    if not all(torch.isfinite(value).all() for value in (times, baseline, excitation_mass, decay)):
        raise ValueError("Hawkes inputs and parameters must be finite")
    if (
        float(times[0]) < 0
        or torch.any(times[1:] < times[:-1])
        or int(marks.min()) < 0
        or int(marks.max()) >= m
    ):
        raise ValueError("times must be ordered and marks in range")
    if torch.any(baseline <= 0) or torch.any(excitation_mass < 0) or torch.any(decay <= 0):
        raise ValueError("invalid Hawkes parameters")

    # Preserve the more precise of the timestamp and parameter dtypes. Otherwise
    # a valid float64 event just below T can round up to T when float32 model
    # parameters are used, spuriously collapsing the terminal exposure.
    dtype = torch.promote_types(times.dtype, baseline.dtype)
    device = baseline.device
    t = times.to(device=device, dtype=dtype)
    z = marks.to(device=device, dtype=torch.long)
    baseline_values = baseline.to(dtype=dtype)
    excitation_values = excitation_mass.to(dtype=dtype)
    decay_values = decay.to(dtype=dtype)
    end = torch.as_tensor(horizon, device=device, dtype=dtype)
    if end.numel() != 1 or not torch.isfinite(end):
        raise ValueError("horizon must be a finite scalar")
    if end <= t[-1]:
        raise ValueError("horizon must be strictly later than the final event")

    log_event = torch.zeros((), device=device, dtype=dtype)
    tiny = torch.finfo(dtype).tiny
    for i in range(t.numel()):
        target = z[i]
        intensity = baseline_values[target]
        if i:
            elapsed = t[i] - t[:i]
            strictly_earlier = elapsed > 0
            source = z[:i][strictly_earlier]
            beta_values = decay_values[source, target]
            elapsed = elapsed[strictly_earlier]
            intensity = intensity + torch.sum(
                excitation_values[source, target]
                * beta_values
                * torch.exp(-beta_values * elapsed)
            )
        log_event = log_event + torch.log(torch.clamp_min(intensity, tiny))

    base_compensator = end * torch.sum(baseline_values)
    remaining = end - t
    source_rows = excitation_values[z]
    decay_rows = decay_values[z]
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
