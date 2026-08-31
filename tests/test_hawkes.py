import math

import torch

from encrypted_topology.hawkes import (
    StableHawkesParameters,
    exponential_hawkes_log_likelihood,
)


def test_poisson_special_case_includes_exact_compensator():
    times = torch.tensor([0.2, 0.8, 1.1], dtype=torch.float64)
    marks = torch.tensor([0, 1, 0])
    baseline = torch.tensor([0.5, 0.25], dtype=torch.float64)
    excitation = torch.zeros(2, 2, dtype=torch.float64)
    decay = torch.ones(2, 2, dtype=torch.float64)
    value = exponential_hawkes_log_likelihood(
        times, marks, baseline, excitation, decay, horizon=1.5
    )
    expected = 2.0 * math.log(0.5) + math.log(0.25) - 1.5 * 0.75
    assert torch.allclose(value, torch.tensor(expected, dtype=torch.float64), atol=1e-12)


def test_hawkes_score_is_finite_and_differentiable():
    parameters = StableHawkesParameters(2).double()
    baseline, excitation, decay = parameters.constrained()
    value = exponential_hawkes_log_likelihood(
        torch.tensor([0.1, 0.4, 0.9], dtype=torch.float64),
        torch.tensor([0, 1, 0]),
        baseline,
        excitation,
        decay,
        1.2,
    )
    value.backward()
    assert torch.isfinite(value)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters.parameters())


def test_branching_mass_is_subcritical_by_spectral_norm():
    parameters = StableHawkesParameters(6)
    _, excitation, _ = parameters.constrained()
    assert float(torch.linalg.matrix_norm(excitation.detach(), ord=2)) <= 0.85001
