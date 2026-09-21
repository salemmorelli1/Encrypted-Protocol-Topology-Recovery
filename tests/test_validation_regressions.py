import numpy as np
import pytest
import torch

from encrypted_topology.diagnostics import (
    binary_log_score,
    importance_ess,
    rank_normalized_folded_rhat,
)
from encrypted_topology.events import EventBatch
from encrypted_topology.hawkes import exponential_hawkes_log_likelihood
from encrypted_topology.simulator import simulate_network


def _batch(**overrides):
    values = {
        "times": torch.tensor([0.1, 0.4]),
        "sizes": torch.tensor([64.0, 96.0]),
        "src": torch.tensor([0, 1]),
        "dst": torch.tensor([1, 0]),
        "node_keys": ("synthetic-0", "synthetic-1"),
        "observation_horizon": 1.0,
    }
    values.update(overrides)
    return EventBatch(**values)


def test_event_batch_rejects_nonfinite_values_and_fractional_indices():
    with pytest.raises(ValueError, match="finite"):
        _batch(times=torch.tensor([0.1, float("nan")]))
    with pytest.raises(TypeError, match="integer"):
        _batch(src=torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="later than"):
        _batch(observation_horizon=0.4)


def test_hawkes_rejects_nonfinite_and_fractional_inputs():
    baseline = torch.tensor([0.5], dtype=torch.float64)
    excitation = torch.zeros(1, 1, dtype=torch.float64)
    decay = torch.ones(1, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="finite"):
        exponential_hawkes_log_likelihood(
            torch.tensor([0.1, float("nan")], dtype=torch.float64),
            torch.tensor([0, 0]),
            baseline,
            excitation,
            decay,
            1.0,
        )
    with pytest.raises(TypeError, match="integer"):
        exponential_hawkes_log_likelihood(
            torch.tensor([0.1, 0.2], dtype=torch.float64),
            torch.tensor([0.0, 0.0]),
            baseline,
            excitation,
            decay,
            1.0,
        )


def test_diagnostics_fail_closed_on_invalid_numeric_inputs():
    with pytest.raises(ValueError, match="finite"):
        importance_ess(torch.tensor([0.0, float("nan")]))
    with pytest.raises(ValueError, match="finite"):
        rank_normalized_folded_rhat(np.full((2, 8), np.nan))
    with pytest.raises(ValueError, match="binary"):
        binary_log_score(np.asarray([0.0, 2.0]), np.asarray([0.4, 0.6]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        binary_log_score(np.asarray([0.0, 1.0]), np.asarray([0.4, 1.1]))


def test_simulator_rejects_invalid_seed_and_window_controls():
    with pytest.raises(ValueError, match="seed"):
        simulate_network(-1)
    with pytest.raises(ValueError, match="horizon"):
        simulate_network(1, horizon=float("inf"))
    with pytest.raises(ValueError, match="max_events"):
        simulate_network(1, max_events=39)
