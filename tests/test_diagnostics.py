import numpy as np
import pytest
import torch

from encrypted_topology.diagnostics import importance_ess, rank_normalized_folded_rhat


def test_equal_importance_weights_have_maximal_ess():
    assert importance_ess(torch.zeros(20)) == pytest.approx(20.0)


def test_rhat_is_reserved_for_actual_chain_arrays():
    rng = np.random.default_rng(42)
    chains = rng.normal(size=(4, 200))
    assert rank_normalized_folded_rhat(chains) < 1.05
    with pytest.raises(ValueError):
        rank_normalized_folded_rhat(np.ones((1, 100)))

