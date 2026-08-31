import numpy as np
import torch

from encrypted_topology.simulator import SPARSITY_LEVELS, _block_connectivity, simulate_network


def test_simulator_is_deterministic_with_binary_link_truth():
    first = simulate_network(19, "moderate", "jitter")
    second = simulate_network(19, "moderate", "jitter")
    assert torch.equal(first.batch.src, second.batch.src)
    assert torch.allclose(first.batch.times, second.batch.times)
    assert set(torch.unique(first.truth_adjacency).tolist()) == {0.0, 1.0}


def test_padding_removes_packet_size_signal():
    result = simulate_network(23, "dense", "padding")
    assert torch.unique(result.batch.sizes).tolist() == [1500.0]


def test_dense_formal_seed_has_binary_link_truth():
    result = simulate_network(2026, "dense", "none")
    assert set(torch.unique(result.truth_adjacency).tolist()) == {0.0, 1.0}


def test_all_frozen_seed_connectivity_draws_include_a_missing_block_edge():
    off_diagonal = ~np.eye(4, dtype=bool)
    for seed in range(2026, 2126):
        for sparsity in SPARSITY_LEVELS:
            rng = np.random.default_rng(seed)
            memberships = np.repeat(np.arange(4), 6)
            rng.shuffle(memberships)
            connected = _block_connectivity(rng, 4, sparsity)
            assert not np.all(connected[off_diagonal])
