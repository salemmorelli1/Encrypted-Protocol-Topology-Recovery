import numpy as np
import torch

from encrypted_topology.simulator import (
    GENERATOR_FAMILIES,
    SPARSITY_LEVELS,
    _block_connectivity,
    _hawkes_cluster_times,
    simulate_network,
    simulation_registry,
)


def test_simulator_is_deterministic_with_binary_link_truth():
    first = simulate_network(19, "moderate", "jitter")
    second = simulate_network(19, "moderate", "jitter")
    assert torch.equal(first.batch.src, second.batch.src)
    assert torch.allclose(first.batch.times, second.batch.times)
    assert set(torch.unique(first.truth_adjacency).tolist()) == {0.0, 1.0}


def test_explicit_default_generator_preserves_frozen_simulator_path():
    implicit = simulate_network(2026, "moderate", "none")
    explicit = simulate_network(2026, "moderate", "none", generator="hawkes_exponential")
    assert torch.equal(implicit.batch.times, explicit.batch.times)
    assert torch.equal(implicit.batch.sizes, explicit.batch.sizes)
    assert torch.equal(implicit.batch.src, explicit.batch.src)
    assert torch.equal(implicit.batch.dst, explicit.batch.dst)
    assert implicit.batch.horizon == 30.0
    assert float(implicit.batch.times[0]) > 0.0


def test_every_registered_generator_is_deterministic_and_has_binary_truth():
    for generator in GENERATOR_FAMILIES:
        first = simulate_network(31, generator=generator)
        second = simulate_network(31, generator=generator)
        assert torch.equal(first.batch.times, second.batch.times)
        assert torch.equal(first.batch.src, second.batch.src)
        assert set(torch.unique(first.truth_adjacency).tolist()) == {0.0, 1.0}
        assert first.condition["generator"] == generator


def test_registry_declares_one_no_signal_negative_control():
    records = simulation_registry()
    assert [record["name"] for record in records] == list(GENERATOR_FAMILIES)
    null_records = [record for record in records if not record["truth_signal"]]
    assert [record["name"] for record in null_records] == ["independent_null"]


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


def test_hawkes_offspring_mass_is_not_window_truncated_twice():
    class StubRng:
        def __init__(self):
            self.poisson_rates = []

        def poisson(self, rate):
            self.poisson_rates.append(float(rate))
            return 1

        @staticmethod
        def uniform(start, stop, count):
            assert (start, stop, count) == (0.0, 1.0, 1)
            return np.asarray([0.9])

        @staticmethod
        def exponential(scale, count):
            assert count == 1
            return np.asarray([0.05])

    rng = StubRng()
    events = _hawkes_cluster_times(
        rng,
        baseline=np.asarray([0.1]),
        excitation=np.asarray([[0.8]]),
        decay=1.0,
        horizon=1.0,
        max_events=2,
    )
    assert rng.poisson_rates == [0.1, 0.8]
    assert events == [(0.9, 0), (0.9500000000000001, 0)]


def test_known_sparse_cox_seeds_meet_the_registered_event_contract():
    for seed in (2035, 2048, 2053):
        result = simulate_network(seed, "sparse", "none", generator="cox_piecewise")
        assert result.batch.num_events >= 40
