import math

import torch

from encrypted_topology.baseline import fit_static_baseline
from encrypted_topology.model import HawkesFlowDSBM
from encrypted_topology.simulator import simulate_network


def test_dynamic_and_static_smoke_paths_are_finite():
    simulated = simulate_network(5, "sparse", "none", horizon=30.0, max_events=180)
    dynamic = HawkesFlowDSBM(blocks=4, latent_dim=4, hidden=16, flow_layers=1)
    fit = dynamic.fit_batch(simulated.batch, epochs=2, seed=5)
    scores = dynamic.link_scores(simulated.batch, samples=2)
    static = fit_static_baseline(simulated.batch, epochs=2, seed=5)
    assert math.isfinite(fit.final_elbo_per_event)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(static.scores).all()
    assert scores.shape == simulated.truth_adjacency.shape
