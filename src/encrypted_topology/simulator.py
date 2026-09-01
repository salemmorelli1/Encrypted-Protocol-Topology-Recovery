"""Controlled-truth simulation registry for topology-recovery research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .events import EventBatch

SPARSITY_LEVELS = ("dense", "moderate", "sparse")
OBFUSCATION_LEVELS = ("none", "padding", "jitter")
GENERATOR_FAMILIES = (
    "hawkes_exponential",
    "hawkes_mixture",
    "cox_piecewise",
    "renewal_gamma",
    "independent_null",
)


@dataclass(frozen=True)
class SimulationFamily:
    """Documented event law used by the robustness experiment."""

    name: str
    temporal_process: str
    truth_signal: bool
    fitted_model_alignment: str
    description: str


SIMULATION_REGISTRY = (
    SimulationFamily(
        "hawkes_exponential",
        "stationary multivariate exponential-kernel Hawkes process",
        True,
        "matched",
        "Primary controlled-truth generator retained for the frozen factorial experiment.",
    ),
    SimulationFamily(
        "hawkes_mixture",
        "stationary multivariate two-scale Hawkes cluster process",
        True,
        "kernel misspecification",
        "Tests robustness when excitation contains both short and long time scales.",
    ),
    SimulationFamily(
        "cox_piecewise",
        "piecewise-constant Cox process with shared latent rate shocks",
        True,
        "nonstationary misspecification",
        "Separates topology-linked rates from exogenous temporal nonstationarity.",
    ),
    SimulationFamily(
        "renewal_gamma",
        "independent gamma-renewal process for each directed block pair",
        True,
        "dependence misspecification",
        "Tests whether non-Poisson renewal timing is mistaken for network excitation.",
    ),
    SimulationFamily(
        "independent_null",
        "independent homogeneous Poisson marks unrelated to the truth graph",
        False,
        "negative control",
        "Checks calibration when the observations contain no recoverable topology signal.",
    ),
)


def simulation_registry() -> list[dict[str, object]]:
    """Return JSON-serializable metadata for every approved synthetic generator."""

    return [
        {
            "name": family.name,
            "temporal_process": family.temporal_process,
            "truth_signal": family.truth_signal,
            "fitted_model_alignment": family.fitted_model_alignment,
            "description": family.description,
        }
        for family in SIMULATION_REGISTRY
    ]


@dataclass(frozen=True)
class SimulatedNetwork:
    batch: EventBatch
    truth_adjacency: torch.Tensor
    memberships: torch.Tensor
    block_connectivity: torch.Tensor
    condition: dict[str, object]


def _block_connectivity(rng: np.random.Generator, blocks: int, sparsity: str) -> np.ndarray:
    probabilities = {"dense": 0.85, "moderate": 0.50, "sparse": 0.25}
    if sparsity not in probabilities:
        raise ValueError(f"unknown sparsity level: {sparsity}")
    # Link AUC requires both positive and negative dyads. Dense Bernoulli draws
    # can occasionally produce a completely connected block graph (including
    # formal seed 2026), so sample from the declared Bernoulli law conditional
    # on at least one absent off-diagonal block edge. Rejection sampling avoids
    # assigning the degenerate draw an arbitrary graph after the fact.
    off_diagonal = ~np.eye(blocks, dtype=bool)
    while True:
        connected = rng.random((blocks, blocks)) < probabilities[sparsity]
        np.fill_diagonal(connected, True)
        if not np.all(connected[off_diagonal]):
            return connected


def _hawkes_cluster_times(
    rng: np.random.Generator,
    baseline: np.ndarray,
    excitation: np.ndarray,
    decay: float,
    horizon: float,
    max_events: int,
) -> list[tuple[float, int]]:
    events: list[tuple[float, int]] = []
    for mark, rate in enumerate(baseline):
        count = rng.poisson(rate * horizon)
        events.extend((float(value), mark) for value in rng.uniform(0.0, horizon, count))
    cursor = 0
    while cursor < len(events) and len(events) < max_events:
        parent_time, parent_mark = events[cursor]
        remaining = horizon - parent_time
        for target_mark, mass in enumerate(excitation[parent_mark]):
            integrated_mass = mass * (1.0 - np.exp(-decay * remaining))
            child_count = rng.poisson(integrated_mass)
            if child_count:
                delays = rng.exponential(1.0 / decay, child_count)
                events.extend(
                    (float(parent_time + delay), target_mark)
                    for delay in delays
                    if delay < remaining
                )
        cursor += 1
    return sorted(events[:max_events])


def _hawkes_mixture_times(
    rng: np.random.Generator,
    baseline: np.ndarray,
    excitation: np.ndarray,
    horizon: float,
    max_events: int,
) -> list[tuple[float, int]]:
    """Simulate a two-scale cluster law outside the fitted single-kernel family."""

    events: list[tuple[float, int]] = []
    for mark, rate in enumerate(baseline):
        count = rng.poisson(rate * horizon)
        events.extend((float(value), mark) for value in rng.uniform(0.0, horizon, count))
    cursor = 0
    while cursor < len(events) and len(events) < max_events:
        parent_time, parent_mark = events[cursor]
        remaining = horizon - parent_time
        for target_mark, mass in enumerate(excitation[parent_mark]):
            child_count = rng.poisson(mass)
            if child_count:
                long_scale = rng.random(child_count) < 0.30
                rates = np.where(long_scale, 0.45, 2.4)
                delays = rng.exponential(1.0 / rates)
                events.extend(
                    (float(parent_time + delay), target_mark)
                    for delay in delays
                    if delay < remaining
                )
        cursor += 1
    return sorted(events[:max_events])


def _cox_piecewise_times(
    rng: np.random.Generator,
    baseline: np.ndarray,
    horizon: float,
    max_events: int,
) -> list[tuple[float, int]]:
    """Simulate shared nonstationarity using piecewise-constant latent rates."""

    events: list[tuple[float, int]] = []
    boundaries = np.linspace(0.0, horizon, 7)
    multipliers = rng.lognormal(mean=-0.18, sigma=0.60, size=6)
    for start, stop, multiplier in zip(boundaries[:-1], boundaries[1:], multipliers, strict=True):
        width = stop - start
        for mark, rate in enumerate(baseline):
            count = rng.poisson(rate * multiplier * width)
            events.extend((float(value), mark) for value in rng.uniform(start, stop, count))
    return sorted(events[:max_events])


def _renewal_gamma_times(
    rng: np.random.Generator,
    baseline: np.ndarray,
    horizon: float,
    max_events: int,
) -> list[tuple[float, int]]:
    """Simulate over-dispersed renewal timing without self-excitation."""

    events: list[tuple[float, int]] = []
    shape = 0.65
    for mark, rate in enumerate(baseline):
        current = float(rng.uniform(0.0, min(horizon, 1.0 / max(rate, 1e-6))))
        scale = 1.0 / (shape * rate)
        while current < horizon and len(events) < max_events:
            events.append((current, mark))
            current += float(rng.gamma(shape, scale))
    return sorted(events[:max_events])


def _generate_times(
    generator: str,
    rng: np.random.Generator,
    baseline: np.ndarray,
    excitation: np.ndarray,
    horizon: float,
    max_events: int,
) -> list[tuple[float, int]]:
    if generator == "hawkes_exponential":
        return _hawkes_cluster_times(rng, baseline, excitation, 1.6, horizon, max_events)
    if generator == "hawkes_mixture":
        return _hawkes_mixture_times(rng, baseline, excitation, horizon, max_events)
    if generator == "cox_piecewise":
        return _cox_piecewise_times(rng, baseline, horizon, max_events)
    if generator == "renewal_gamma":
        return _renewal_gamma_times(rng, baseline, horizon, max_events)
    if generator == "independent_null":
        null_baseline = np.full_like(baseline, 0.35)
        return _hawkes_cluster_times(
            rng, null_baseline, np.zeros_like(excitation), 1.6, horizon, max_events
        )
    raise ValueError(f"unknown generator family: {generator}")


def simulate_network(
    seed: int,
    sparsity: str = "moderate",
    obfuscation: str = "none",
    nodes: int = 24,
    blocks: int = 4,
    horizon: float = 30.0,
    max_events: int = 700,
    generator: str = "hawkes_exponential",
) -> SimulatedNetwork:
    """Generate truth-labelled node links and synthetic marked events under one DOE cell."""

    if obfuscation not in OBFUSCATION_LEVELS:
        raise ValueError(f"unknown obfuscation level: {obfuscation}")
    if generator not in GENERATOR_FAMILIES:
        raise ValueError(f"unknown generator family: {generator}")
    if blocks < 2:
        raise ValueError("at least two blocks are required for binary link evaluation")
    if nodes < 2 * blocks or nodes % blocks:
        raise ValueError("nodes must be divisible by blocks with at least two nodes per block")
    rng = np.random.default_rng(seed)
    memberships = np.repeat(np.arange(blocks), nodes // blocks)
    rng.shuffle(memberships)
    connected = _block_connectivity(rng, blocks, sparsity)
    marks = blocks * blocks
    baseline_scale = {"dense": 0.65, "moderate": 0.42, "sparse": 0.28}[sparsity]
    baseline = np.where(connected.reshape(-1), baseline_scale, 0.025)
    excitation = np.eye(marks) * 0.20
    for source in range(marks):
        source_row, source_column = divmod(source, blocks)
        for target in range(marks):
            target_row, target_column = divmod(target, blocks)
            if source != target and (source_row == target_row or source_column == target_column):
                excitation[source, target] = 0.012
    events = _generate_times(generator, rng, baseline, excitation, horizon, max_events)
    if len(events) < 40:
        raise RuntimeError("simulator generated too few events; increase the horizon")

    times: list[float] = []
    sizes: list[float] = []
    sources: list[int] = []
    destinations: list[int] = []
    members = [np.flatnonzero(memberships == block) for block in range(blocks)]
    for event_time, mark in events:
        source_block, destination_block = divmod(mark, blocks)
        source = int(rng.choice(members[source_block]))
        destination = int(rng.choice(members[destination_block]))
        if source == destination:
            alternatives = members[destination_block][members[destination_block] != source]
            destination = int(rng.choice(alternatives))
        location = 6.15 + 0.18 * source_block - 0.12 * destination_block
        packet_size = float(np.clip(np.exp(rng.normal(location, 0.45)), 64, 1500))
        times.append(event_time)
        sizes.append(packet_size)
        sources.append(source)
        destinations.append(destination)

    if obfuscation == "padding":
        sizes = [1500.0] * len(sizes)
        dummy_count = max(1, len(times) // 5)
        for _ in range(dummy_count):
            source, destination = rng.choice(nodes, size=2, replace=False)
            times.append(float(rng.uniform(0.0, horizon)))
            sizes.append(1500.0)
            sources.append(int(source))
            destinations.append(int(destination))
    elif obfuscation == "jitter":
        jitter = rng.normal(0.0, 0.20, len(times))
        times = np.clip(np.asarray(times) + jitter, 0.0, horizon).tolist()

    order = np.argsort(times, kind="stable")
    ordered_times = np.asarray(times)[order]
    ordered_times = ordered_times - ordered_times[0]
    batch = EventBatch(
        torch.tensor(ordered_times, dtype=torch.float64),
        torch.tensor(np.asarray(sizes)[order], dtype=torch.float32),
        torch.tensor(np.asarray(sources)[order], dtype=torch.long),
        torch.tensor(np.asarray(destinations)[order], dtype=torch.long),
        tuple(f"synthetic-node-{index:03d}" for index in range(nodes)),
    )
    truth = connected[memberships[:, None], memberships[None, :]].astype(np.float32)
    np.fill_diagonal(truth, 0.0)
    return SimulatedNetwork(
        batch=batch,
        truth_adjacency=torch.tensor(truth),
        memberships=torch.tensor(memberships, dtype=torch.long),
        block_connectivity=torch.tensor(connected),
        condition={
            "seed": seed,
            "sparsity": sparsity,
            "obfuscation": obfuscation,
            "generator": generator,
            "nodes": nodes,
            "blocks": blocks,
            "horizon": horizon,
            "events": batch.num_events,
        },
    )
