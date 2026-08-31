"""Restartable 2 x 3 x 3 controlled-truth factorial experiment."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from scipy.stats import t as student_t
from sklearn.metrics import roc_auc_score
from statsmodels.regression.mixed_linear_model import MixedLM

from .baseline import fit_static_baseline
from .diagnostics import binary_log_score, importance_ess
from .model import HawkesFlowDSBM
from .simulator import OBFUSCATION_LEVELS, SPARSITY_LEVELS, simulate_network

ARCHITECTURES = ("hawkes_flow_dsbm", "static_gae_louvain")
BASE_SEED = 2026
FIELDS = (
    "seed",
    "architecture",
    "obfuscation",
    "sparsity",
    "events",
    "link_auc",
    "link_log_score",
    "latency_ms_per_event",
    "occupied_communities",
    "importance_ess_fraction",
    "status",
)


@dataclass(frozen=True)
class CellResult:
    seed: int
    architecture: str
    obfuscation: str
    sparsity: str
    events: int
    link_auc: float
    link_log_score: float
    latency_ms_per_event: float
    occupied_communities: int
    importance_ess_fraction: float
    status: str = "complete"


def _off_diagonal(matrix: torch.Tensor) -> np.ndarray:
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask].detach().cpu().numpy()


def _dynamic_importance_ess(model: HawkesFlowDSBM, batch, draws: int = 24) -> float:
    weights = []
    with torch.no_grad():
        for _ in range(draws):
            _, _, log_q, log_p = model.posterior(batch, temperature=0.5)
            weights.append(torch.sum(log_p - log_q))
    return importance_ess(torch.stack(weights)) / draws


def run_cell(
    seed: int,
    architecture: str,
    obfuscation: str,
    sparsity: str,
    epochs: int = 60,
    device: str | torch.device = "cpu",
) -> CellResult:
    simulated = simulate_network(seed, sparsity=sparsity, obfuscation=obfuscation)
    batch = simulated.batch.to(device)
    labels = _off_diagonal(simulated.truth_adjacency.to(device))
    if np.unique(labels).size != 2:
        raise RuntimeError("truth adjacency lacks both link classes")
    if architecture == "hawkes_flow_dsbm":
        model = HawkesFlowDSBM().to(device)
        fit = model.fit_batch(batch, epochs=epochs, seed=seed)
        raw_scores = model.link_scores(batch)
        probabilities = torch.sigmoid(
            (raw_scores - torch.median(raw_scores)) / torch.clamp_min(torch.std(raw_scores), 1e-4)
        )
        latency = fit.latency_ms
        occupied = fit.occupied_blocks
        ess_fraction = _dynamic_importance_ess(model, batch)
    elif architecture == "static_gae_louvain":
        fit = fit_static_baseline(batch, epochs=epochs, seed=seed)
        probabilities = fit.scores
        latency = fit.latency_ms
        occupied = fit.occupied_communities
        ess_fraction = float("nan")
    else:
        raise ValueError(f"unknown architecture: {architecture}")
    scores = _off_diagonal(probabilities)
    return CellResult(
        seed=seed,
        architecture=architecture,
        obfuscation=obfuscation,
        sparsity=sparsity,
        events=batch.num_events,
        link_auc=float(roc_auc_score(labels, scores)),
        link_log_score=binary_log_score(labels, scores),
        latency_ms_per_event=latency / batch.num_events,
        occupied_communities=occupied,
        importance_ess_fraction=ess_fraction,
    )


def _read_completed(path: Path) -> dict[tuple[int, str, str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        (int(row["seed"]), row["architecture"], row["obfuscation"], row["sparsity"]): row
        for row in rows
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_factorial(
    output: Path = Path("data/results/factorial_results.csv"),
    seeds: int = 100,
    epochs: int = 60,
    device: str | torch.device | None = None,
) -> int:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    completed = _read_completed(output)
    rows: list[dict[str, object]] = list(completed.values())
    design = [
        (seed, architecture, obfuscation, sparsity)
        for seed in range(BASE_SEED, BASE_SEED + seeds)
        for obfuscation in OBFUSCATION_LEVELS
        for sparsity in SPARSITY_LEVELS
        for architecture in ARCHITECTURES
    ]
    rng = np.random.default_rng(2026)
    rng.shuffle(design)
    for cell in design:
        if cell in completed:
            continue
        result = run_cell(*cell, epochs=epochs, device=selected_device)
        row = asdict(result)
        rows.append(row)
        completed[cell] = {key: str(value) for key, value in row.items()}
        rows.sort(key=lambda value: (
            int(value["seed"]), str(value["obfuscation"]), str(value["sparsity"]), str(value["architecture"])
        ))
        _write_rows(output, rows)
    return len(rows)


def factorial_contrast_matrix(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Construct the full-rank orthogonal 2 x 3 x 3 fixed and random designs."""

    architecture = np.asarray([
        0.5 if row["architecture"] == "hawkes_flow_dsbm" else -0.5 for row in rows
    ])
    linear = np.asarray([-1.0, 0.0, 1.0]) / math.sqrt(2.0)
    quadratic = np.asarray([1.0, -2.0, 1.0]) / math.sqrt(6.0)
    obfuscation_index = {level: index for index, level in enumerate(OBFUSCATION_LEVELS)}
    sparsity_index = {level: index for index, level in enumerate(SPARSITY_LEVELS)}
    ob_l = np.asarray([linear[obfuscation_index[row["obfuscation"]]] for row in rows])
    ob_q = np.asarray([quadratic[obfuscation_index[row["obfuscation"]]] for row in rows])
    sp_l = np.asarray([linear[sparsity_index[row["sparsity"]]] for row in rows])
    sp_q = np.asarray([quadratic[sparsity_index[row["sparsity"]]] for row in rows])
    base = {
        "architecture": architecture,
        "obfuscation_linear": ob_l,
        "obfuscation_quadratic": ob_q,
        "sparsity_linear": sp_l,
        "sparsity_quadratic": sp_q,
    }
    columns: list[tuple[str, np.ndarray]] = [("intercept", np.ones(len(rows)))]
    columns.extend(base.items())
    for ob_name in ("obfuscation_linear", "obfuscation_quadratic"):
        columns.append((f"architecture:{ob_name}", architecture * base[ob_name]))
    for sp_name in ("sparsity_linear", "sparsity_quadratic"):
        columns.append((f"architecture:{sp_name}", architecture * base[sp_name]))
    for ob_name in ("obfuscation_linear", "obfuscation_quadratic"):
        for sp_name in ("sparsity_linear", "sparsity_quadratic"):
            interaction = base[ob_name] * base[sp_name]
            columns.append((f"{ob_name}:{sp_name}", interaction))
            columns.append((f"architecture:{ob_name}:{sp_name}", architecture * interaction))
    names = [name for name, _ in columns]
    fixed = np.column_stack([column for _, column in columns])
    random = np.column_stack([np.ones(len(rows)), architecture])
    groups = np.asarray([int(row["seed"]) for row in rows])
    if fixed.shape[1] != 18 or np.linalg.matrix_rank(fixed) != 18:
        raise RuntimeError("factorial contrast matrix is not full rank")
    return fixed, names, random, groups


def _fit_mixed_model(rows: list[dict[str, str]], endpoint: str) -> dict[str, object]:
    fixed, names, random, groups = factorial_contrast_matrix(rows)
    response = np.asarray([float(row[endpoint]) for row in rows])
    transformed = endpoint == "latency_ms_per_event"
    if transformed:
        response = np.log(np.clip(response, np.finfo(float).tiny, None))
    model = MixedLM(response, fixed, groups=groups, exog_re=random)
    fit = model.fit(reml=True, method="lbfgs", maxiter=500, disp=False)
    if not fit.converged:
        fallback = MixedLM(response, fixed, groups=groups, exog_re=random[:, :1])
        fit = fallback.fit(reml=True, method="lbfgs", maxiter=500, disp=False)
        random_structure = "seed_random_intercept"
    else:
        random_structure = "seed_random_intercept_and_architecture_slope"
    estimates = np.asarray(fit.fe_params)
    standard_errors = np.asarray(fit.bse_fe)
    coefficients = []
    for name, estimate, standard_error in zip(names, estimates, standard_errors, strict=True):
        z_value = estimate / standard_error
        coefficients.append({
            "term": name,
            "estimate": float(estimate),
            "standard_error": float(standard_error),
            "lower_95": float(estimate - 1.96 * standard_error),
            "upper_95": float(estimate + 1.96 * standard_error),
            "z_value": float(z_value),
            "p_value_two_sided": float(2.0 * norm.sf(abs(z_value))),
        })
    return {
        "endpoint": endpoint,
        "response_scale": "natural_log" if transformed else "identity",
        "random_structure": random_structure,
        "converged": bool(fit.converged),
        "coefficients": coefficients,
    }


def analyze_factorial(
    results_path: Path = Path("data/results/factorial_results.csv"),
    public_summary: Path = Path("data/empirical_summary.json"),
    expected_seeds: int = 100,
) -> dict[str, object]:
    with results_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = expected_seeds * len(ARCHITECTURES) * len(OBFUSCATION_LEVELS) * len(SPARSITY_LEVELS)
    keys = {
        (row["seed"], row["architecture"], row["obfuscation"], row["sparsity"])
        for row in rows
    }
    if len(rows) != expected or len(keys) != expected or any(row["status"] != "complete" for row in rows):
        raise RuntimeError(f"analysis remains locked: expected {expected} unique completed cells")
    endpoints = ("link_auc", "link_log_score", "latency_ms_per_event")
    contrasts: list[dict[str, object]] = []
    for obfuscation in OBFUSCATION_LEVELS:
        for sparsity in SPARSITY_LEVELS:
            selected = [row for row in rows if row["obfuscation"] == obfuscation and row["sparsity"] == sparsity]
            by_seed = {}
            for row in selected:
                by_seed.setdefault(int(row["seed"]), {})[row["architecture"]] = row
            for endpoint in endpoints:
                differences = np.asarray([
                    float(pair["hawkes_flow_dsbm"][endpoint])
                    - float(pair["static_gae_louvain"][endpoint])
                    for pair in by_seed.values()
                ])
                mean = float(differences.mean())
                standard_error = float(differences.std(ddof=1) / math.sqrt(differences.size))
                critical = float(student_t.ppf(0.975, differences.size - 1))
                contrasts.append({
                    "endpoint": endpoint,
                    "obfuscation": obfuscation,
                    "sparsity": sparsity,
                    "estimate_hawkes_minus_static": mean,
                    "lower_95": mean - critical * standard_error,
                    "upper_95": mean + critical * standard_error,
                    "paired_seeds": int(differences.size),
                })
    summary = {
        "schema": "encrypted-topology-empirical-summary-v1",
        "status": "complete_controlled_truth_factorial",
        "rows": len(rows),
        "seeds": expected_seeds,
        "contrasts": contrasts,
        "mixed_effects_models": [_fit_mixed_model(rows, endpoint) for endpoint in endpoints],
        "claim_boundary": (
            "Controlled-truth topology recovery and authorized local metadata feasibility only; "
            "not attribution, payload recovery, C2 identification, or operational SIGINT validation."
        ),
    }
    public_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
