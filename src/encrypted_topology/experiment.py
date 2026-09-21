"""Restartable controlled-truth factorial and robustness experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import norm
from scipy.stats import t as student_t
from sklearn.metrics import roc_auc_score
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from .baseline import fit_static_baseline
from .diagnostics import binary_log_score, importance_ess
from .model import HawkesFlowDSBM
from .simulator import (
    GENERATOR_FAMILIES,
    OBFUSCATION_LEVELS,
    SPARSITY_LEVELS,
    simulate_network,
    simulation_registry,
)

ARCHITECTURES = ("hawkes_flow_dsbm", "static_gae_louvain")
BASE_SEED = 2026
ANALYSIS_ENGINE_VERSION = "1.2.0"
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
ROBUSTNESS_FIELDS = ("generator",) + FIELDS
FactorialKey = tuple[int, str, str, str]
RobustnessKey = tuple[int, str, str, str, str]


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
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("off-diagonal extraction requires a square matrix")
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask].detach().cpu().numpy()


def _dynamic_importance_ess(model: HawkesFlowDSBM, batch, draws: int = 24) -> float:
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
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
    generator: str = "hawkes_exponential",
) -> CellResult:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture: {architecture}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    simulated = simulate_network(
        seed, sparsity=sparsity, obfuscation=obfuscation, generator=generator
    )
    batch = simulated.batch.to(device)
    labels = _off_diagonal(simulated.truth_adjacency.to(device))
    if np.unique(labels).size != 2:
        raise RuntimeError("truth adjacency lacks both link classes")
    if architecture == "hawkes_flow_dsbm":
        # Seed before construction so parameter initialization is part of the
        # declared, repeatable cell seed—not only the stochastic fit loop.
        torch.manual_seed(seed)
        model = HawkesFlowDSBM().to(device)
        dynamic_fit = model.fit_batch(batch, epochs=epochs, seed=seed)
        raw_scores = model.link_scores(batch)
        probabilities = torch.sigmoid(
            (raw_scores - torch.median(raw_scores)) / torch.clamp_min(torch.std(raw_scores), 1e-4)
        )
        latency = dynamic_fit.latency_ms
        occupied = dynamic_fit.occupied_blocks
        ess_fraction = _dynamic_importance_ess(model, batch)
    elif architecture == "static_gae_louvain":
        static_fit = fit_static_baseline(batch, epochs=epochs, seed=seed)
        probabilities = static_fit.scores
        latency = static_fit.latency_ms
        occupied = static_fit.occupied_communities
        ess_fraction = float("nan")
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


def _positive_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _factorial_design_keys(seeds: int) -> set[FactorialKey]:
    _positive_count(seeds, "seeds")
    return {
        (seed, architecture, obfuscation, sparsity)
        for seed in range(BASE_SEED, BASE_SEED + seeds)
        for architecture in ARCHITECTURES
        for obfuscation in OBFUSCATION_LEVELS
        for sparsity in SPARSITY_LEVELS
    }


def _validated_generators(generators: tuple[str, ...]) -> tuple[str, ...]:
    if not generators or len(set(generators)) != len(generators):
        raise ValueError("generator families must be a nonempty unique sequence")
    unknown = set(generators) - set(GENERATOR_FAMILIES)
    if unknown:
        raise ValueError(f"unknown generator families: {sorted(unknown)}")
    return generators


def _robustness_design_keys(
    seeds: int, generators: tuple[str, ...]
) -> set[RobustnessKey]:
    _positive_count(seeds, "seeds")
    _validated_generators(generators)
    return {
        (seed, generator, architecture, obfuscation, sparsity)
        for seed in range(BASE_SEED, BASE_SEED + seeds)
        for generator in generators
        for architecture in ARCHITECTURES
        for obfuscation in OBFUSCATION_LEVELS
        for sparsity in SPARSITY_LEVELS
    }


def _parse_int(value: str, field: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"row {row_number}: {field} must be an integer") from error
    return parsed


def _parse_finite(value: str, field: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"row {row_number}: {field} must be numeric") from error
    if not math.isfinite(parsed):
        raise RuntimeError(f"row {row_number}: {field} must be finite")
    return parsed


def _validate_result_row(
    row: dict[str, str], row_number: int, *, robustness: bool
) -> FactorialKey | RobustnessKey:
    seed = _parse_int(row["seed"], "seed", row_number)
    events = _parse_int(row["events"], "events", row_number)
    occupied = _parse_int(row["occupied_communities"], "occupied_communities", row_number)
    architecture = row["architecture"]
    obfuscation = row["obfuscation"]
    sparsity = row["sparsity"]
    if architecture not in ARCHITECTURES:
        raise RuntimeError(f"row {row_number}: unknown architecture")
    if obfuscation not in OBFUSCATION_LEVELS or sparsity not in SPARSITY_LEVELS:
        raise RuntimeError(f"row {row_number}: unknown design level")
    if events < 40 or occupied < 1:
        raise RuntimeError(f"row {row_number}: invalid event or community count")
    auc = _parse_finite(row["link_auc"], "link_auc", row_number)
    log_score = _parse_finite(row["link_log_score"], "link_log_score", row_number)
    latency = _parse_finite(row["latency_ms_per_event"], "latency_ms_per_event", row_number)
    if not 0.0 <= auc <= 1.0 or log_score > 0.0 or latency < 0.0:
        raise RuntimeError(f"row {row_number}: endpoint outside its valid range")
    ess_text = row["importance_ess_fraction"]
    if architecture == "hawkes_flow_dsbm":
        ess = _parse_finite(ess_text, "importance_ess_fraction", row_number)
        if not 0.0 < ess <= 1.0:
            raise RuntimeError(f"row {row_number}: importance ESS fraction outside (0, 1]")
    elif ess_text.strip().lower() != "nan":
        raise RuntimeError(f"row {row_number}: static ESS must be nan")
    if row["status"] != "complete":
        raise RuntimeError(f"row {row_number}: status must be complete")
    base: FactorialKey = (seed, architecture, obfuscation, sparsity)
    if not robustness:
        return base
    generator = row["generator"]
    if generator not in GENERATOR_FAMILIES:
        raise RuntimeError(f"row {row_number}: unknown generator")
    return (seed, generator, architecture, obfuscation, sparsity)


def _read_validated_rows(
    path: Path, *, fieldnames: tuple[str, ...], robustness: bool
) -> tuple[list[dict[str, str]], dict[FactorialKey | RobustnessKey, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != fieldnames:
            raise RuntimeError("results schema does not match the registered field order")
        rows = []
        indexed: dict[FactorialKey | RobustnessKey, dict[str, str]] = {}
        for row_number, raw_row in enumerate(reader, start=2):
            if None in raw_row or any(raw_row.get(field) is None for field in fieldnames):
                raise RuntimeError(f"row {row_number}: malformed column count")
            row = {field: str(raw_row[field]) for field in fieldnames}
            key = _validate_result_row(row, row_number, robustness=robustness)
            if key in indexed:
                raise RuntimeError(f"row {row_number}: duplicate result key {key}")
            indexed[key] = row
            rows.append(row)
    return rows, indexed


def _read_completed(path: Path) -> dict[FactorialKey, dict[str, str]]:
    if not path.exists():
        return {}
    rows, _ = _read_validated_rows(path, fieldnames=FIELDS, robustness=False)
    return {
        (int(row["seed"]), row["architecture"], row["obfuscation"], row["sparsity"]): row
        for row in rows
    }


def _write_rows(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: tuple[str, ...] = FIELDS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_factorial(
    output: Path = Path("data/results/factorial_results.csv"),
    seeds: int = 100,
    epochs: int = 60,
    device: str | torch.device | None = None,
) -> int:
    expected_keys = _factorial_design_keys(seeds)
    _positive_count(epochs, "epochs")
    selected_device = device or "cpu"
    completed = _read_completed(output)
    unexpected = set(completed) - expected_keys
    if unexpected:
        raise RuntimeError(f"resume file contains out-of-design cells: {sorted(unexpected)[:3]}")
    rows: list[dict[str, object]] = [
        {field: value for field, value in row.items()} for row in completed.values()
    ]
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
            int(str(value["seed"])),
            str(value["obfuscation"]),
            str(value["sparsity"]),
            str(value["architecture"]),
        ))
        _write_rows(output, rows)
    return len(rows)


def _read_robustness_completed(
    path: Path,
) -> dict[RobustnessKey, dict[str, str]]:
    if not path.exists():
        return {}
    rows, _ = _read_validated_rows(
        path, fieldnames=ROBUSTNESS_FIELDS, robustness=True
    )
    return {
        (
            int(row["seed"]),
            row["generator"],
            row["architecture"],
            row["obfuscation"],
            row["sparsity"],
        ): row
        for row in rows
    }


def run_robustness(
    output: Path = Path("data/results/robustness_results.csv"),
    seeds: int = 30,
    epochs: int = 60,
    device: str | torch.device | None = None,
    generators: tuple[str, ...] = GENERATOR_FAMILIES,
) -> int:
    """Run a restartable cross-generator misspecification and null experiment."""

    expected_keys = _robustness_design_keys(seeds, generators)
    _positive_count(epochs, "epochs")
    selected_device = device or "cpu"
    completed = _read_robustness_completed(output)
    unexpected = set(completed) - expected_keys
    if unexpected:
        raise RuntimeError(f"resume file contains out-of-design cells: {sorted(unexpected)[:3]}")
    rows: list[dict[str, object]] = [
        {field: value for field, value in row.items()} for row in completed.values()
    ]
    design = [
        (seed, generator, architecture, obfuscation, sparsity)
        for seed in range(BASE_SEED, BASE_SEED + seeds)
        for generator in generators
        for obfuscation in OBFUSCATION_LEVELS
        for sparsity in SPARSITY_LEVELS
        for architecture in ARCHITECTURES
    ]
    rng = np.random.default_rng(1100)
    rng.shuffle(design)
    for seed, generator, architecture, obfuscation, sparsity in design:
        key = (seed, generator, architecture, obfuscation, sparsity)
        if key in completed:
            continue
        result = run_cell(
            seed,
            architecture,
            obfuscation,
            sparsity,
            epochs=epochs,
            device=selected_device,
            generator=generator,
        )
        row = {"generator": generator} | asdict(result)
        rows.append(row)
        completed[key] = {field: str(value) for field, value in row.items()}
        rows.sort(
            key=lambda value: (
                int(str(value["seed"])),
                str(value["generator"]),
                str(value["obfuscation"]),
                str(value["sparsity"]),
                str(value["architecture"]),
            )
        )
        _write_rows(output, rows, ROBUSTNESS_FIELDS)
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in the original order."""

    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Holm adjustment requires finite p-values in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running_maximum = 0.0
    family_size = values.size
    for rank, index in enumerate(order):
        candidate = min(1.0, (family_size - rank) * values[index])
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum
    return adjusted.tolist()


def _paired_inference(differences: np.ndarray) -> dict[str, object]:
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or differences.size < 2 or not np.isfinite(differences).all():
        raise ValueError("paired inference requires at least two finite differences")
    mean = float(differences.mean())
    standard_error = float(differences.std(ddof=1) / math.sqrt(differences.size))
    degrees_of_freedom = int(differences.size - 1)
    critical = float(student_t.ppf(0.975, degrees_of_freedom))
    if standard_error == 0.0:
        statistic = 0.0 if mean == 0.0 else None
        p_value = 1.0 if mean == 0.0 else 0.0
    else:
        statistic = mean / standard_error
        p_value = float(2.0 * student_t.sf(abs(statistic), degrees_of_freedom))
    return {
        "estimate_hawkes_minus_static": mean,
        "standard_error": standard_error,
        "lower_95": mean - critical * standard_error,
        "upper_95": mean + critical * standard_error,
        "t_value": statistic,
        "zero_standard_error": standard_error == 0.0,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value_two_sided": p_value,
        "paired_seeds": int(differences.size),
    }


def _mixed_fit_diagnostics(
    fit: Any, caught_warnings: list[warnings.WarningMessage]
) -> dict[str, object]:
    estimates = np.asarray(fit.fe_params, dtype=float)
    standard_errors = np.asarray(fit.bse_fe, dtype=float)
    fixed_covariance = np.asarray(fit.cov_params(), dtype=float)[
        : estimates.size, : estimates.size
    ]
    convergence_messages = [
        str(item.message)
        for item in caught_warnings
        if issubclass(item.category, ConvergenceWarning)
    ]
    covariance_finite = bool(np.isfinite(fixed_covariance).all())
    covariance_symmetric = bool(
        covariance_finite
        and np.allclose(fixed_covariance, fixed_covariance.T, atol=1e-10, rtol=1e-8)
    )
    minimum_eigenvalue = (
        float(np.linalg.eigvalsh(fixed_covariance).min())
        if covariance_symmetric
        else None
    )
    reasons = []
    if not bool(fit.converged):
        reasons.append("optimizer_did_not_converge")
    if convergence_messages:
        reasons.append("statsmodels_convergence_warning")
    if not np.isfinite(estimates).all():
        reasons.append("nonfinite_fixed_effect_estimate")
    if not np.isfinite(standard_errors).all() or np.any(standard_errors <= 0.0):
        reasons.append("invalid_fixed_effect_standard_error")
    if not covariance_symmetric or minimum_eigenvalue is None:
        reasons.append("invalid_fixed_effect_covariance")
    elif minimum_eigenvalue <= 0.0:
        reasons.append("non_positive_definite_fixed_effect_covariance")
    return {
        "converged": bool(fit.converged),
        "admissible": not reasons,
        "rejection_reasons": reasons,
        "convergence_warnings": convergence_messages,
        "fixed_effect_covariance_minimum_eigenvalue": minimum_eigenvalue,
    }


def _fit_mixed_candidate(
    response: np.ndarray,
    fixed: np.ndarray,
    groups: np.ndarray,
    random: np.ndarray,
    random_structure: str,
) -> tuple[Any, dict[str, object]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fit = MixedLM(response, fixed, groups=groups, exog_re=random).fit(
            reml=True, method="lbfgs", maxiter=500, disp=False
        )
        diagnostics = _mixed_fit_diagnostics(fit, caught)
    diagnostics["model_family"] = "mixed_linear_model"
    diagnostics["random_structure"] = random_structure
    return fit, diagnostics


def _coefficient_table(
    fit: Any, names: list[str], reference: str
) -> list[dict[str, object]]:
    estimates = np.asarray(fit.params, dtype=float)[: len(names)]
    standard_errors = np.asarray(fit.bse, dtype=float)[: len(names)]
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
    if reference == "student_t":
        confidence = np.asarray(fit.conf_int())[: len(names)]
        for index, coefficient in enumerate(coefficients):
            coefficient.update(
                {
                    "lower_95": float(confidence[index, 0]),
                    "upper_95": float(confidence[index, 1]),
                    "t_value": float(fit.tvalues[index]),
                    "p_value_two_sided": float(fit.pvalues[index]),
                }
            )
            coefficient.pop("z_value", None)
    return coefficients


def _fit_mixed_model(rows: list[dict[str, str]], endpoint: str) -> dict[str, object]:
    fixed, names, random, groups = factorial_contrast_matrix(rows)
    response = np.asarray([float(row[endpoint]) for row in rows])
    transformed = endpoint == "latency_ms_per_event"
    if transformed:
        response = np.log(np.clip(response, np.finfo(float).tiny, None))

    attempts: list[dict[str, object]] = []
    fit = None
    random_structure = "seed_random_intercept_and_architecture_slope"
    for candidate_random, candidate_name in (
        (random, "seed_random_intercept_and_architecture_slope"),
        (random[:, :1], "seed_random_intercept"),
    ):
        try:
            candidate_fit, diagnostics = _fit_mixed_candidate(
                response, fixed, groups, candidate_random, candidate_name
            )
        except (np.linalg.LinAlgError, ValueError) as error:
            attempts.append(
                {
                    "model_family": "mixed_linear_model",
                    "random_structure": candidate_name,
                    "converged": False,
                    "admissible": False,
                    "rejection_reasons": ["fit_exception"],
                    "convergence_warnings": [],
                    "fit_exception": f"{type(error).__name__}: {error}",
                    "fixed_effect_covariance_minimum_eigenvalue": None,
                }
            )
            continue
        attempts.append(diagnostics)
        if diagnostics["admissible"]:
            fit = candidate_fit
            random_structure = candidate_name
            break

    if fit is None:
        fit = OLS(response, fixed).fit(
            cov_type="cluster", cov_kwds={"groups": groups}, use_t=True
        )
        random_structure = "none_seed_cluster_robust"
        model_family = "ordinary_least_squares"
        reference_distribution = "student_t"
        covariance = np.asarray(fit.cov_params(), dtype=float)
        covariance_finite = bool(np.isfinite(covariance).all())
        covariance_symmetric = bool(
            covariance_finite
            and np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-8)
        )
        covariance_eigenvalues = (
            np.linalg.eigvalsh(covariance) if covariance_symmetric else np.asarray([])
        )
        minimum_eigenvalue = (
            float(covariance_eigenvalues.min()) if covariance_eigenvalues.size else None
        )
        eigenvalue_tolerance = (
            1e-10 * max(1.0, float(np.max(np.abs(covariance_eigenvalues))))
            if covariance_eigenvalues.size
            else 0.0
        )
        final_admissible = bool(
            np.isfinite(np.asarray(fit.params)).all()
            and np.isfinite(np.asarray(fit.bse)).all()
            and np.all(np.asarray(fit.bse) > 0.0)
            and minimum_eigenvalue is not None
            and minimum_eigenvalue >= -eigenvalue_tolerance
        )
        attempts.append(
            {
                "model_family": "ordinary_least_squares",
                "random_structure": random_structure,
                "converged": True,
                "admissible": final_admissible,
                "rejection_reasons": []
                if final_admissible
                else ["invalid_cluster_robust_fixed_effect_inference"],
                "convergence_warnings": [],
                "fixed_effect_covariance_minimum_eigenvalue": minimum_eigenvalue,
            }
        )
        if not final_admissible:
            raise RuntimeError(
                f"no admissible inference model for endpoint {endpoint}; "
                "see fit-attempt diagnostics"
            )
    else:
        model_family = "mixed_linear_model"
        reference_distribution = "normal"
        final_admissible = True

    coefficients = _coefficient_table(fit, names, reference_distribution)
    return {
        "endpoint": endpoint,
        "response_scale": "natural_log" if transformed else "identity",
        "model_family": model_family,
        "random_structure": random_structure,
        "reference_distribution": reference_distribution,
        "inference_degrees_of_freedom": (
            float(getattr(fit, "df_resid_inference", fit.df_resid))
            if reference_distribution == "student_t"
            else None
        ),
        "admissible": final_admissible,
        "fit_attempts": attempts,
        "coefficients": coefficients,
    }


def analyze_factorial(
    results_path: Path = Path("data/results/factorial_results.csv"),
    public_summary: Path = Path("data/empirical_summary.json"),
    expected_seeds: int = 100,
) -> dict[str, object]:
    expected_keys = _factorial_design_keys(expected_seeds)
    if results_path.resolve() == public_summary.resolve():
        raise ValueError("results and summary paths must be different")
    results_sha256_before = _sha256_file(results_path)
    rows, indexed = _read_validated_rows(
        results_path, fieldnames=FIELDS, robustness=False
    )
    results_sha256_after = _sha256_file(results_path)
    if results_sha256_before != results_sha256_after:
        raise RuntimeError("results file changed while analysis was reading it")
    observed_keys: set[FactorialKey] = {
        (int(row["seed"]), row["architecture"], row["obfuscation"], row["sparsity"])
        for row in rows
    }
    if observed_keys != expected_keys or len(indexed) != len(expected_keys):
        raise RuntimeError(
            f"analysis remains locked: expected {len(expected_keys)} exact completed cells"
        )
    endpoints = ("link_auc", "link_log_score", "latency_ms_per_event")
    contrasts: list[dict[str, object]] = []
    for obfuscation in OBFUSCATION_LEVELS:
        for sparsity in SPARSITY_LEVELS:
            selected = [row for row in rows if row["obfuscation"] == obfuscation and row["sparsity"] == sparsity]
            by_seed: dict[int, dict[str, dict[str, str]]] = {}
            for row in selected:
                by_seed.setdefault(int(row["seed"]), {})[row["architecture"]] = row
            for endpoint in endpoints:
                differences = np.asarray(
                    [
                        float(by_seed[seed]["hawkes_flow_dsbm"][endpoint])
                        - float(by_seed[seed]["static_gae_louvain"][endpoint])
                        for seed in sorted(by_seed)
                    ]
                )
                contrasts.append(
                    {
                        "endpoint": endpoint,
                        "obfuscation": obfuscation,
                        "sparsity": sparsity,
                    }
                    | _paired_inference(differences)
                )

    for endpoint in endpoints:
        family = [contrast for contrast in contrasts if contrast["endpoint"] == endpoint]
        adjusted = _holm_adjust(
            [float(str(contrast["p_value_two_sided"])) for contrast in family]
        )
        for contrast, adjusted_p in zip(family, adjusted, strict=True):
            contrast["p_value_holm"] = adjusted_p
            contrast["reject_holm_0_05"] = adjusted_p <= 0.05
    summary = {
        "schema": "encrypted-topology-empirical-summary-v3",
        "analysis_engine_version": ANALYSIS_ENGINE_VERSION,
        "status": "complete_controlled_truth_factorial",
        "rows": len(rows),
        "seeds": expected_seeds,
        "results_sha256": results_sha256_before,
        "results_hash_algorithm": "sha256",
        "paired_inference": {
            "method": "paired_seed_t",
            "confidence_intervals": "condition-specific unadjusted 95% intervals",
            "multiplicity": "Holm adjustment within each endpoint across nine conditions",
            "familywise_alpha": 0.05,
        },
        "contrasts": contrasts,
        "mixed_effects_models": [_fit_mixed_model(rows, endpoint) for endpoint in endpoints],
        "claim_boundary": (
            "Controlled-truth simulation only; not evidence about real communications, identity, "
            "intent, attribution, content recovery, or operational SIGINT performance."
        ),
    }
    _write_json_atomic(public_summary, summary)
    return summary


def analyze_robustness(
    results_path: Path = Path("data/results/robustness_results.csv"),
    public_summary: Path = Path("data/robustness_summary.json"),
    expected_seeds: int = 30,
    generators: tuple[str, ...] = GENERATOR_FAMILIES,
) -> dict[str, object]:
    """Analyze paired model differences across generator families without mutating results."""

    expected_keys = _robustness_design_keys(expected_seeds, generators)
    if results_path.resolve() == public_summary.resolve():
        raise ValueError("results and summary paths must be different")

    results_sha256_before = _sha256_file(results_path)
    rows, indexed = _read_validated_rows(
        results_path, fieldnames=ROBUSTNESS_FIELDS, robustness=True
    )
    results_sha256_after = _sha256_file(results_path)
    if results_sha256_before != results_sha256_after:
        raise RuntimeError("results file changed while analysis was reading it")

    observed_keys: set[RobustnessKey] = {
        (
            int(row["seed"]),
            row["generator"],
            row["architecture"],
            row["obfuscation"],
            row["sparsity"],
        )
        for row in rows
    }
    if (
        len(indexed) != len(expected_keys)
        or observed_keys != expected_keys
    ):
        raise RuntimeError(
            f"analysis remains locked: expected {len(expected_keys)} exact completed cells"
        )

    endpoints = ("link_auc", "link_log_score", "latency_ms_per_event")
    contrasts: list[dict[str, object]] = []
    for generator in generators:
        for obfuscation in OBFUSCATION_LEVELS:
            for sparsity in SPARSITY_LEVELS:
                selected = [
                    row
                    for row in rows
                    if row["generator"] == generator
                    and row["obfuscation"] == obfuscation
                    and row["sparsity"] == sparsity
                ]
                by_seed: dict[int, dict[str, dict[str, str]]] = {}
                for row in selected:
                    by_seed.setdefault(int(row["seed"]), {})[row["architecture"]] = row
                for endpoint in endpoints:
                    differences = np.asarray(
                        [
                            float(by_seed[seed]["hawkes_flow_dsbm"][endpoint])
                            - float(by_seed[seed]["static_gae_louvain"][endpoint])
                            for seed in sorted(by_seed)
                        ]
                    )
                    contrasts.append(
                        {
                            "generator": generator,
                            "endpoint": endpoint,
                            "obfuscation": obfuscation,
                            "sparsity": sparsity,
                        }
                        | _paired_inference(differences)
                    )

    for endpoint in endpoints:
        family = [contrast for contrast in contrasts if contrast["endpoint"] == endpoint]
        adjusted = _holm_adjust(
            [float(str(contrast["p_value_two_sided"])) for contrast in family]
        )
        for contrast, adjusted_p in zip(family, adjusted, strict=True):
            contrast["p_value_holm"] = adjusted_p
            contrast["reject_holm_0_05"] = adjusted_p <= 0.05

    approved_registry = {
        record["name"]: record
        for record in simulation_registry()
        if record["name"] in generators
    }
    summary = {
        "schema": "encrypted-topology-robustness-summary-v2",
        "analysis_engine_version": ANALYSIS_ENGINE_VERSION,
        "status": "complete_simulation_robustness_experiment",
        "rows": len(rows),
        "seeds": expected_seeds,
        "generators": list(generators),
        "simulation_registry": [approved_registry[name] for name in generators],
        "results_sha256": results_sha256_before,
        "results_hash_algorithm": "sha256",
        "paired_inference": {
            "method": "paired_seed_t",
            "confidence_intervals": "condition-specific unadjusted 95% intervals",
            "multiplicity": (
                "Holm adjustment within each endpoint across all generator-by-condition contrasts"
            ),
            "familywise_alpha": 0.05,
        },
        "contrasts": contrasts,
        "claim_boundary": (
            "Synthetic robustness and negative-control evidence only; not evidence about real "
            "communications, identity, intent, attribution, content recovery, or operational "
            "SIGINT performance."
        ),
    }
    _write_json_atomic(public_summary, summary)
    return summary
