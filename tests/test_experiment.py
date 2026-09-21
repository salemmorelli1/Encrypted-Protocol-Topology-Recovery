import csv
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from encrypted_topology import experiment
from encrypted_topology.experiment import (
    ARCHITECTURES,
    FIELDS,
    _fit_mixed_model,
    _holm_adjust,
    _paired_inference,
    analyze_factorial,
    factorial_contrast_matrix,
    run_cell,
    run_factorial,
)
from encrypted_topology.simulator import OBFUSCATION_LEVELS, SPARSITY_LEVELS


def _factorial_rows(seeds=(2026, 2027)):
    return [
        {
            "seed": str(seed),
            "architecture": architecture,
            "obfuscation": obfuscation,
            "sparsity": sparsity,
            "events": "100",
            "link_auc": str(
                0.7
                + 0.01 * (seed - 2026)
                + (0.02 if architecture == "hawkes_flow_dsbm" else 0.0)
            ),
            "link_log_score": str(
                -0.5
                - 0.01 * (seed - 2026)
                + (0.03 if architecture == "hawkes_flow_dsbm" else 0.0)
            ),
            "latency_ms_per_event": str(
                1.0
                + 0.1 * (seed - 2026)
                + (0.5 if architecture == "hawkes_flow_dsbm" else 0.0)
            ),
            "occupied_communities": "3",
            "importance_ess_fraction": (
                "0.5" if architecture == "hawkes_flow_dsbm" else "nan"
            ),
            "status": "complete",
        }
        for seed in seeds
        for architecture in ARCHITECTURES
        for obfuscation in OBFUSCATION_LEVELS
        for sparsity in SPARSITY_LEVELS
    ]


def test_factorial_contrast_design_is_full_rank_and_architecture_signed():
    rows = _factorial_rows()
    fixed, names, random, groups = factorial_contrast_matrix(rows)
    assert fixed.shape == (36, 18)
    assert np.linalg.matrix_rank(fixed) == 18
    assert random.shape == (36, 2)
    assert np.unique(groups).tolist() == [2026, 2027]
    architecture_column = fixed[:, names.index("architecture")]
    assert set(np.unique(architecture_column)) == {-0.5, 0.5}


def test_holm_adjustment_is_step_down_and_order_preserving():
    assert _holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_paired_inference_uses_seed_level_differences():
    result = _paired_inference(np.asarray([1.0, 2.0, 3.0, 4.0]))
    assert result["estimate_hawkes_minus_static"] == pytest.approx(2.5)
    assert result["paired_seeds"] == 4
    assert result["degrees_of_freedom"] == 3
    assert 0.0 <= result["p_value_two_sided"] <= 1.0


def test_diagnostic_failure_selects_simpler_mixed_model(monkeypatch):
    fake_fit = SimpleNamespace(
        params=np.zeros(18),
        bse=np.ones(18),
        df_resid=18.0,
    )
    attempts = iter(
        [
            {
                "model_family": "mixed_linear_model",
                "random_structure": "seed_random_intercept_and_architecture_slope",
                "converged": True,
                "admissible": False,
                "rejection_reasons": ["statsmodels_convergence_warning"],
                "convergence_warnings": ["boundary"],
                "fixed_effect_covariance_minimum_eigenvalue": 0.1,
            },
            {
                "model_family": "mixed_linear_model",
                "random_structure": "seed_random_intercept",
                "converged": True,
                "admissible": True,
                "rejection_reasons": [],
                "convergence_warnings": [],
                "fixed_effect_covariance_minimum_eigenvalue": 0.1,
            },
        ]
    )

    def fake_candidate(*args, **kwargs):
        return fake_fit, next(attempts)

    monkeypatch.setattr(experiment, "_fit_mixed_candidate", fake_candidate)
    result = _fit_mixed_model(_factorial_rows(), "link_auc")
    assert result["model_family"] == "mixed_linear_model"
    assert result["random_structure"] == "seed_random_intercept"
    assert len(result["fit_attempts"]) == 2
    assert result["fit_attempts"][0]["admissible"] is False


def test_two_rejected_mixed_models_select_clustered_ols(monkeypatch):
    def fake_candidate(*args, **kwargs):
        name = args[-1]
        return None, {
            "model_family": "mixed_linear_model",
            "random_structure": name,
            "converged": False,
            "admissible": False,
            "rejection_reasons": ["optimizer_did_not_converge"],
            "convergence_warnings": [],
            "fixed_effect_covariance_minimum_eigenvalue": None,
        }

    class FakeOLSResult:
        params = np.zeros(18)
        bse = np.ones(18)
        tvalues = np.zeros(18)
        pvalues = np.ones(18)
        df_resid = 18.0
        df_resid_inference = 99.0

        @staticmethod
        def cov_params():
            return np.eye(18)

        @staticmethod
        def conf_int():
            return np.column_stack((-np.ones(18), np.ones(18)))

    class FakeOLS:
        def fit(self, **kwargs):
            return FakeOLSResult()

    monkeypatch.setattr(experiment, "_fit_mixed_candidate", fake_candidate)
    monkeypatch.setattr(experiment, "OLS", lambda *args, **kwargs: FakeOLS())
    result = _fit_mixed_model(_factorial_rows(), "link_auc")
    assert result["model_family"] == "ordinary_least_squares"
    assert result["random_structure"] == "none_seed_cluster_robust"
    assert result["reference_distribution"] == "student_t"
    assert result["inference_degrees_of_freedom"] == 99.0
    assert len(result["fit_attempts"]) == 3


def test_analysis_records_sha256_and_holm_inference(tmp_path, monkeypatch):
    results_path = tmp_path / "factorial_results.csv"
    summary_path = tmp_path / "empirical_summary.json"
    rows = _factorial_rows()
    with results_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    expected_hash = hashlib.sha256(results_path.read_bytes()).hexdigest()

    monkeypatch.setattr(
        experiment,
        "_fit_mixed_model",
        lambda selected_rows, endpoint: {"endpoint": endpoint, "admissible": True},
    )
    summary = analyze_factorial(results_path, summary_path, expected_seeds=2)
    stored = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["schema"] == "encrypted-topology-empirical-summary-v3"
    assert summary["results_sha256"] == expected_hash
    assert stored["results_sha256"] == expected_hash
    assert len(summary["contrasts"]) == 27
    assert all("p_value_holm" in contrast for contrast in summary["contrasts"])
    assert all("reject_holm_0_05" in contrast for contrast in summary["contrasts"])


def test_analysis_rejects_wrong_seed_set_without_overwriting_summary(tmp_path):
    results_path = tmp_path / "factorial_results.csv"
    summary_path = tmp_path / "empirical_summary.json"
    rows = _factorial_rows()
    for row in rows:
        if row["seed"] == "2027":
            row["seed"] = "2028"
    with results_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="exact completed cells"):
        analyze_factorial(results_path, summary_path, expected_seeds=2)
    assert summary_path.read_text(encoding="utf-8") == "sentinel\n"


def test_factorial_resume_rejects_duplicate_keys(tmp_path):
    output = tmp_path / "factorial.csv"
    row = _factorial_rows(seeds=(2026,))[0]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([row, row])

    with pytest.raises(RuntimeError, match="duplicate result key"):
        run_factorial(output, seeds=1, epochs=1)


def test_analysis_refuses_to_overwrite_its_results_source(tmp_path):
    path = tmp_path / "factorial.csv"
    path.write_text("placeholder\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be different"):
        analyze_factorial(path, path, expected_seeds=2)


def test_dynamic_cell_is_repeatable_from_declared_seed():
    first = run_cell(2026, "hawkes_flow_dsbm", "jitter", "dense", epochs=1)
    second = run_cell(2026, "hawkes_flow_dsbm", "jitter", "dense", epochs=1)
    assert first.link_auc == pytest.approx(second.link_auc, abs=0.0)
    assert first.link_log_score == pytest.approx(second.link_log_score, abs=0.0)
    assert first.occupied_communities == second.occupied_communities
    assert first.importance_ess_fraction == pytest.approx(
        second.importance_ess_fraction, abs=0.0
    )
