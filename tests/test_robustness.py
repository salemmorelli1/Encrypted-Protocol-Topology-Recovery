import csv
import hashlib

import pytest

from encrypted_topology.experiment import (
    ARCHITECTURES,
    BASE_SEED,
    ROBUSTNESS_FIELDS,
    CellResult,
    analyze_robustness,
    run_robustness,
)
from encrypted_topology.simulator import (
    GENERATOR_FAMILIES,
    OBFUSCATION_LEVELS,
    SPARSITY_LEVELS,
)


def _rows(seeds: int = 2) -> list[dict[str, object]]:
    return [
        {
            "generator": generator,
            "seed": seed,
            "architecture": architecture,
            "obfuscation": obfuscation,
            "sparsity": sparsity,
            "events": 120,
            "link_auc": 0.60 + 0.02 * (architecture == "hawkes_flow_dsbm") + seed / 1e6,
            "link_log_score": -0.55 + 0.01 * (architecture == "hawkes_flow_dsbm") - seed / 1e6,
            "latency_ms_per_event": 1.0 + 0.4 * (architecture == "hawkes_flow_dsbm"),
            "occupied_communities": 4,
            "importance_ess_fraction": (
                0.5 if architecture == "hawkes_flow_dsbm" else "nan"
            ),
            "status": "complete",
        }
        for seed in range(BASE_SEED, BASE_SEED + seeds)
        for generator in GENERATOR_FAMILIES
        for architecture in ARCHITECTURES
        for obfuscation in OBFUSCATION_LEVELS
        for sparsity in SPARSITY_LEVELS
    ]


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROBUSTNESS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_robustness_analysis_locks_schema_keys_hash_and_multiplicity(tmp_path):
    results_path = tmp_path / "robustness.csv"
    summary_path = tmp_path / "summary.json"
    _write(results_path, _rows())
    frozen_results = results_path.read_bytes()
    expected_hash = hashlib.sha256(frozen_results).hexdigest()

    summary = analyze_robustness(results_path, summary_path, expected_seeds=2)

    assert summary["schema"] == "encrypted-topology-robustness-summary-v2"
    assert summary["results_sha256"] == expected_hash
    assert len(summary["contrasts"]) == 135
    assert sum(item["endpoint"] == "link_auc" for item in summary["contrasts"]) == 45
    assert all("p_value_holm" in item for item in summary["contrasts"])
    assert results_path.read_bytes() == frozen_results


def test_robustness_analysis_rejects_incomplete_exact_design(tmp_path):
    results_path = tmp_path / "robustness.csv"
    _write(results_path, _rows()[:-1])
    with pytest.raises(RuntimeError, match="exact completed cells"):
        analyze_robustness(results_path, tmp_path / "summary.json", expected_seeds=2)


def test_robustness_runner_is_restartable_and_uses_separate_schema(tmp_path, monkeypatch):
    calls = []

    def fake_cell(seed, architecture, obfuscation, sparsity, **kwargs):
        calls.append((seed, architecture, obfuscation, sparsity, kwargs["generator"]))
        return CellResult(
            seed=seed,
            architecture=architecture,
            obfuscation=obfuscation,
            sparsity=sparsity,
            events=100,
            link_auc=0.5,
            link_log_score=-0.6,
            latency_ms_per_event=1.0,
            occupied_communities=3,
            importance_ess_fraction=(
                0.5 if architecture == "hawkes_flow_dsbm" else float("nan")
            ),
        )

    monkeypatch.setattr("encrypted_topology.experiment.run_cell", fake_cell)
    output = tmp_path / "robustness.csv"
    assert run_robustness(
        output, seeds=1, epochs=1, generators=("independent_null",)
    ) == 18
    assert len(calls) == 18
    assert run_robustness(
        output, seeds=1, epochs=1, generators=("independent_null",)
    ) == 18
    assert len(calls) == 18
    with output.open(newline="", encoding="utf-8") as stream:
        assert tuple(csv.DictReader(stream).fieldnames or ()) == ROBUSTNESS_FIELDS
