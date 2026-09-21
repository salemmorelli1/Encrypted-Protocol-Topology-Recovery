"""Fail closed if simulation-only publication or reproducibility contracts drift."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "report" / "Encrypted_Protocol_Topology_Recovery_APA_Report.pdf"
HTML = ROOT / "index.html"
STATUS = ROOT / "data" / "project_status.json"
HISTORICAL = ROOT / "data" / "empirical_summary.json"
DESIGN = ROOT / "data" / "experiment_design.csv"
LOCK = ROOT / "requirements-ci-lock.txt"
RESULTS_SHA256 = "d19433ff32f2222655f5408810b9ed6fd59fa01ea461e05e3f1aa5a59977309a"
HISTORICAL_SUMMARY_SHA256 = (
    "a772fda74881a03c5664c8e822877d1ce52227d669491fc8fa3850ef5869de88"
)
ACTION_PINS = {
    "actions/checkout": "11d5960a326750d5838078e36cf38b85af677262",
    "actions/setup-python": "a26af69be951a213d495a4c3e4e4022e16d87065",
    "actions/configure-pages": "983d7736d9b0ae728b81ab479565c72886d7745b",
    "actions/upload-pages-artifact": "56afc609e74202658d3ffba0e8f6dda462b719fa",
    "actions/deploy-pages": "d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e",
}


def _verify_pdf(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"report PDF is missing: {path}")
    reader = PdfReader(str(path))
    if len(reader.pages) != 27:
        raise SystemExit(f"{path.name} has {len(reader.pages)} pages; expected 27")
    report_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for phrase in ("Simulation-Only", "quarantined", "1,800", "2,700", "operational SIGINT"):
        if phrase.lower() not in report_text.lower():
            raise SystemExit(f"{path.name} phrase missing: {phrase}")


def _verify_dashboard() -> None:
    html = HTML.read_text(encoding="utf-8")
    required = (
        "Simulation-only",
        "No operational cyber-SIGINT claim",
        RESULTS_SHA256,
        "historical results quarantined",
        "corrected rerun pending",
        "robustness runs pending",
        "independent_null",
        "Synthetic cryptography",
        "key-withheld",
        "no key recovery",
    )
    missing = [phrase for phrase in required if phrase.lower() not in html.lower()]
    if missing:
        raise SystemExit(f"dashboard claim-boundary phrases missing: {missing}")
    forbidden = (
        "--authorized-capture",
        "encrypted-topology live",
        'data-view="capture"',
        "Primary study complete",
    )
    present = [phrase for phrase in forbidden if phrase.lower() in html.lower()]
    if present:
        raise SystemExit(f"dashboard contains forbidden acquisition or evidence claim: {present}")


def _verify_status() -> None:
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    expected = {
        "schema": "encrypted-topology-project-status-v3",
        "version": "1.2.0",
        "data_mode": "simulation_only",
        "external_input_supported": False,
        "external_ciphertext_supported": False,
        "collection_modules_present": False,
        "synthetic_crypto_lab_present": True,
        "authorized_synthetic_decryption_only": True,
        "key_recovery_supported": False,
        "key_withheld_control_present": True,
        "historical_factorial_reported": True,
        "historical_factorial_cells": 1800,
        "historical_factorial_results_sha256": RESULTS_SHA256,
        "historical_factorial_summary_path": "data/empirical_summary.json",
        "historical_factorial_original_summary_sha256": HISTORICAL_SUMMARY_SHA256,
        "historical_factorial_source_present": False,
        "historical_factorial_quarantined": True,
        "controlled_truth_factorial_complete": False,
        "planned_factorial_cells": 1800,
        "factorial_analyzer": "1.2.0",
        "robustness_experiment_complete": False,
        "planned_robustness_cells": 2700,
        "operational_sigint_claim": False,
    }
    drift = {
        key: (status.get(key), value)
        for key, value in expected.items()
        if status.get(key) != value
    }
    if drift:
        raise SystemExit(f"machine-readable status drift: {drift}")

    historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    trace = historical
    if (
        historical.get("status") != "quarantined_not_current_evidence"
        or historical.get("current_evidence") is not False
        or historical.get("source_results_present") is not False
        or historical.get("historical_values_preserved") is not True
        or historical.get("historical_original_summary_sha256")
        != HISTORICAL_SUMMARY_SHA256
        or len(historical.get("quarantine_reasons", [])) < 3
        or trace.get("rows") != 1800
        or trace.get("results_sha256") != RESULTS_SHA256
        or len(trace.get("contrasts", [])) != 27
    ):
        raise SystemExit("historical evidence quarantine drift")
    contrasts = trace["contrasts"]
    for endpoint in ("link_auc", "link_log_score", "latency_ms_per_event"):
        family = [item for item in contrasts if item.get("endpoint") == endpoint]
        if len(family) != 9:
            raise SystemExit(f"historical {endpoint} contrast-count drift")
        order = sorted(range(9), key=lambda index: family[index]["p_value_two_sided"])
        running = 0.0
        expected_adjusted = [0.0] * 9
        for rank, index in enumerate(order):
            candidate = min(1.0, (9 - rank) * family[index]["p_value_two_sided"])
            running = max(running, candidate)
            expected_adjusted[index] = running
        if any(
            not math.isclose(item["p_value_holm"], expected, rel_tol=1e-12, abs_tol=1e-15)
            for item, expected in zip(family, expected_adjusted, strict=True)
        ):
            raise SystemExit(f"historical {endpoint} Holm-adjustment drift")


def _verify_design() -> None:
    with DESIGN.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    observed = {
        (row["architecture"], row["obfuscation"], row["sparsity"])
        for row in rows
    }
    expected = {
        (architecture, obfuscation, sparsity)
        for architecture in ("hawkes_flow_dsbm", "static_gae_louvain")
        for obfuscation in ("none", "padding", "jitter")
        for sparsity in ("dense", "moderate", "sparse")
    }
    if len(rows) != 18 or observed != expected:
        raise SystemExit("registered primary design must contain exactly 18 unique cells")
    if any(row["planned_seeds"] != "100" or row["planned_runs"] != "100" for row in rows):
        raise SystemExit("registered primary design counts drift")


def _verify_pins() -> None:
    lock_lines = [
        line.strip()
        for line in LOCK.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lock_lines) != 13 or any(line.count("==") != 1 for line in lock_lines):
        raise SystemExit("CI lock must contain 13 exact direct dependency pins")

    workflow_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    )
    uses = re.findall(r"uses:\s*([^@\s]+)@([^\s#]+)", workflow_text)
    seen = {action for action, _ in uses}
    drift = {
        action: (reference, ACTION_PINS.get(action))
        for action, reference in uses
        if action not in ACTION_PINS or reference != ACTION_PINS[action]
    }
    drift.update(
        {
            action: (None, pin)
            for action, pin in ACTION_PINS.items()
            if action not in seen
        }
    )
    if drift:
        raise SystemExit(f"GitHub Action pin drift: {drift}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--built-report", type=Path)
    args = parser.parse_args()

    _verify_pdf(PDF)
    if args.built_report is not None:
        _verify_pdf(args.built_report)
    _verify_dashboard()
    _verify_status()
    _verify_design()
    _verify_pins()
    print(
        "verified: committed and rebuilt 27-page report contracts, historical quarantine, "
        "registered design, simulation-only dashboard, dependency pins, and Action pins"
    )


if __name__ == "__main__":
    main()
