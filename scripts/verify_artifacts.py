"""Fail closed if simulation-only publication artifacts or status drift."""

from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "report" / "Encrypted_Protocol_Topology_Recovery_APA_Report.pdf"
HTML = ROOT / "index.html"
STATUS = ROOT / "data" / "project_status.json"
RESULTS_SHA256 = "d19433ff32f2222655f5408810b9ed6fd59fa01ea461e05e3f1aa5a59977309a"


def main() -> None:
    if not PDF.exists():
        raise SystemExit("report PDF is missing")
    reader = PdfReader(str(PDF))
    if len(reader.pages) != 27:
        raise SystemExit(f"report has {len(reader.pages)} pages; expected 27")
    report_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for phrase in ("Simulation-Only", "1,800", "2,700", "operational SIGINT"):
        if phrase.lower() not in report_text.lower():
            raise SystemExit(f"report phrase missing: {phrase}")

    html = HTML.read_text(encoding="utf-8")
    required = (
        "Simulation-only",
        "No operational cyber-SIGINT claim",
        RESULTS_SHA256,
        "robustness pending",
        "independent_null",
    )
    missing = [phrase for phrase in required if phrase.lower() not in html.lower()]
    if missing:
        raise SystemExit(f"dashboard claim-boundary phrases missing: {missing}")
    forbidden = ("--authorized-capture", "encrypted-topology live", 'data-view="capture"')
    present = [phrase for phrase in forbidden if phrase.lower() in html.lower()]
    if present:
        raise SystemExit(f"dashboard contains removed acquisition surface: {present}")

    status = json.loads(STATUS.read_text(encoding="utf-8"))
    expected_status = {
        "version": "1.1.0",
        "data_mode": "simulation_only",
        "external_input_supported": False,
        "collection_modules_present": False,
        "controlled_truth_factorial_complete": True,
        "factorial_cells": 1800,
        "factorial_results_sha256": RESULTS_SHA256,
        "robustness_experiment_complete": False,
        "planned_robustness_cells": 2700,
        "operational_sigint_claim": False,
    }
    drift = {
        key: (status.get(key), expected)
        for key, expected in expected_status.items()
        if status.get(key) != expected
    }
    if drift:
        raise SystemExit(f"machine-readable status drift: {drift}")
    print(
        f"verified: {PDF.name}, 27 pages; simulation-only dashboard and status boundary present"
    )


if __name__ == "__main__":
    main()
