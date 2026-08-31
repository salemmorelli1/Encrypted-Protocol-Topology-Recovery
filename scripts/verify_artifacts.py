"""Fail closed if publication artifacts or claim-boundary text drift."""

from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "report" / "Encrypted_Protocol_Topology_Recovery_APA_Report.pdf"
HTML = ROOT / "index.html"


def main() -> None:
    if not PDF.exists():
        raise SystemExit("report PDF is missing")
    pages = len(PdfReader(str(PDF)).pages)
    if pages != 27:
        raise SystemExit(f"report has {pages} pages; expected 27")
    html = HTML.read_text(encoding="utf-8")
    required = (
        "--authorized-capture",
        "No operational cyber-SIGINT claim",
        "Empirical contrasts unavailable",
        "no live capture",
    )
    missing = [phrase for phrase in required if phrase.lower() not in html.lower()]
    if missing:
        raise SystemExit(f"dashboard claim-boundary phrases missing: {missing}")
    print(f"verified: {PDF.name}, {pages} pages; dashboard claim boundary present")


if __name__ == "__main__":
    main()
