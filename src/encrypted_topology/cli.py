"""Command-line entry points for authorized acquisition and controlled experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

from .capture import CapturePolicy, capture_to_file
from .experiment import analyze_factorial, run_factorial
from .io import load_capture
from .model import HawkesFlowDSBM
from .simulator import simulate_network


def list_interfaces() -> dict[str, object]:
    response: dict[str, object] = {"scapy": [], "tshark": []}
    try:
        from scapy.all import get_if_list

        response["scapy"] = list(get_if_list())
    except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - platform dependent
        response["scapy_error"] = str(exc)
    try:
        completed = subprocess.run(
            ["tshark", "-D"], capture_output=True, text=True, check=False, shell=False
        )
        response["tshark"] = completed.stdout.splitlines()
    except OSError as exc:  # pragma: no cover
        response["tshark_error"] = str(exc)
    return response


def _capture_policy(args: argparse.Namespace) -> CapturePolicy:
    return CapturePolicy(
        interface=args.interface,
        duration_seconds=args.duration,
        packet_limit=args.packet_limit,
        capture_filter=args.capture_filter,
        backend=args.backend,
        authorized_capture=args.authorized_capture,
    )


def _default_capture_path() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("data/captures") / f"metadata-{stamp}.json.gz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encrypted topology recovery under an explicit authorized-capture boundary"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("interfaces", help="list locally available Scapy and TShark interfaces")

    for command in ("capture", "live"):
        capture = subparsers.add_parser(command)
        capture.add_argument("--backend", choices=("scapy", "tshark"), default="scapy")
        capture.add_argument("--interface", required=True)
        capture.add_argument("--duration", type=float, default=60.0)
        capture.add_argument("--packet-limit", type=int, default=10_000)
        capture.add_argument("--capture-filter", default="ip or ip6")
        capture.add_argument("--output", type=Path)
        capture.add_argument("--authorized-capture", action="store_true")
        if command == "live":
            capture.add_argument("--epochs", type=int, default=80)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--epochs", type=int, default=80)

    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--seed", type=int, default=2026)
    simulate.add_argument("--sparsity", choices=("dense", "moderate", "sparse"), default="moderate")
    simulate.add_argument("--obfuscation", choices=("none", "padding", "jitter"), default="none")
    simulate.add_argument("--epochs", type=int, default=20)

    factorial = subparsers.add_parser("run-factorial")
    factorial.add_argument("--seeds", type=int, default=100)
    factorial.add_argument("--epochs", type=int, default=60)
    factorial.add_argument("--output", type=Path, default=Path("data/results/factorial_results.csv"))

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--results", type=Path, default=Path("data/results/factorial_results.csv"))
    analyze.add_argument("--seeds", type=int, default=100)
    return parser


def _fit_and_summarize(batch, epochs: int) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HawkesFlowDSBM().to(device)
    fit = model.fit_batch(batch.to(device), epochs=epochs)
    return {
        "events": batch.num_events,
        "nodes": batch.num_nodes,
        "device": str(device),
        "latency_ms": fit.latency_ms,
        "final_elbo_per_event": fit.final_elbo_per_event,
        "occupied_blocks": fit.occupied_blocks,
        "claim_boundary": "Unlabelled topology posterior; no live link-AUC claim without external truth.",
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "interfaces":
        result = list_interfaces()
    elif args.command in {"capture", "live"}:
        destination = args.output or _default_capture_path()
        body = capture_to_file(_capture_policy(args), destination)
        result = {
            "output": str(destination),
            "events": len(body["events"]),
            "sha256": body["sha256"],
            "payload_retained": False,
        }
        if args.command == "live":
            result["inference"] = _fit_and_summarize(load_capture(destination), args.epochs)
    elif args.command == "infer":
        result = _fit_and_summarize(load_capture(args.input), args.epochs)
    elif args.command == "simulate":
        simulated = simulate_network(args.seed, args.sparsity, args.obfuscation)
        result = simulated.condition | {"inference": _fit_and_summarize(simulated.batch, args.epochs)}
    elif args.command == "run-factorial":
        result = {
            "rows": run_factorial(args.output, args.seeds, args.epochs),
            "output": str(args.output),
        }
    else:
        result = analyze_factorial(args.results, expected_seeds=args.seeds)
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
