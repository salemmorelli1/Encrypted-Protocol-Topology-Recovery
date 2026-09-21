"""Command-line entry points for the simulation-only research laboratory."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .crypto_lab import run_synthetic_crypto_lab
from .events import EventBatch
from .experiment import (
    analyze_factorial,
    analyze_robustness,
    run_factorial,
    run_robustness,
)
from .model import HawkesFlowDSBM
from .simulator import (
    GENERATOR_FAMILIES,
    OBFUSCATION_LEVELS,
    SPARSITY_LEVELS,
    simulate_network,
    simulation_registry,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulation-only topology-recovery research; no capture or external-data commands"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "simulation-registry", help="describe every synthetic event generator"
    )

    simulate = subparsers.add_parser("simulate", help="fit one controlled synthetic sequence")
    simulate.add_argument("--seed", type=_nonnegative_int, default=2026)
    simulate.add_argument("--sparsity", choices=SPARSITY_LEVELS, default="moderate")
    simulate.add_argument("--obfuscation", choices=OBFUSCATION_LEVELS, default="none")
    simulate.add_argument(
        "--generator", choices=GENERATOR_FAMILIES, default=GENERATOR_FAMILIES[0]
    )
    simulate.add_argument("--epochs", type=_positive_int, default=20)
    simulate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    crypto_lab = subparsers.add_parser(
        "crypto-lab", help="run an in-memory AES-GCM experiment on synthetic events"
    )
    crypto_lab.add_argument("--seed", type=_nonnegative_int, default=2026)
    crypto_lab.add_argument("--sparsity", choices=SPARSITY_LEVELS, default="moderate")
    crypto_lab.add_argument("--obfuscation", choices=OBFUSCATION_LEVELS, default="none")
    crypto_lab.add_argument(
        "--generator", choices=GENERATOR_FAMILIES, default=GENERATOR_FAMILIES[0]
    )

    factorial = subparsers.add_parser(
        "run-factorial", help="resume the registered corrected matched-generator factorial"
    )
    factorial.add_argument("--seeds", type=_positive_int, default=100)
    factorial.add_argument("--epochs", type=_positive_int, default=60)
    factorial.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    factorial.add_argument(
        "--output", type=Path, default=Path("data/results/factorial_results.csv")
    )

    analyze = subparsers.add_parser(
        "analyze", help="verify and analyze corrected factorial results"
    )
    analyze.add_argument(
        "--results", type=Path, default=Path("data/results/factorial_results.csv")
    )
    analyze.add_argument(
        "--summary", type=Path, default=Path("data/empirical_summary.json")
    )
    analyze.add_argument("--seeds", type=_positive_int, default=100)

    robustness = subparsers.add_parser(
        "run-robustness", help="resume the generator-misspecification experiment"
    )
    robustness.add_argument("--seeds", type=_positive_int, default=30)
    robustness.add_argument("--epochs", type=_positive_int, default=60)
    robustness.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    robustness.add_argument(
        "--output", type=Path, default=Path("data/results/robustness_results.csv")
    )
    robustness.add_argument(
        "--generators", nargs="+", choices=GENERATOR_FAMILIES, default=GENERATOR_FAMILIES
    )

    analyze_robust = subparsers.add_parser(
        "analyze-robustness", help="verify and analyze complete robustness results"
    )
    analyze_robust.add_argument(
        "--results", type=Path, default=Path("data/results/robustness_results.csv")
    )
    analyze_robust.add_argument(
        "--summary", type=Path, default=Path("data/robustness_summary.json")
    )
    analyze_robust.add_argument("--seeds", type=_positive_int, default=30)
    analyze_robust.add_argument(
        "--generators", nargs="+", choices=GENERATOR_FAMILIES, default=GENERATOR_FAMILIES
    )
    return parser


def _fit_and_summarize(
    batch: EventBatch, epochs: int, device_name: str, seed: int
) -> dict[str, object]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    torch.manual_seed(seed)
    model = HawkesFlowDSBM().to(device)
    fit = model.fit_batch(batch.to(device), epochs=epochs, seed=seed)
    return {
        "events": batch.num_events,
        "nodes": batch.num_nodes,
        "device": str(device),
        "latency_ms": fit.latency_ms,
        "final_elbo_per_event": fit.final_elbo_per_event,
        "occupied_blocks": fit.occupied_blocks,
        "claim_boundary": (
            "Controlled synthetic topology posterior only; no inference about real entities, "
            "intent, content, or attribution."
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "simulation-registry":
        result: object = {"generators": simulation_registry()}
    elif args.command == "simulate":
        simulated = simulate_network(
            args.seed,
            args.sparsity,
            args.obfuscation,
            generator=args.generator,
        )
        result = simulated.condition | {
            "inference": _fit_and_summarize(
                simulated.batch, args.epochs, args.device, args.seed
            )
        }
    elif args.command == "crypto-lab":
        simulated = simulate_network(
            args.seed,
            args.sparsity,
            args.obfuscation,
            generator=args.generator,
        )
        summary = run_synthetic_crypto_lab(simulated.batch, seed=args.seed)
        result = simulated.condition | {
            "synthetic_cryptography": asdict(summary),
            "all_authorized_round_trips_exact": summary.all_authorized_round_trips_exact,
            "key_withheld_control_passed": summary.key_withheld_control.passed,
            "claim_boundary": (
                "Experiment-owned synthetic plaintext and temporary key only; no key recovery, "
                "external traffic, TLS, Wi-Fi, identity, intent, or attribution claim."
            ),
        }
    elif args.command == "run-factorial":
        result = {
            "rows": run_factorial(
                args.output, args.seeds, args.epochs, device=args.device
            ),
            "output": str(args.output),
        }
    elif args.command == "analyze":
        result = analyze_factorial(args.results, args.summary, expected_seeds=args.seeds)
    elif args.command == "run-robustness":
        result = {
            "rows": run_robustness(
                args.output,
                args.seeds,
                args.epochs,
                device=args.device,
                generators=tuple(args.generators),
            ),
            "output": str(args.output),
        }
    else:
        result = analyze_robustness(
            args.results,
            args.summary,
            expected_seeds=args.seeds,
            generators=tuple(args.generators),
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
