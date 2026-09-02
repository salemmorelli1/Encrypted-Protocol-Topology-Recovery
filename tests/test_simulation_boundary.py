import argparse
import importlib.util
from pathlib import Path

from encrypted_topology.cli import build_parser


def test_removed_collection_modules_are_not_importable():
    for module in ("capture", "io", "privacy"):
        assert importlib.util.find_spec(f"encrypted_topology.{module}") is None


def test_cli_exposes_only_simulation_and_analysis_commands():
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    assert set(subparsers.choices) == {
        "simulation-registry",
        "simulate",
        "crypto-lab",
        "run-factorial",
        "analyze",
        "run-robustness",
        "analyze-robustness",
    }


def test_runtime_dependencies_exclude_packet_collection_libraries():
    project = Path(__file__).parents[1]
    configuration = (project / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "scapy" not in configuration
    assert "pyshark" not in configuration
    assert "pcap" not in configuration
