from pathlib import Path

import pytest

from encrypted_topology.capture import AuthorizationError, CapturePolicy
from encrypted_topology.events import MetadataEvent
from encrypted_topology.privacy import (
    pseudonymize_endpoint,
    read_verified_metadata_block,
    write_metadata_block,
)


def test_capture_requires_explicit_authorization():
    with pytest.raises(AuthorizationError):
        CapturePolicy(interface="Ethernet")


def test_pseudonym_has_no_address_prefix_and_is_stable():
    salt = b"x" * 32
    first = pseudonymize_endpoint("192.0.2.4", salt)
    second = pseudonymize_endpoint("192.0.2.4", salt)
    assert first == second
    assert "192" not in first
    assert len(first) == 24


def test_metadata_round_trip_excludes_payload_and_addresses(tmp_path: Path):
    destination = tmp_path / "capture.json.gz"
    events = [
        MetadataEvent(1.0, 512, "a" * 24, "b" * 24, "ipv4", "tcp"),
        MetadataEvent(1.2, 256, "b" * 24, "a" * 24, "ipv4", "udp"),
    ]
    write_metadata_block(destination, events, {"payload_retained": False})
    body = read_verified_metadata_block(destination)
    serialized = str(body).lower()
    assert "payload': true" not in serialized
    assert "192.0.2" not in serialized
    assert len(body["events"]) == 2

