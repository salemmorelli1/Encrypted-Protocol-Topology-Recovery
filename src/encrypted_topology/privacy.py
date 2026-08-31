"""Local-only pseudonymization and atomic metadata storage."""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import os
import secrets
import tempfile
from collections.abc import Iterable
from pathlib import Path

from .events import MetadataEvent

SCHEMA = "encrypted-topology-metadata-v1"


def load_or_create_salt(path: Path) -> bytes:
    """Load or create a local 256-bit HMAC key; the key is never written to Git."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        value = path.read_bytes()
        if len(value) != 32:
            raise ValueError("pseudonym salt must contain exactly 32 bytes")
        return value
    value = secrets.token_bytes(32)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        os.write(descriptor, value)
        os.close(descriptor)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise
    return value


def pseudonymize_endpoint(endpoint: str, salt: bytes) -> str:
    """Return a stable local pseudonym without preserving an address prefix."""

    normalized = endpoint.strip().lower().encode("utf-8")
    return hmac.new(salt, normalized, hashlib.sha256).hexdigest()[:24]


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_metadata_block(
    destination: Path,
    events: Iterable[MetadataEvent],
    capture_metadata: dict[str, object],
) -> dict[str, object]:
    """Atomically persist payload-free events with a content checksum."""

    materialized = [event.to_dict() for event in events]
    if not materialized:
        raise ValueError("refusing to write an empty capture")
    body: dict[str, object] = {
        "schema": SCHEMA,
        "capture": capture_metadata,
        "events": materialized,
    }
    body["sha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(suffix=".json.gz", dir=destination.parent)
    os.close(descriptor)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump(body, stream, separators=(",", ":"))
        os.replace(temporary, destination)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return body


def read_verified_metadata_block(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        body = json.load(stream)
    expected = body.pop("sha256", None)
    actual = hashlib.sha256(_canonical_json(body)).hexdigest()
    body["sha256"] = expected
    if not isinstance(expected, str) or not hmac.compare_digest(expected, actual):
        raise ValueError(f"metadata checksum mismatch: {path}")
    forbidden = {"payload", "raw_bytes", "ip_address", "mac_address", "port"}
    if any(forbidden.intersection(event) for event in body.get("events", [])):
        raise ValueError("forbidden packet content found in metadata block")
    return body

