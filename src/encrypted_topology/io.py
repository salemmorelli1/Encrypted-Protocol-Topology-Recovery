"""Load verified capture blocks without exposing local pseudonym keys."""

from __future__ import annotations

from pathlib import Path

from .events import MetadataEvent, events_to_batch
from .privacy import read_verified_metadata_block


def load_capture(path: Path):
    body = read_verified_metadata_block(path)
    events = [MetadataEvent(**event) for event in body["events"]]
    return events_to_batch(events)

