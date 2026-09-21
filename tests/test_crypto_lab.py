from dataclasses import fields, replace

import pytest
import torch
from cryptography.exceptions import InvalidTag

from encrypted_topology.crypto_lab import (
    MetadataOnlyView,
    MissingExperimentKey,
    SyntheticPlaintextMessage,
    decrypt_synthetic_messages,
    encrypt_synthetic_messages,
    generate_synthetic_plaintexts,
    metadata_only_view,
    run_key_withheld_control,
    run_synthetic_crypto_lab,
)
from encrypted_topology.events import EventBatch


def _batch() -> EventBatch:
    return EventBatch(
        times=torch.tensor([0.1, 0.4, 0.8, 1.3]),
        sizes=torch.tensor([72.0, 96.0, 128.0, 80.0]),
        src=torch.tensor([0, 1, 2, 0]),
        dst=torch.tensor([1, 2, 0, 2]),
        node_keys=("sim-node-0", "sim-node-1", "sim-node-2"),
        observation_horizon=2.0,
    )


def _nonce_counter():
    counter = 0

    def next_nonce(length: int) -> bytes:
        nonlocal counter
        assert length == 12
        counter += 1
        return counter.to_bytes(length, "big")

    return next_nonce


def test_authorized_aes_gcm_round_trip_is_exact_and_authenticated():
    batch = _batch()
    plaintexts = generate_synthetic_plaintexts(batch, seed=2026)
    key = bytes(range(32))
    encrypted = encrypt_synthetic_messages(
        plaintexts, key=key, nonce_factory=_nonce_counter()
    )
    recovered = decrypt_synthetic_messages(encrypted, key=key)

    assert recovered == tuple(message.plaintext for message in plaintexts)
    assert len({message.nonce for message in encrypted}) == batch.num_events
    assert all(len(message.authentication_tag) == 16 for message in encrypted)


def test_modified_authentication_tag_is_rejected():
    plaintexts = generate_synthetic_plaintexts(_batch(), seed=2026)
    key = bytes(range(32))
    encrypted = encrypt_synthetic_messages(
        plaintexts, key=key, nonce_factory=_nonce_counter()
    )
    first = encrypted[0]
    modified = bytes([first.authentication_tag[0] ^ 1]) + first.authentication_tag[1:]
    tampered = (replace(first, authentication_tag=modified), *encrypted[1:])

    with pytest.raises(InvalidTag):
        decrypt_synthetic_messages(tampered, key=key)


def test_decryption_refuses_to_run_when_key_is_withheld():
    plaintexts = generate_synthetic_plaintexts(_batch(), seed=2026)
    encrypted = encrypt_synthetic_messages(
        plaintexts, key=bytes(range(32)), nonce_factory=_nonce_counter()
    )

    with pytest.raises(MissingExperimentKey):
        decrypt_synthetic_messages(encrypted, key=None)


def test_metadata_boundary_contains_no_plaintext_ciphertext_or_key_fields():
    view = metadata_only_view(_batch())
    names = {field.name for field in fields(MetadataOnlyView)}

    assert names == {
        "times",
        "sizes",
        "src",
        "dst",
        "node_keys",
        "observation_horizon",
    }
    assert not names.intersection({"plaintext", "ciphertext", "key", "nonce", "tag"})
    assert view.num_events == 4


def test_key_withheld_control_emits_no_plaintext():
    result = run_key_withheld_control(metadata_only_view(_batch()))

    assert result.passed
    assert result.observed_events == 4
    assert result.recovered_plaintexts == 0
    assert not result.key_provided
    assert not result.decryption_attempted


def test_end_to_end_lab_summary_contains_no_secret_material():
    result = run_synthetic_crypto_lab(_batch(), seed=2026)

    assert result.all_authorized_round_trips_exact
    assert result.key_withheld_control.passed
    assert result.topology_feature_names == (
        "times",
        "sizes",
        "src",
        "dst",
        "node_keys",
        "observation_horizon",
    )
    assert not result.experiment_key_persisted
    assert not hasattr(result, "key")
    assert not hasattr(result, "plaintexts")


def test_reused_nonce_is_rejected():
    plaintexts = generate_synthetic_plaintexts(_batch(), seed=2026)

    with pytest.raises(ValueError, match="nonce reuse"):
        encrypt_synthetic_messages(
            plaintexts,
            key=bytes(range(32)),
            nonce_factory=lambda length: b"0" * length,
        )


def test_unmarked_plaintext_is_rejected_by_the_simulation_boundary():
    with pytest.raises(ValueError, match="synthetic-only marker"):
        SyntheticPlaintextMessage(
            event_id=0,
            plaintext=b"external message",
            associated_data=b"encrypted-topology:synthetic-event:0",
        )
