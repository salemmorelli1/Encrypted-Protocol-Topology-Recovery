"""In-memory cryptography controls for wholly synthetic experiment events.

This module performs authorized AES-GCM round trips with experiment-owned keys. It has no
network, packet-file, external-ciphertext, key-recovery, or persistent-key interface.
"""

from __future__ import annotations

import random
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .events import EventBatch

AES_GCM_KEY_BYTES = 32
AES_GCM_NONCE_BYTES = 12
AES_GCM_TAG_BYTES = 16
TOPOLOGY_FEATURE_NAMES = ("times", "sizes", "src", "dst", "node_keys")
PLAINTEXT_MARKER = b"SYNTHETIC|"
ASSOCIATED_DATA_PREFIX = b"encrypted-topology:synthetic-event:"


class MissingExperimentKey(ValueError):
    """Raised when authorized decryption is requested without an experiment key."""


@dataclass(frozen=True)
class MetadataOnlyView:
    """The complete information boundary available to topology recovery."""

    times: tuple[float, ...]
    sizes: tuple[float, ...]
    src: tuple[int, ...]
    dst: tuple[int, ...]
    node_keys: tuple[str, ...]

    @property
    def num_events(self) -> int:
        return len(self.times)


@dataclass(frozen=True)
class SyntheticPlaintextMessage:
    """A generated message with no correspondence to an external communication."""

    event_id: int
    plaintext: bytes
    associated_data: bytes

    def __post_init__(self) -> None:
        if self.event_id < 0 or not self.plaintext or not self.associated_data:
            raise ValueError("synthetic messages require an id, plaintext, and associated data")
        if not self.plaintext.startswith(PLAINTEXT_MARKER):
            raise ValueError("plaintext must carry the synthetic-only marker")
        if not self.associated_data.startswith(ASSOCIATED_DATA_PREFIX):
            raise ValueError("associated data must carry the synthetic-only prefix")


@dataclass(frozen=True)
class EncryptedSyntheticMessage:
    """Separated AES-GCM ciphertext and authentication material for one synthetic event."""

    event_id: int
    nonce: bytes
    ciphertext: bytes
    authentication_tag: bytes
    associated_data: bytes

    def __post_init__(self) -> None:
        if self.event_id < 0:
            raise ValueError("event_id must be nonnegative")
        if len(self.nonce) != AES_GCM_NONCE_BYTES:
            raise ValueError("AES-GCM experiment nonces must contain exactly 12 bytes")
        if len(self.authentication_tag) != AES_GCM_TAG_BYTES:
            raise ValueError("AES-GCM authentication tags must contain exactly 16 bytes")
        if not self.ciphertext or not self.associated_data:
            raise ValueError("ciphertext and associated data cannot be empty")
        if not self.associated_data.startswith(ASSOCIATED_DATA_PREFIX):
            raise ValueError("associated data must carry the synthetic-only prefix")


@dataclass(frozen=True)
class KeyWithheldControl:
    """System-level negative control in which no decryption interface is invoked."""

    observed_events: int
    key_provided: bool = False
    decryption_attempted: bool = False
    recovered_plaintexts: int = 0

    @property
    def passed(self) -> bool:
        return (
            self.observed_events > 0
            and not self.key_provided
            and not self.decryption_attempted
            and self.recovered_plaintexts == 0
        )


@dataclass(frozen=True)
class SyntheticCryptoLabSummary:
    """Non-secret summary of one authorized synthetic encryption experiment."""

    total_messages: int
    exact_plaintext_matches: int
    authenticated_decryptions: int
    unique_nonces: int
    topology_feature_names: tuple[str, ...]
    key_withheld_control: KeyWithheldControl
    experiment_key_persisted: bool = False

    @property
    def all_authorized_round_trips_exact(self) -> bool:
        return (
            self.total_messages > 0
            and self.exact_plaintext_matches == self.total_messages
            and self.authenticated_decryptions == self.total_messages
            and self.unique_nonces == self.total_messages
        )


def metadata_only_view(batch: EventBatch) -> MetadataOnlyView:
    """Copy an event batch into the payload-free topology boundary."""

    return MetadataOnlyView(
        times=tuple(float(value) for value in batch.times.detach().cpu().tolist()),
        sizes=tuple(float(value) for value in batch.sizes.detach().cpu().tolist()),
        src=tuple(int(value) for value in batch.src.detach().cpu().tolist()),
        dst=tuple(int(value) for value in batch.dst.detach().cpu().tolist()),
        node_keys=batch.node_keys,
    )


def generate_synthetic_plaintexts(
    batch: EventBatch, *, seed: int
) -> tuple[SyntheticPlaintextMessage, ...]:
    """Generate deterministic, experiment-only plaintext aligned to synthetic events."""

    view = metadata_only_view(batch)
    rng = random.Random(seed)
    messages: list[SyntheticPlaintextMessage] = []
    alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    for event_id, (size, src, dst) in enumerate(zip(view.sizes, view.src, view.dst, strict=True)):
        target_size = max(len(PLAINTEXT_MARKER), round(size))
        header = f"SYNTHETIC|event={event_id}|src={src}|dst={dst}|".encode()
        filler_size = max(0, target_size - len(header))
        filler = bytes(alphabet[rng.randrange(len(alphabet))] for _ in range(filler_size))
        plaintext = (header + filler)[:target_size]
        associated_data = f"encrypted-topology:synthetic-event:{event_id}".encode()
        messages.append(SyntheticPlaintextMessage(event_id, plaintext, associated_data))

    return tuple(messages)


def create_temporary_experiment_key() -> bytes:
    """Create a fresh 256-bit AES key for a single in-memory experiment run."""

    return AESGCM.generate_key(bit_length=AES_GCM_KEY_BYTES * 8)


def encrypt_synthetic_messages(
    messages: Sequence[SyntheticPlaintextMessage],
    *,
    key: bytes,
    nonce_factory: Callable[[int], bytes] = secrets.token_bytes,
) -> tuple[EncryptedSyntheticMessage, ...]:
    """Encrypt generated messages with an explicitly supplied experiment-owned key."""

    if not messages:
        raise ValueError("at least one synthetic message is required")
    cipher = AESGCM(key)
    encrypted: list[EncryptedSyntheticMessage] = []
    used_nonces: set[bytes] = set()

    for message in messages:
        nonce = nonce_factory(AES_GCM_NONCE_BYTES)
        if len(nonce) != AES_GCM_NONCE_BYTES:
            raise ValueError("nonce_factory must return exactly 12 bytes")
        if nonce in used_nonces:
            raise ValueError("AES-GCM nonce reuse is forbidden within an experiment key")
        used_nonces.add(nonce)
        sealed = cipher.encrypt(nonce, message.plaintext, message.associated_data)
        encrypted.append(
            EncryptedSyntheticMessage(
                event_id=message.event_id,
                nonce=nonce,
                ciphertext=sealed[:-AES_GCM_TAG_BYTES],
                authentication_tag=sealed[-AES_GCM_TAG_BYTES:],
                associated_data=message.associated_data,
            )
        )

    return tuple(encrypted)


def decrypt_synthetic_messages(
    messages: Sequence[EncryptedSyntheticMessage], *, key: bytes | None
) -> tuple[bytes, ...]:
    """Perform authenticated decryption only when an authorized key is supplied."""

    if key is None:
        raise MissingExperimentKey("authorized AES-GCM decryption requires the experiment key")
    if not messages:
        raise ValueError("at least one encrypted synthetic message is required")
    cipher = AESGCM(key)
    return tuple(
        cipher.decrypt(
            message.nonce,
            message.ciphertext + message.authentication_tag,
            message.associated_data,
        )
        for message in messages
    )


def run_key_withheld_control(view: MetadataOnlyView) -> KeyWithheldControl:
    """Return a blinded control that emits no plaintext and never calls decryption."""

    return KeyWithheldControl(observed_events=view.num_events)


def run_synthetic_crypto_lab(batch: EventBatch, *, seed: int) -> SyntheticCryptoLabSummary:
    """Run an authorized AES-GCM round trip and a separate key-withheld control."""

    view = metadata_only_view(batch)
    plaintexts = generate_synthetic_plaintexts(batch, seed=seed)
    temporary_key = create_temporary_experiment_key()
    encrypted = encrypt_synthetic_messages(plaintexts, key=temporary_key)
    recovered = decrypt_synthetic_messages(encrypted, key=temporary_key)
    exact_matches = sum(
        original.plaintext == candidate
        for original, candidate in zip(plaintexts, recovered, strict=True)
    )
    withheld = run_key_withheld_control(view)

    return SyntheticCryptoLabSummary(
        total_messages=len(plaintexts),
        exact_plaintext_matches=exact_matches,
        authenticated_decryptions=len(recovered),
        unique_nonces=len({message.nonce for message in encrypted}),
        topology_feature_names=TOPOLOGY_FEATURE_NAMES,
        key_withheld_control=withheld,
    )
