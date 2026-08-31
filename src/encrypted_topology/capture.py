"""Authorized, non-promiscuous live metadata capture through Scapy or TShark."""

from __future__ import annotations

import csv
import queue
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .events import MetadataEvent
from .privacy import load_or_create_salt, pseudonymize_endpoint, write_metadata_block


class AuthorizationError(PermissionError):
    """Raised unless the operator explicitly confirms capture authorization."""


@dataclass(frozen=True)
class CapturePolicy:
    interface: str
    duration_seconds: float = 60.0
    packet_limit: int = 10_000
    capture_filter: str = "ip or ip6"
    backend: str = "scapy"
    authorized_capture: bool = False

    def __post_init__(self) -> None:
        if not self.authorized_capture:
            raise AuthorizationError(
                "capture requires --authorized-capture and permission for the selected interface"
            )
        if not self.interface.strip():
            raise ValueError("an explicit capture interface is required")
        if self.backend not in {"scapy", "tshark"}:
            raise ValueError("backend must be scapy or tshark")
        if self.duration_seconds <= 0 or self.packet_limit < 2:
            raise ValueError("capture duration and packet limit must be positive")


def _families(proto: int | None) -> tuple[str, str]:
    network = "ipv6" if proto == 6 else "ipv4"
    return network, "other"


def extract_scapy_event(packet: object, salt: bytes) -> MetadataEvent | None:
    """Extract only timestamp, frame size, pseudonyms, and coarse protocol families."""

    try:
        from scapy.layers.inet import IP, TCP, UDP
        from scapy.layers.inet6 import IPv6
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError("Scapy is required for the scapy capture backend") from exc

    if packet.haslayer(IP):
        layer = packet[IP]
        network_family = "ipv4"
    elif packet.haslayer(IPv6):
        layer = packet[IPv6]
        network_family = "ipv6"
    else:
        return None
    src, dst = str(layer.src), str(layer.dst)
    if not src or not dst or src == dst:
        return None
    transport_family = "tcp" if packet.haslayer(TCP) else "udp" if packet.haslayer(UDP) else "other"
    timestamp = float(getattr(packet, "time", time.time()))
    return MetadataEvent(
        timestamp=timestamp,
        packet_size=len(packet),
        src_hash=pseudonymize_endpoint(src, salt),
        dst_hash=pseudonymize_endpoint(dst, salt),
        network_family=network_family,
        transport_family=transport_family,
    )


def capture_scapy(policy: CapturePolicy, salt: bytes) -> list[MetadataEvent]:
    """Capture from one explicitly named interface with promiscuous mode disabled."""

    try:
        from scapy.all import sniff
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Scapy is not installed") from exc
    events: list[MetadataEvent] = []

    def consume(packet: object) -> None:
        event = extract_scapy_event(packet, salt)
        if event is not None:
            events.append(event)

    try:
        sniff(
            iface=policy.interface,
            filter=policy.capture_filter,
            prn=consume,
            store=False,
            timeout=policy.duration_seconds,
            count=policy.packet_limit,
            promisc=False,
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        if "winpcap is not installed" in message or "libpcap" in message:
            raise RuntimeError(
                "Scapy capture on Windows requires the Npcap packet-capture driver. "
                "Install or repair Npcap through the official Wireshark installer, "
                "reopen Git Bash, and retry on an authorized interface."
            ) from exc
        raise
    return events


def _parse_tshark_row(row: list[str], salt: bytes) -> MetadataEvent | None:
    if len(row) != 8:
        return None
    timestamp, length, ip_src, ip_dst, ipv6_src, ipv6_dst, tcp_flag, udp_flag = row
    src, dst = (ip_src, ip_dst) if ip_src and ip_dst else (ipv6_src, ipv6_dst)
    if not src or not dst or src == dst:
        return None
    return MetadataEvent(
        timestamp=float(timestamp),
        packet_size=int(length),
        src_hash=pseudonymize_endpoint(src, salt),
        dst_hash=pseudonymize_endpoint(dst, salt),
        network_family="ipv4" if ip_src else "ipv6",
        transport_family="tcp" if tcp_flag else "udp" if udp_flag else "other",
    )


def capture_tshark(policy: CapturePolicy, salt: bytes) -> list[MetadataEvent]:
    """Capture metadata fields from TShark stdout; no PCAP or payload is written."""

    executable = shutil.which("tshark")
    if executable is None:
        raise RuntimeError("tshark was not found; install Wireshark with TShark enabled")
    command = [
        executable,
        "-l",
        "-n",
        "-i",
        policy.interface,
        "-f",
        policy.capture_filter,
        "-a",
        f"duration:{policy.duration_seconds:g}",
        "-c",
        str(policy.packet_limit),
        "-T",
        "fields",
        "-E",
        "separator=/t",
        "-E",
        "occurrence=f",
    ]
    for field in (
        "frame.time_epoch",
        "frame.len",
        "ip.src",
        "ip.dst",
        "ipv6.src",
        "ipv6.dst",
        "tcp.flags",
        "udp.length",
    ):
        command.extend(["-e", field])
    completed = subprocess.run(command, capture_output=True, text=True, check=False, shell=False)
    if completed.returncode not in {0, 2}:  # TShark may return 2 when a timed capture closes.
        raise RuntimeError(f"tshark capture failed: {completed.stderr.strip()}")
    reader = csv.reader(completed.stdout.splitlines(), delimiter="\t")
    return [event for row in reader if (event := _parse_tshark_row(row, salt)) is not None]


def capture_to_file(
    policy: CapturePolicy,
    destination: Path,
    salt_path: Path = Path("data/private/pseudonym_salt.bin"),
) -> dict[str, object]:
    salt = load_or_create_salt(salt_path)
    started = time.time()
    events = capture_scapy(policy, salt) if policy.backend == "scapy" else capture_tshark(policy, salt)
    if len(events) < 2:
        raise RuntimeError("fewer than two eligible metadata events were captured")
    return write_metadata_block(
        destination,
        events,
        {
            "backend": policy.backend,
            "interface_recorded": False,
            "duration_seconds": policy.duration_seconds,
            "packet_limit": policy.packet_limit,
            "capture_filter": policy.capture_filter,
            "promiscuous": False,
            "started_at_epoch": started,
            "authorized_by_operator": True,
            "payload_retained": False,
        },
    )


class MetadataQueueStream:
    """Thread-safe iterator used to feed sanitized events into online windows."""

    def __init__(self, maxsize: int = 4096) -> None:
        self._queue: queue.Queue[MetadataEvent | None] = queue.Queue(maxsize=maxsize)

    def publish(self, event: MetadataEvent) -> None:
        self._queue.put(event)

    def close(self) -> None:
        self._queue.put(None)

    def __iter__(self) -> Iterator[MetadataEvent]:
        while True:
            value = self._queue.get()
            if value is None:
                return
            yield value
