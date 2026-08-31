# Capture and Privacy Protocol

## Authorization gate

The operator must name the interface and pass `--authorized-capture`. Capture is permitted only on a computer and network the operator owns or is expressly authorized to monitor. No attempt is made to bypass operating-system capture permissions or security policy.

## Data minimization

Persisted event rows contain exactly:

1. UTC epoch timestamp;
2. frame length in bytes;
3. source HMAC pseudonym;
4. destination HMAC pseudonym;
5. coarse network family (`ipv4` or `ipv6`); and
6. coarse transport family (`tcp`, `udp`, or `other`).

Payload bytes, DNS names, ports, TCP sequence numbers, raw IP addresses, MAC addresses, interface names, and PCAP files are excluded. A schema audit fails closed if a forbidden key appears.

## Pseudonymization

Each endpoint is transformed as

`HMAC-SHA256(local_salt, canonical_endpoint)`.

The 256-bit salt is generated once, stored under `data/private/`, permission-restricted where the operating system supports it, and excluded from Git. HMAC prevents the simple unsalted dictionary attack enabled by plain hashing. Pseudonymization reduces exposure but is not anonymization: relational patterns can remain identifying and must be treated as sensitive.

## Acquisition paths

- **Scapy:** calls `sniff(..., store=False, promisc=False)` and discards packets after field extraction.
- **TShark:** requests only named fields on stdout using `-T fields`; no `-w` output is used.

Each gzip JSON block includes a SHA-256 digest of its canonicalized content. Writes are atomic. Capture files and inferred local outputs are ignored by Git.

## Retention

Use the shortest retention compatible with the approved protocol. Publish only aggregate results and non-sensitive controlled simulations. Never commit local capture files, HMAC salts, credentials, or endpoint maps.
