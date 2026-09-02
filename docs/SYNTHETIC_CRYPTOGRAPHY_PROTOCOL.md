# Synthetic Cryptography Protocol

## Purpose

This protocol adds an authorized cryptography control to the simulation-only topology
laboratory. It demonstrates the difference between recovering synthetic network structure
from metadata and reconstructing synthetic content with an experiment-owned key. It does
not attempt key recovery, cryptanalysis of real protocols, or interception.

## Component separation

The experiment follows seven steps:

1. Generate synthetic plaintext messages aligned to simulated network events.
2. Encrypt each message with AES-256-GCM and a fresh temporary experiment key.
3. Supply the authorized key, 96-bit nonce, ciphertext, associated data, and 128-bit
   authentication tag only to the decryption component.
4. Require every authenticated plaintext reconstruction to match its synthetic original
   byte for byte.
5. Restrict topology recovery to event time, size, source index, destination index, and
   synthetic node label.
6. Run a key-withheld negative control that invokes no decryption interface and produces
   zero plaintext outputs.
7. Reserve deliberately weak toy ciphers for separately reviewed future experiments with
   tiny artificial key spaces; TLS, Wi-Fi, and real communications are excluded.

`MetadataOnlyView` makes the separation inspectable in code. It has no plaintext,
ciphertext, nonce, authentication tag, or key field. `SyntheticCryptoLabSummary` similarly
returns counts and pass/fail indicators without returning or persisting secret material.

## Cryptographic rules

- Keys are generated in memory for one experiment run and are not written to disk.
- A unique 96-bit nonce is required for every message encrypted under one key.
- Authentication-tag failure aborts decryption; unauthenticated plaintext is never returned.
- The public CLI accepts only simulation parameters and cannot accept an interface, packet
  file, ciphertext file, endpoint, external key, or external message stream.
- The implementation uses the maintained `cryptography` AES-GCM primitive rather than a
  custom cipher implementation.

Python does not guarantee immediate memory zeroization. Therefore, “temporary” means that
the key is scoped to one in-memory run and is neither returned in the summary nor persisted,
not that physical memory erasure is formally verified.

## Interpretation of controls

The authorized round trip validates implementation correctness: given the experiment-owned
key and valid authentication material, every synthetic plaintext must be reconstructed
exactly. The key-withheld condition validates the software boundary: without a key, the
system performs no decryption and emits no plaintext.

This negative control is not offered as empirical proof of universal cryptographic security.
It demonstrates that the topology pipeline does not silently consume content or cryptographic
secrets and that machine learning is not treated as a substitute for an AES-GCM key.

## Safe conditions

- All data, devices, identities, events, messages, and keys are simulated.
- Runtime inputs are generated and used solely inside the controlled experiment.
- Any future system testing requires explicit written authorization defining systems,
  methods, dates, data handling, and reporting.
- No other person’s communications may be intercepted.
- No test may operate outside its approved scope.

These are research-engineering controls, not legal advice. Any proposal involving real
systems or communications belongs in a separately governed project reviewed by the system
owner and qualified legal and institutional authorities before work begins.

## Reproduction

Run the isolated synthetic cryptography experiment with:

```bash
encrypted-topology crypto-lab \
  --generator hawkes_exponential \
  --seed 2026 \
  --sparsity moderate \
  --obfuscation none
```

The JSON result reports exact authorized matches, authenticated decryptions, nonce
uniqueness, the topology feature boundary, and the key-withheld control. It contains no key,
plaintext, ciphertext, nonce, or authentication tag.
