# Claim Boundary

## Supported by the repository

- The exponential Hawkes event term and full-window compensator are implemented and tested.
- Equal-time events do not excite one another; finite-window offspring are censored once.
- The MAF transform is invertible to numerical tolerance and contributes its log Jacobian.
- The controlled-truth `2 × 3 × 3 × 100` experiment is specified and restartable.
- Analyzer v1.2.0 validates exact keys, schema, numeric ranges, source hash, paired inference,
  endpoint-wise Holm correction, and the mixed-model-to-clustered-OLS fallback hierarchy.
- Five registered simulation families support matched, misspecified, and no-signal studies.
- The isolated cryptography component performs authenticated AES-GCM round trips on generated
  plaintext with a temporary experiment-owned key and verifies byte-exact reconstruction.
- A key-withheld software control invokes no decryption and emits no plaintext.

The historical 1,800-row aggregate and its SHA-256 are retained for traceability only. The
source result CSV is absent, model construction preceded cell seeding, and version 1.2.0
corrects the simulation window. It is quarantined and supports no current architecture
ranking or performance conclusion.

## Supported after the registered studies complete

- Architecture contrasts within each declared generator, perturbation, and sparsity cell.
- Sensitivity of link metrics and latency to specific simulated forms of misspecification.
- Calibration behavior under the declared independent no-signal negative control.

## Not supported by this design

- Claims about real network traffic, devices, people, groups, or organizations.
- Breaking encryption, recovering keys, or recovering content without an authorized
  experiment-owned key.
- Payload or message claims about any external communication.
- Human identity, intent, attribution, or command-and-control identification.
- Operational, field, intelligence, or SIGINT performance.
- Generalization to event laws or perturbations absent from the registry.
- Treating importance ESS as MCMC ESS or reporting rank-normalized R-hat for VI iterates.

The defensible description is **simulation-only latent-topology recovery plus authorized
synthetic cryptographic round-trip testing under controlled truth**.
