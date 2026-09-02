# Claim Boundary

## Supported by the repository

- The exponential Hawkes event term and compensator are implemented and tested.
- The MAF transform is invertible to numerical tolerance and contributes its log Jacobian.
- The controlled-truth `2 × 3 × 3 × 100` experiment is fully specified and restartable.
- The completed factorial source is bound to SHA-256
  `d19433ff32f2222655f5408810b9ed6fd59fa01ea461e05e3f1aa5a59977309a`.
- Analyzer v1.0.2 reports 27 paired contrasts, endpoint-wise Holm correction, and a recorded
  mixed-model-to-clustered-OLS fallback hierarchy without changing cell results.
- Five registered simulation families support matched, misspecified, and no-signal studies.
- The isolated cryptography component performs authenticated AES-GCM round trips on generated
  plaintext with a temporary experiment-owned key and verifies byte-exact reconstruction.
- A key-withheld software control invokes no decryption and emits no plaintext.

## Supported after the separate robustness study completes

- Architecture contrasts within each declared generator, perturbation, and sparsity cell.
- Sensitivity of link metrics and latency to specific simulated forms of misspecification.
- Calibration behavior under the declared independent no-signal negative control.

## Not supported by this design

- Claims about real network traffic, devices, people, groups, or organizations.
- Breaking encryption, recovering keys, or recovering content without an authorized
  experiment-owned key.
- Payload or message claims about any external communication.
- Human identity, intent, or attribution.
- Command-and-control identification or attribution.
- Operational, field, intelligence, or SIGINT performance.
- Generalization to event laws or perturbations absent from the registry.
- Treating importance ESS as MCMC ESS or reporting rank-normalized R-hat for VI iterates.

The defensible description is **simulation-only latent-topology recovery plus authorized
synthetic cryptographic round-trip testing under controlled truth**.
