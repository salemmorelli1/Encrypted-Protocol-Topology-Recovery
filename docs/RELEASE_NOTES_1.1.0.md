# Release 1.1.0

> Historical note: version 1.2.0 quarantines the numerical evidence described below because
> the source CSV is absent and the declared cell seed did not cover model initialization.
> These statements are not current performance claims.

This release makes the project simulation-only and adds a separate misspecification and
negative-control research design. It preserves the completed 1,800-cell factorial and the
v1.0.2 analyzer's inferential corrections.

## Boundary changes

- Removed packet capture, interface discovery, external-trace import, pseudonymization, and
  live-inference modules and commands.
- Removed the packet-capture runtime dependency.
- Added tests that lock the six-command simulation-only CLI and removed module boundary.
- Replaced the capture protocol with a simulation-only protocol.

## Statistical additions

- Added a documented five-family simulation registry.
- Added mixture-Hawkes, piecewise-Cox, gamma-renewal, and independent-null generators.
- Added a restartable 2,700-cell default robustness experiment in a separate CSV schema.
- Added an exact-key analyzer with source SHA-256 verification, 135 paired contrasts, and
  Holm adjustment across 45 generator-by-condition comparisons within each endpoint.

## Compatibility

The primary `hawkes_exponential` random-number path and frozen factorial output schema remain
unchanged. Version 1.1.0 does not rewrite or rerun any of the 1,800 primary cells.
