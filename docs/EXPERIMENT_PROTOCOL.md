# Frozen Experiment Protocol

## Objective

Estimate the architecture effect and its interactions with obfuscation and graph sparsity without reusing live-network observations as truth.

## Allocation

- Blocks: 100 independent simulator seeds (`2026` through `2125`).
- Cells per block: 18.
- Total architecture runs: 1,800.
- Within-block cell order: deterministic pseudorandom permutation keyed by seed.

## Perturbations

- `none`: event times and empirical size marks unchanged.
- `padding`: observed sizes mapped to a common upper-size profile while time remains unchanged.
- `jitter`: bounded stochastic timing displacement followed by stable reordering; the latent truth graph is unchanged.

## Analysis

No cell is analyzed until all expected cells exist. Architecture is sum-coded; ordered three-level factors use orthogonal linear and quadratic contrasts. Report estimates with 95% confidence intervals and multiplicity-controlled interaction follow-ups. Latency is log-transformed if right skew is material. Failed fits remain in the denominator and are reported as a separate reliability endpoint rather than silently discarded.

## Live external lane

Authorized local captures may be processed after the controlled protocol is frozen. Because endpoint truth is absent, the live lane reports acquisition volume, compute latency, occupied blocks, posterior uncertainty, sensitivity across windows, and pseudonym stability. It cannot validate link AUC or attribution.
