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

No cell is analyzed until all expected cells exist. Architecture is sum-coded; ordered
three-level factors use orthogonal linear and quadratic contrasts. The condition-specific
architecture comparisons are paired by seed. Each endpoint's family of nine comparisons
reports a paired *t* estimate, an unadjusted 95% confidence interval, a two-sided raw
*p* value, and a Holm-adjusted *p* value controlling familywise error across the nine
conditions.

The full factorial model first attempts a seed random intercept plus architecture slope.
A mixed-model attempt is admissible only when the optimizer converges without a Statsmodels
convergence warning and its fixed-effect estimates, standard errors, and covariance matrix
are finite and valid. An inadmissible fit falls back first to a seed random-intercept model
and then, if necessary, to the same 18-column factorial fixed design with seed-clustered
standard errors. Every attempt and rejection reason is retained in the summary. Latency is
analyzed on the natural-log scale. Failed experimental cells remain in the denominator and
are reported as a separate reliability endpoint rather than silently discarded.

The analyzer computes the source CSV's SHA-256 digest before and after reading it, aborts if
the file changes during analysis, and writes the verified digest into the public summary.
This binds the aggregate inference to the exact private result artifact without publishing
the cell-level file.

## Live external lane

Authorized local captures may be processed after the controlled protocol is frozen. Because endpoint truth is absent, the live lane reports acquisition volume, compute latency, occupied blocks, posterior uncertainty, sensitivity across windows, and pseudonym stability. It cannot validate link AUC or attribution.
