# Corrected Factorial and Robustness Protocol

## Primary factorial

The primary objective is to estimate the architecture effect and its interactions with
synthetic perturbation and graph sparsity.

- Blocks: 100 simulator seeds (`2026` through `2125`).
- Cells per block: 18.
- Total architecture runs: 1,800.
- Generator: matched `hawkes_exponential` only.
- Within-block order: deterministic pseudorandom permutation.

Perturbations are `none`, deterministic packet-size `padding`, and stochastic timing
`jitter`. Sparsity is `dense`, `moderate`, or `sparse`. The truth graph is unchanged by the
perturbation.

The analyzer requires all expected cells. Architecture comparisons are paired by seed. Each
endpoint's family of nine comparisons reports a paired *t* estimate, unadjusted 95% interval,
two-sided raw *p* value, and Holm-adjusted *p* value. The factorial model first attempts a
seed random intercept plus architecture slope, then a random intercept only, and finally the
same 18-column fixed design with seed-clustered standard errors. Every attempt and rejection
reason is retained. Latency is analyzed on the natural-log scale.

The source CSV is hashed before and after analysis. The analyzer aborts if it changes and
publishes the verified SHA-256 in the summary. It also requires exact field order, the exact
registered key set, complete status, and valid numeric ranges. The corrected 1,800-cell run
is pending. Historical aggregates are quarantined because their source CSV is absent and
their declared seed did not cover model initialization.

## Robustness experiment

The extension crosses the same two architectures, three perturbations, and three sparsity
levels with five registered event laws. The default 30 seeds create 2,700 cells. It writes a
separate schema and output file, so the primary design remains intact.

The matched family estimates performance under model alignment. Mixture-Hawkes,
piecewise-Cox, and gamma-renewal families test three distinct departures from the fitted
likelihood. The independent-null family makes truth adjacency unrelated to observations and
serves as a no-signal calibration check.

Analysis requires the exact generator-by-seed-by-condition key set. Within every generator
and condition, the architecture difference is paired by seed. Holm correction controls
familywise error across all 45 generator-by-condition contrasts separately for each of link
AUC, binary link log score, and latency per event.

## Interpretation

Both studies measure recovery of simulated truth. They do not validate claims about real
communications, identity, intent, attribution, encrypted content, or operational use.
