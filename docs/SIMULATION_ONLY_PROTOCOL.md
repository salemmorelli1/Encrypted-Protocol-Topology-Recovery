# Simulation-Only Protocol

## Runtime boundary

Version 1.1.0 accepts no network interface, packet file, endpoint list, external event stream,
or address map. The package contains no acquisition backend, packet parser, live inference
command, or pseudonymization key. All runtime observations are synthetic tensors generated
in memory from an explicit integer seed.

The CLI is restricted to six commands: `simulation-registry`, `simulate`, `run-factorial`,
`analyze`, `run-robustness`, and `analyze-robustness`. Tests lock that command set, assert
that the former collection modules are not importable, and reject packet-capture libraries
in runtime dependencies.

## Generator registry

Every generator returns the same controlled object:

- a time-ordered synthetic event batch;
- synthetic node labels with no external correspondence;
- a binary directed truth adjacency matrix;
- block memberships and block connectivity; and
- a condition record containing the seed, event law, perturbation, and design parameters.

The registry contains one matched exponential Hawkes law, three deliberately misspecified
signal-bearing laws, and one independent no-signal negative control. Registry metadata states
whether topology affects the observation law and how that law relates to the fitted model.

## Reproducibility

The primary generator remains `hawkes_exponential`; its random-number order and parameter
defaults are unchanged so the frozen 1,800-cell experiment remains reproducible. The
robustness experiment is a separate output schema and never appends to or rewrites the
frozen factorial table.

Each experiment is restartable. Completed cells are keyed by all design coordinates and
written atomically through a temporary file. An analyzer verifies the source SHA-256 before
and after reading, requires the exact completed design, and publishes aggregate results only
after the lock passes.

## Data rule

Do not adapt this repository to ingest real traffic for this study. Suitable extensions are
new stochastic processes, graph priors, synthetic perturbations, calibration diagnostics,
simulation-based calibration, posterior-predictive checks, and compute benchmarks. External
validity questions should be addressed with published aggregate benchmarks or a separately
governed project, not by adding collection code here.
