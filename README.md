# Encrypted Protocol Topology Recovery

[![Validate](https://github.com/salemmorelli1/Encrypted-Protocol-Topology-Recovery/actions/workflows/validate.yml/badge.svg)](https://github.com/salemmorelli1/Encrypted-Protocol-Topology-Recovery/actions/workflows/validate.yml)
[![Pages](https://github.com/salemmorelli1/Encrypted-Protocol-Topology-Recovery/actions/workflows/pages.yml/badge.svg)](https://github.com/salemmorelli1/Encrypted-Protocol-Topology-Recovery/actions/workflows/pages.yml)

This repository is a **simulation-only computational statistics laboratory** for latent
directed-topology recovery. It combines an exact exponential-kernel multivariate Hawkes
likelihood, an amortized dynamic stochastic block model, masked autoregressive flows, and a
static graph-autoencoder/Louvain comparator.

Version 1.1.0 has no packet-capture, interface-enumeration, external-trace import, address
pseudonymization, or live-inference path. Every event, node label, graph, mark, and truth
label used by the runtime is generated in memory from a declared seed.

## Scientific scope

The frozen primary study is a randomized complete-block `2 × 3 × 3 × 100` factorial:

- architecture: Hawkes-flow DSBM or static GAE–Louvain;
- perturbation: none, deterministic padding, or timing jitter;
- graph sparsity: dense, moderate, or sparse; and
- block: 100 seeds (`2026` through `2125`), for 1,800 runs.

The completed factorial artifact is identified by SHA-256
`d19433ff32f2222655f5408810b9ed6fd59fa01ea461e05e3f1aa5a59977309a`. Analyzer v1.0.2
does not modify those cell results: it verifies the source hash before and after reading,
reports 27 seed-paired contrasts, applies Holm adjustment across nine conditions within each
endpoint, and records diagnostic-aware mixed-model fallbacks through seed-clustered OLS.

Version 1.1.0 adds a separate robustness design across five event laws:

| Generator | Role | Truth signal |
|---|---|---|
| `hawkes_exponential` | matched primary generator | yes |
| `hawkes_mixture` | excitation-kernel misspecification | yes |
| `cox_piecewise` | shared nonstationary-rate misspecification | yes |
| `renewal_gamma` | non-Poisson renewal misspecification | yes |
| `independent_null` | calibrated no-signal negative control | no |

With the default 30 seeds, this adds 2,700 restartable runs. The robustness analyzer locks
the exact generator-by-condition design and applies Holm correction across all 45 contrasts
within each endpoint.

## Statistical model

For directed synthetic mark process (r), events \(\{(t_i,r_i)\}_{i=1}^{n}\) on \([0,T]\)
have conditional intensity

\[
\lambda_r(t\mid\mathcal H_t)=\mu_r+\sum_{j:t_j<t} A_{r_jr}\,\beta
e^{-\beta(t-t_j)}.
\]

The implemented log likelihood is

\[
\ell=\sum_i\log\lambda_{r_i}(t_i)-T\sum_r\mu_r-
\sum_j\sum_r A_{r_jr}\{1-e^{-\beta(T-t_j)}\}.
\]

The compensator is required for a valid point-process likelihood. Excitation is scaled below
the stability boundary. A neural encoder produces node-wise Gaussian base variables, an
invertible MAF transforms the variational law, and relaxed block assignments parameterize
directed link probabilities.

## Install and verify in Git Bash

```bash
cd "/c/Users/salem/GitHub/Encrypted-Protocol-Topology-Recovery"
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e ".[report,dev]"
python -m ruff check .
python -m pytest -q
python scripts/build_report.py
python scripts/verify_artifacts.py
```

## Run the simulations

Inspect the generator registry and run one smoke cell:

```bash
encrypted-topology simulation-registry
encrypted-topology simulate \
  --generator hawkes_mixture \
  --seed 2026 \
  --sparsity moderate \
  --obfuscation jitter \
  --epochs 20
```

Resume the frozen factorial or the separate robustness study:

```bash
encrypted-topology run-factorial --seeds 100 --epochs 60
encrypted-topology analyze --seeds 100

encrypted-topology run-robustness --seeds 30 --epochs 60
encrypted-topology analyze-robustness --seeds 30
```

Generated result tables remain local by default. Public summaries carry the verified source
hash, design size, multiplicity rule, and explicit claim boundary.

## Claim boundary

This code can evaluate statistical recovery of a known synthetic graph under declared event
laws and perturbations. It cannot establish facts about real communications, recover
encrypted content, identify people or organizations, infer intent, attribute command and
control, or demonstrate operational SIGINT performance. The independent-null family is a
negative control, not a detector of real-world absence or innocence.

See [`docs/SIMULATION_ONLY_PROTOCOL.md`](docs/SIMULATION_ONLY_PROTOCOL.md),
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md),
[`docs/STATISTICAL_MODEL.md`](docs/STATISTICAL_MODEL.md), and
[`docs/CLAIM_BOUNDARY.md`](docs/CLAIM_BOUNDARY.md).

## Repository map

```text
src/encrypted_topology/   simulator, likelihood, flows, models, diagnostics, CLI
tests/                    mathematical, boundary, robustness, and smoke tests
data/                     frozen design and machine-readable project status
docs/                     protocols, model specification, releases, claim boundary
report/                   exact 27-page statistical report
index.html                interactive evidence dashboard for GitHub Pages
scripts/                  report builder and artifact verifier
```

## License

MIT for repository code and original documentation.
