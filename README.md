# Encrypted Protocol Topology Recovery

[![Validate](https://github.com/salemmorelli1/Encrypted-Protocol-Topology-Recovery/actions/workflows/validate.yml/badge.svg)](https://github.com/salemmorelli1/Encrypted-Protocol-Topology-Recovery/actions/workflows/validate.yml)
[![Pages](https://github.com/salemmorelli1/Encrypted-Protocol-Topology-Recovery/actions/workflows/pages.yml/badge.svg)](https://github.com/salemmorelli1/Encrypted-Protocol-Topology-Recovery/actions/workflows/pages.yml)

This repository is a **simulation-only computational statistics laboratory** for latent
directed-topology recovery. It combines an exponential-kernel multivariate Hawkes
likelihood, an amortized dynamic stochastic block model, masked autoregressive flows, and a
static graph-autoencoder/Louvain comparator.

Version 1.2.0 has no packet-capture, interface-enumeration, external-trace import, address
pseudonymization, or live-inference path. Every event, node label, graph, mark, and truth
label used by the runtime is generated in memory from a declared seed.

The synthetic cryptography laboratory is a separate in-memory component. It can decrypt only
the synthetic messages it encrypts when supplied its temporary experiment-owned AES-GCM key;
it does not recover keys or accept external ciphertext.

## Audit status

The registered primary study is a randomized complete-block `2 × 3 × 3 × 100` factorial:

- architecture: Hawkes-flow DSBM or static GAE–Louvain;
- perturbation: none, deterministic padding, or timing jitter;
- graph sparsity: dense, moderate, or sparse; and
- block: 100 seeds (`2026` through `2125`), for 1,800 runs.

A historical aggregate reported those 1,800 cells under SHA-256
`d19433ff32f2222655f5408810b9ed6fd59fa01ea461e05e3f1aa5a59977309a`.
It is **quarantined, not current evidence** because the source CSV is absent, dynamic-model
parameters were initialized before the declared cell seed was applied, and version 1.2.0
corrects the observation window and finite-window offspring simulation. A fresh 1,800-cell
run is required before making any architecture-performance claim. The old aggregate is
preserved in `data/empirical_summary.json` under an explicit quarantine status for traceability.

Analyzer v1.2.0 requires the exact registered keys and field order, finite/range-valid
endpoints, architecture-specific ESS values, and complete status. It hashes the immutable
source before and after reading, calculates 27 seed-paired contrasts, applies Holm adjustment
within endpoint, records mixed-model fallbacks, and writes its JSON summary atomically.

The separate robustness design remains pending:

| Generator | Role | Truth signal |
|---|---|---|
| `hawkes_exponential` | matched primary generator | yes |
| `hawkes_mixture` | excitation-kernel misspecification | yes |
| `cox_piecewise` | shared nonstationary-rate misspecification | yes |
| `renewal_gamma` | non-Poisson renewal misspecification | yes |
| `independent_null` | calibrated no-signal negative control | no |

With 30 seeds, that design contains 2,700 restartable runs. It uses a distinct schema and
applies Holm correction across all 45 generator-by-condition contrasts within each endpoint.

## Statistical model

For directed synthetic mark process \(r\), events \(\{(t_i,r_i)\}_{i=1}^{n}\) on \([0,T]\)
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

The explicit observation horizon retains both initial and terminal event-free exposure.
Simultaneous events do not excite one another because only strictly earlier events enter the
intensity. Excitation is scaled below the stability boundary. A neural encoder and invertible
MAF form the variational law; inferred block-specific baseline event rates yield directed
link scores.

## Install and verify in Git Bash

Python 3.12 and CPU execution are the reproducibility baseline:

```bash
cd "/c/Users/salem/GitHub/Encrypted-Protocol-Topology-Recovery"
py -3.12 -m venv .venv
source .venv/Scripts/activate

python -m pip install --disable-pip-version-check \
  torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --disable-pip-version-check \
  -r requirements-ci-lock.txt
python -m pip install --disable-pip-version-check --no-deps -e .

python -m pip check
python -m ruff check .
python -m mypy src scripts
python -m pytest -q
python -m compileall -q src scripts tests
python scripts/build_report.py
python scripts/verify_artifacts.py \
  --built-report output/pdf/Encrypted_Protocol_Topology_Recovery_APA_Report.pdf
git diff --check
git diff --exit-code
```

The default report build writes under ignored `output/` and must not dirty the repository.
Maintainers use `python scripts/build_report.py --publish` only when intentionally updating
the tracked PDF and figures.

## Run the simulations

```bash
encrypted-topology simulation-registry
encrypted-topology simulate \
  --generator hawkes_mixture \
  --seed 2026 \
  --sparsity moderate \
  --obfuscation jitter \
  --epochs 20 \
  --device cpu

encrypted-topology run-factorial --seeds 100 --epochs 60 --device cpu
encrypted-topology analyze --seeds 100

encrypted-topology run-robustness --seeds 30 --epochs 60 --device cpu
encrypted-topology analyze-robustness --seeds 30
```

Generated result CSVs remain local by default. Preserve a completed source CSV: a hash alone
cannot reproduce or audit the underlying cells.

## Run the synthetic cryptography control

```bash
encrypted-topology crypto-lab \
  --generator hawkes_exponential \
  --seed 2026 \
  --sparsity moderate \
  --obfuscation none
```

The command generates synthetic plaintext, encrypts it with a temporary 256-bit experiment
key and unique nonces, performs authenticated decryption, and verifies byte-exact recovery.
A separate key-withheld condition produces no plaintext output. The topology boundary is
limited to time, size, source, destination, synthetic node labels, and observation horizon;
it never receives keys, nonces, authentication tags, ciphertext, or plaintext.

This control demonstrates correct authorized decryption, not encryption breaking. It has no
network, packet-file, external-ciphertext, TLS, Wi-Fi, credential-guessing, or key-recovery
path.

## Claim boundary

This code can evaluate statistical recovery of a known synthetic graph under declared event
laws and perturbations after its registered studies are run and their exact sources pass the
analyzer. It cannot establish facts about real communications, identify people or
organizations, infer intent, attribute command and control, recover external content, or
demonstrate operational SIGINT performance.

See [`docs/SIMULATION_ONLY_PROTOCOL.md`](docs/SIMULATION_ONLY_PROTOCOL.md),
[`docs/SYNTHETIC_CRYPTOGRAPHY_PROTOCOL.md`](docs/SYNTHETIC_CRYPTOGRAPHY_PROTOCOL.md),
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md),
[`docs/STATISTICAL_MODEL.md`](docs/STATISTICAL_MODEL.md), and
[`docs/CLAIM_BOUNDARY.md`](docs/CLAIM_BOUNDARY.md).

## Repository map

```text
src/encrypted_topology/   simulator, cryptography controls, models, diagnostics, CLI
tests/                    mathematical, cryptography, boundary, robustness, and smoke tests
data/                     registered design, quarantine record, and project status
docs/                     protocols, model specification, releases, claim boundary
report/                   exact 27-page statistical report
index.html                interactive evidence dashboard for GitHub Pages
scripts/                  report builder and artifact verifier
```

## License

MIT for repository code and original documentation.
