# Encrypted Protocol Topology Recovery

[![Validate](https://github.com/salemmorelli1/Encrypted-Protocol-Topology-Recovery/actions/workflows/validate.yml/badge.svg)](https://github.com/salemmorelli1/Encrypted-Protocol-Topology-Recovery/actions/workflows/validate.yml)
[![Pages](https://github.com/salemmorelli1/Encrypted-Protocol-Topology-Recovery/actions/workflows/pages.yml/badge.svg)](https://github.com/salemmorelli1/Encrypted-Protocol-Topology-Recovery/actions/workflows/pages.yml)

This repository is a reproducible, privacy-minimized laboratory for inferring **latent communication topology** from authorized, local packet-metadata streams. It joins an exact exponential-kernel multivariate Hawkes likelihood (event term and compensator), an amortized dynamic stochastic block model, masked autoregressive flows, and a controlled-truth `2 × 3 × 3` randomized complete-block experiment.

The repository does **not** inspect payloads, store ports, retain raw IP/MAC addresses, claim command-and-control attribution, or claim operational cyber-SIGINT validation. Live topology is unlabelled unless an independently defined truth graph is supplied.

## Evidence lanes

| Lane | Source | Inferential role | Permitted claim |
|---|---|---|---|
| Controlled truth | Branching multivariate Hawkes simulator | Architecture comparison across obfuscation and sparsity | Link AUC and link log score when the full experiment is executed |
| Authorized live metadata | Scapy or TShark on one named local interface | External feasibility and posterior topology exploration | Timing, event counts, posterior structure; no live link AUC without truth |

Project status is machine-readable in [`data/project_status.json`](data/project_status.json). Shipped status is **methodology ready / empirical execution pending**; no results are fabricated.

## Statistical model

For directed dyad/type process \(r\), observed events \(\{(t_i,r_i)\}_{i=1}^{n}\) on \([0,T]\) have conditional intensity

\[
\lambda_r(t\mid\mathcal H_t)=\mu_r+\sum_{j:t_j<t} A_{r_jr}\,\beta e^{-\beta(t-t_j)}.
\]

The implemented log likelihood is

\[
\ell=\sum_i\log\lambda_{r_i}(t_i)-T\sum_r\mu_r-
\sum_j\sum_r A_{r_jr}\{1-e^{-\beta(T-t_j)}\}.
\]

The compensator is not optional: omitting it does not define the point-process likelihood. Stability is enforced by scaling the excitation matrix to spectral norm at most `0.85`. A neural encoder produces node-wise Gaussian base variables; an explicitly invertible MAF transforms the variational law, and relaxed block assignments parameterize directed link probabilities.

The static comparator is a graph autoencoder plus Louvain partition. The controlled study is a **randomized complete-block repeated-measures factorial**, not a split plot: every seed is evaluated at all `2 × 3 × 3` cells.

## Safe local installation

Use only a computer and network interface you own or are expressly authorized to monitor. Wireshark/TShark typically requires Npcap on Windows.

```bash
cd "/c/Users/salem/GitHub/Encrypted-Protocol-Topology-Recovery"
deactivate 2>/dev/null || true
python -m venv .venv
source .venv/Scripts/activate
python -c "import sys; print(sys.executable)"
python -m pip install --upgrade pip
python -m pip install -e ".[report,dev]"
python -m pytest -q
```

The printed interpreter must end in
`Encrypted-Protocol-Topology-Recovery\\.venv\\Scripts\\python.exe`. If it instead
points to another project, stop and reactivate this repository's environment before
installing or executing anything.

If an organizational Windows policy blocks the PyTorch DLL, install the signed CPU wheel through the approved environment or ask the device administrator; do not disable endpoint security.

On Windows, Scapy and TShark capture require the Npcap driver. If the command reports
that WinPcap or libpcap is unavailable, install or repair Npcap through the official
Wireshark installer, reopen Git Bash, and retry. Do not bypass organizational capture
policy or monitor an interface without authorization.

## Authorized live capture

List the interfaces exactly as Scapy and TShark see them:

```bash
encrypted-topology interfaces
```

Capture through Scapy and immediately run inference:

```bash
encrypted-topology live \
  --backend scapy \
  --interface "Wi-Fi" \
  --duration 60 \
  --packet-limit 10000 \
  --authorized-capture
```

Or use TShark's field stream:

```bash
encrypted-topology live \
  --backend tshark \
  --interface "5" \
  --duration 60 \
  --packet-limit 10000 \
  --authorized-capture
```

The authorization flag is mandatory. The capture uses a named interface, `promisc=False` in Scapy, and stores only timestamp, frame length, keyed endpoint pseudonyms, and coarse protocol families. The raw HMAC salt and all local captures are ignored by Git.

## Controlled-truth experiment

Run one smoke cell:

```bash
encrypted-topology simulate --seed 2026 --sparsity moderate --obfuscation jitter --epochs 20
```

Run the preregistered 1,800-cell experiment (`100 seeds × 18 cells`) with restartable output:

```bash
encrypted-topology run-factorial --seeds 100 --epochs 60
encrypted-topology analyze --seeds 100
```

Primary controlled-truth endpoints are link AUC, held-out binary link log score, and synchronized compute latency per event. Self-normalized importance ESS is reported only for the variational architecture and is **not** treated as a cross-method MCMC mixing endpoint. Rank-normalized \(\widehat R\) is included only for genuine multi-chain draws, never as an ordinary variational diagnostic.

## Repository map

```text
src/encrypted_topology/   capture, privacy, likelihood, flows, models, CLI
tests/                    mathematical, privacy, and smoke tests
data/                     design and machine-readable status (no private captures)
docs/                     protocol, model specification, claim boundary
report/                   exact 27-page APA-style methodology report
index.html                interactive evidence dashboard for GitHub Pages
scripts/                  report and status-site builders
```

## Reproducibility and claim boundary

- The capture layer never writes PCAP/PCAPNG files.
- Raw addresses and ports never enter persisted rows.
- Pseudonyms are HMAC-SHA256 digests under a local 256-bit salt.
- Experimental results remain absent until the locked `2 × 3 × 3 × 100` run completes.
- A live network run is observational and unlabelled; controlled link truth comes only from simulation or a separately authorized testbed.
- Findings must be stated as topology-recovery evidence, not identity, intent, or C2 attribution.

See [`docs/CAPTURE_AND_PRIVACY_PROTOCOL.md`](docs/CAPTURE_AND_PRIVACY_PROTOCOL.md), [`docs/STATISTICAL_MODEL.md`](docs/STATISTICAL_MODEL.md), and [`docs/CLAIM_BOUNDARY.md`](docs/CLAIM_BOUNDARY.md).

## References

- Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika, 58*(1), 83–90. https://doi.org/10.1093/biomet/58.1.83
- Matias, C., & Miele, V. (2017). Statistical clustering of temporal networks through a dynamic stochastic block model. *JRSS B, 79*(4), 1119–1141. https://doi.org/10.1111/rssb.12200
- Papamakarios, G., Pavlakou, T., & Murray, I. (2017). Masked autoregressive flow for density estimation. https://arxiv.org/abs/1705.07057
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization. *Bayesian Analysis, 16*(2), 667–718. https://doi.org/10.1214/20-BA1221
- Wireshark Foundation. (n.d.). *TShark manual page*. https://www.wireshark.org/docs/man-pages/tshark.html
- Scapy Project. (n.d.). *Scapy usage documentation*. https://scapy.readthedocs.io/en/latest/usage.html

## License

MIT for repository code and original documentation. Packet data remain subject to the operator's authorization, institutional rules, and applicable law.
