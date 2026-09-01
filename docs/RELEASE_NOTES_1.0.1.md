# Release 1.0.1

> Historical record: the acquisition features described in this release were removed in
> version 1.1.0. They are not current commands or installation requirements.

This maintenance release repairs the empirical-execution path discovered during the first
Windows smoke run.

## Corrected behavior

- Dense block-graph draws are sampled from the declared Bernoulli law conditional on at
  least one absent off-diagonal block edge. This guarantees that link AUC has both positive
  and negative truth classes across all 100 frozen seeds.
- The test suite now checks formal seed 2026 directly and audits all frozen seeds across all
  three sparsity levels.
- Windows Scapy failures caused by a missing packet-capture provider now return a concise
  Npcap installation or repair message.
- Setup instructions explicitly create and verify a repository-specific virtual environment.

## Validation

- 16 tests pass.
- Ruff reports no violations.
- The 27-page report and dashboard claim boundary pass artifact verification.
- A complete 18-cell smoke factorial finishes successfully.

This release does not change the frozen `2 x 3 x 3 x 100` estimand, primary endpoints, or
claim boundary. Live capture still requires an authorized interface and a functioning Npcap
provider on Windows.
