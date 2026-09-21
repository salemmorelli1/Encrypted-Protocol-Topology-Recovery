# Simulation-Only Protocol

## Runtime boundary

Version 1.2.0 accepts no network interface, packet file, external ciphertext, endpoint list,
external event stream, external key, or address map. The package contains no acquisition
backend, packet parser, live inference command, key-recovery routine, or pseudonymization
key. All runtime observations and plaintexts are synthetic objects generated in memory.

The CLI is restricted to seven commands: `simulation-registry`, `simulate`, `crypto-lab`,
`run-factorial`, `analyze`, `run-robustness`, and `analyze-robustness`. `crypto-lab` accepts
only generator design parameters and performs an in-memory, experiment-owned AES-GCM round
trip. Tests lock that command set, assert that the former collection modules are not
importable, and reject packet-capture libraries in runtime dependencies.

## Synthetic cryptography boundary

The cryptography component and topology component are separate. AES-GCM receives generated
plaintext, a temporary experiment key, a unique nonce, associated data, ciphertext, and an
authentication tag. Topology recovery continues to receive only event times, sizes, source
and destination indices, synthetic node labels, and the declared observation horizon.

The authorized condition requires byte-exact plaintext reconstruction. The key-withheld
condition never invokes decryption and must emit zero plaintext outputs. This is a software
boundary control, not a claim that metadata experimentally proves the security of AES-GCM.
See `SYNTHETIC_CRYPTOGRAPHY_PROTOCOL.md` for the complete design and interpretation.

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

The primary generator remains `hawkes_exponential`. Version 1.2.0 corrects finite-window
offspring sampling, retains the full `[0,T]` observation window, and seeds before model
construction. These corrections require a fresh 1,800-cell primary run. The historical
aggregate is quarantined, and the robustness experiment remains a separate output schema.

Each experiment is restartable. Completed cells are keyed by all design coordinates and
written atomically through a temporary file. An analyzer verifies strict schema and numeric
contracts plus the exact key set, checks source SHA-256 before and after reading, and writes
aggregate JSON atomically only after every lock passes.

## Data rule

Do not adapt this repository to ingest real traffic, external ciphertext, or third-party keys
for this study. Suitable extensions are new stochastic processes, graph priors, synthetic
perturbations, controlled cryptographic round trips, calibration diagnostics, simulation-based
calibration, posterior-predictive checks, and compute benchmarks. External validity questions
should be addressed with published aggregate benchmarks or a separately governed project,
not by adding collection, interception, or key-recovery code here.
