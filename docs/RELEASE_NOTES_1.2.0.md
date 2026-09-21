# Release 1.2.0

This audit release corrects deterministic initialization, explicit observation-window
handling, finite-window Hawkes offspring sampling, and equal-time history semantics. It also
adds strict result-schema and exact-design validation, atomic summaries, pinned two-platform
CI, type checking, and idempotent report generation.

The previously reported 1,800-row aggregate is quarantined. Its source CSV is absent, and
the declared cell seed did not cover dynamic-model parameter initialization. Version 1.2.0
therefore makes no architecture ranking. A corrected 1,800-cell primary rerun and the
2,700-cell robustness study remain pending.

The simulation-only and authorized synthetic-cryptography boundaries are unchanged: no real
traffic, external ciphertext, interception, key recovery, identity, intent, attribution,
command-and-control, or operational SIGINT claim is supported.
