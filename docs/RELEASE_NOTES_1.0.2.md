# Release 1.0.2

This release hardens the analyzer after the completed factorial run exposed mixed-model
boundary and Hessian warnings. It does not modify or rerun the 1,800 experimental cells.

## Corrected behavior

- Mixed-model acceptance now requires optimizer convergence, no Statsmodels convergence
  warning, finite fixed-effect estimates and positive standard errors, and a valid
  fixed-effect covariance matrix.
- The analyzer records every fit attempt. It tries a seed random intercept plus architecture
  slope, then a seed random intercept, and finally the same factorial fixed design with
  seed-clustered standard errors.
- Each endpoint's nine seed-paired architecture contrasts now include two-sided raw and
  Holm-adjusted *p* values. Holm adjustment controls familywise error within endpoint.
- The empirical summary now stores the verified SHA-256 digest of the exact source CSV. The
  analyzer computes the digest before and after reading and aborts if the file changes.
- Five regression tests cover Holm adjustment, paired inference, both fallback transitions,
  and end-to-end summary provenance.

## Disclosure

The convergence warnings were discovered after the formal cells were generated. This
revision is therefore a post-execution correction to inferential reporting and should be
identified as such in the manuscript or release history. The frozen simulator seeds,
architectures, factor levels, endpoint values, and cell-level result file remain unchanged.

