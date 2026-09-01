# Statistical Model and Estimands

## Observation process

Let `r=(u,v)` index a directed synthetic dyad. Conditional on history `H_t`,

```math
\lambda_r(t)=\mu_r+\sum_{j:t_j<t}A_{r_jr}\beta\exp\{-\beta(t-t_j)\}.
```

For observation window `[0,T]`, the implemented log likelihood is

```math
\ell(\theta)=\sum_{i=1}^{n}\log \lambda_{r_i}(t_i)
-T\sum_r\mu_r
-\sum_{j=1}^{n}\sum_r A_{r_jr}\{1-e^{-\beta(T-t_j)}\}.
```

The last two terms are the compensator. Positive parameters use smooth transforms, and the
excitation operator is scaled below the stability boundary.

## Variational dynamic block model

The encoder maps node histories to a factorized Gaussian base law. Masked autoregressive
bijections transform base draws to a non-Gaussian variational law. Relaxed block indicators
are obtained with Gumbel–Softmax. Directed link probabilities derive from block-pair logits;
synthetic size marks receive a conditional Gaussian term. The optimized objective is a Monte
Carlo evidence lower bound with the MAF log-Jacobian included in `log q`.

## Comparator

The comparator uses a static graph autoencoder trained on aggregated synthetic adjacency and
a Louvain partition. It has no continuous-time likelihood and is deliberately simpler.

## Estimands and diagnostics

Primary endpoints are link AUC, binary link log score, and compute latency per event. The
architecture estimand is the Hawkes-flow DSBM result minus the static result within a shared
seed, event law, perturbation, and sparsity condition.

Importance ESS summarizes variational-weight degeneracy only for the dynamic method. Genuine
MCMC R-hat and ESS require independent chains; optimization trajectories are not chains.

The generator registry separates model alignment from recovery. In the independent-null
family, graph truth remains available for scoring but does not influence event generation.
Near-chance discrimination is therefore a calibration target rather than a failure to detect
a real entity or relationship.
