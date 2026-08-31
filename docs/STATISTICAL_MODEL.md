# Statistical Model and Estimands

## Observation process

Let `r=(u,v)` index a directed pseudonymous dyad. Conditional on history `H_t`,

```math
\lambda_r(t)=\mu_r+\sum_{j:t_j<t}A_{r_jr}\beta\exp\{-\beta(t-t_j)\}.
```

For observation window `[0,T]`, the implemented log likelihood is

```math
\ell(\theta)=\sum_{i=1}^{n}\log \lambda_{r_i}(t_i)
-T\sum_r\mu_r
-\sum_{j=1}^{n}\sum_r A_{r_jr}\{1-e^{-\beta(T-t_j)}\}.
```

The last two terms are the compensator. Positive parameters are obtained by smooth transforms, and the excitation operator is rescaled below the stability boundary.

## Variational dynamic block model

The encoder maps node histories to a factorized Gaussian base law. A stack of masked autoregressive bijections transforms base draws to a non-Gaussian variational law. Relaxed block indicators are obtained with Gumbel–Softmax. Directed link probabilities derive from block-pair logits; packet-size marks receive a conditional Gaussian term. The optimized objective is a Monte Carlo evidence lower bound with the MAF log-Jacobian included in `log q`.

## Comparator

The comparator uses a static graph autoencoder trained on the aggregated adjacency matrix and a Louvain partition. It has no continuous-time likelihood and therefore answers a different, deliberately simpler approximation question.

## Experimental design

Each of 100 independently seeded truth graphs is observed under all 18 combinations:

- architecture: Hawkes-flow DSBM vs. static GAE–Louvain;
- obfuscation: none, deterministic padding, hostile timing jitter;
- sparsity: dense, moderate, sparse.

This is a randomized complete-block repeated-measures design. Seed is the block; cell order is randomized within seed. Fixed effects include all interactions. A confirmatory mixed model uses a seed random intercept, with a prespecified random architecture slope retained when estimable. Cluster-robust or parametric-bootstrap inference is preferred when Gaussian or homoscedastic residual assumptions fail.

## Endpoints

Primary controlled-truth endpoints are link AUC, binary link log score, and compute latency per event. Importance ESS summarizes variational weight degeneracy only for the dynamic method. Genuine MCMC R-hat and ESS require actual independent chains; optimization trajectories are not chains.
