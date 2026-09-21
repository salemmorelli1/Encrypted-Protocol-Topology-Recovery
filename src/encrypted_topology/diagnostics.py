"""Diagnostics with estimand-specific interpretation."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import norm, rankdata


def importance_ess(log_weights: torch.Tensor) -> float:
    """Self-normalized importance ESS; this is not Markov-chain ESS."""

    values = log_weights.detach().double().flatten()
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("importance ESS requires finite nonempty log weights")
    normalized = torch.softmax(values, dim=0)
    return float(1.0 / torch.sum(normalized**2))


def rank_normalized_folded_rhat(chains: np.ndarray) -> float:
    """Folded split rank-normalized R-hat for genuine replicated chains only."""

    values = np.asarray(chains, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 8:
        raise ValueError("R-hat requires at least two chains and eight draws per chain")
    if not np.isfinite(values).all():
        raise ValueError("R-hat requires finite draws")
    half = values.shape[1] // 2
    split = np.concatenate([values[:, :half], values[:, -half:]], axis=0)
    folded = np.abs(split - np.median(split))
    ranks = rankdata(folded.ravel(), method="average")
    probabilities = (ranks - 0.375) / (ranks.size + 0.25)
    normalized = norm.ppf(probabilities).reshape(folded.shape)
    n = normalized.shape[1]
    chain_means = normalized.mean(axis=1)
    between = n * chain_means.var(ddof=1)
    within = normalized.var(axis=1, ddof=1).mean()
    if not np.isfinite(within) or within <= 0.0:
        raise ValueError("R-hat is undefined when within-chain variance is zero")
    variance = (n - 1.0) / n * within + between / n
    return float(math.sqrt(variance / within))


def binary_log_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or labels.size == 0:
        raise ValueError("labels and probabilities must be aligned and nonempty")
    if not np.isfinite(labels).all() or not np.isfinite(probabilities).all():
        raise ValueError("labels and probabilities must be finite")
    if not np.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError("binary labels must contain only zero and one")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    epsilon = np.finfo(float).eps
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return float(np.mean(labels * np.log(probabilities) + (1.0 - labels) * np.log1p(-probabilities)))
