"""Diagnostics with estimand-specific interpretation."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import norm, rankdata


def importance_ess(log_weights: torch.Tensor) -> float:
    """Self-normalized importance ESS; this is not Markov-chain ESS."""

    values = log_weights.detach().double().flatten()
    normalized = torch.softmax(values, dim=0)
    return float(1.0 / torch.sum(normalized**2))


def rank_normalized_folded_rhat(chains: np.ndarray) -> float:
    """Folded split rank-normalized R-hat for genuine replicated chains only."""

    values = np.asarray(chains, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 8:
        raise ValueError("R-hat requires at least two chains and eight draws per chain")
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
    variance = (n - 1.0) / n * within + between / n
    return float(math.sqrt(variance / within))


def binary_log_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    epsilon = np.finfo(float).eps
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return float(np.mean(labels * np.log(probabilities) + (1.0 - labels) * np.log1p(-probabilities)))

