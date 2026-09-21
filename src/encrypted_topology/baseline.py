"""Static graph autoencoder with a Louvain community summary."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import networkx as nx
import torch
from torch import nn

from .events import EventBatch
from .model import aggregate_node_features


def adjacency_from_events(batch: EventBatch) -> torch.Tensor:
    adjacency = torch.zeros(batch.num_nodes, batch.num_nodes, device=batch.sizes.device)
    ones = torch.ones(batch.num_events, device=batch.sizes.device)
    adjacency.index_put_((batch.src, batch.dst), ones, accumulate=True)
    return adjacency


class StaticGraphAutoencoder(nn.Module):
    def __init__(self, feature_dim: int = 4, hidden: int = 32, latent: int = 8) -> None:
        super().__init__()
        if min(feature_dim, hidden, latent) < 1:
            raise ValueError("autoencoder dimensions must be positive")
        self.first = nn.Linear(feature_dim, hidden, bias=False)
        self.second = nn.Linear(hidden, latent, bias=False)

    def encode(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        n = adjacency.shape[0]
        augmented = adjacency + adjacency.T + torch.eye(n, device=adjacency.device)
        degree = torch.sum(augmented, dim=1)
        inv_sqrt = torch.rsqrt(torch.clamp_min(degree, 1.0))
        normalized = inv_sqrt[:, None] * augmented * inv_sqrt[None, :]
        return normalized @ torch.relu(self.first(normalized @ features)) @ self.second.weight.T

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        latent = self.encode(features, adjacency)
        return torch.sigmoid(latent @ latent.T)


@dataclass(frozen=True)
class StaticFitResult:
    scores: torch.Tensor
    latency_ms: float
    occupied_communities: int
    final_loss: float


def fit_static_baseline(
    batch: EventBatch, epochs: int = 80, learning_rate: float = 5e-3, seed: int = 2026
) -> StaticFitResult:
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    torch.manual_seed(seed)
    adjacency = adjacency_from_events(batch)
    target = (adjacency > 0).float()
    target.fill_diagonal_(0.0)
    features = aggregate_node_features(batch)
    model = StaticGraphAutoencoder().to(features.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    mask = ~torch.eye(batch.num_nodes, dtype=torch.bool, device=features.device)
    positive = torch.sum(target[mask])
    negative = mask.sum() - positive
    positive_weight = torch.clamp(negative / torch.clamp_min(positive, 1.0), max=30.0)
    if features.is_cuda:
        torch.cuda.synchronize(features.device)
    started = time.perf_counter()
    loss = torch.tensor(float("nan"), device=features.device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features, adjacency)
        weights = torch.where(target > 0, positive_weight, 1.0)
        loss = nn.functional.binary_cross_entropy(prediction[mask], target[mask], weight=weights[mask])
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite static baseline objective")
        loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError("non-finite static baseline gradient")
        optimizer.step()
    if features.is_cuda:
        torch.cuda.synchronize(features.device)
    latency = (time.perf_counter() - started) * 1000.0
    with torch.no_grad():
        scores = model(features, adjacency)
        scores.fill_diagonal_(0.0)

    graph = nx.DiGraph()
    graph.add_nodes_from(range(batch.num_nodes))
    rows, columns = torch.where(adjacency.detach().cpu() > 0)
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        graph.add_edge(row, column, weight=float(adjacency[row, column]))
    undirected = graph.to_undirected()
    communities = nx.community.louvain_communities(undirected, seed=seed, weight="weight")
    return StaticFitResult(scores, latency, len(communities), float(loss.detach()))
