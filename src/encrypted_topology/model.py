"""Flow-amortized dynamic block inference under an exact conditional Hawkes target."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .events import EventBatch
from .flows import MaskedAutoregressiveFlow
from .hawkes import StableHawkesParameters, exponential_hawkes_log_likelihood


def aggregate_node_features(batch: EventBatch) -> torch.Tensor:
    """Construct topology-blind node features from counts, sizes, and timing."""

    device = batch.sizes.device
    n = batch.num_nodes
    out_count = torch.bincount(batch.src, minlength=n).float()
    in_count = torch.bincount(batch.dst, minlength=n).float()
    log_size = torch.log(batch.sizes.float())
    size_sum = torch.zeros(n, device=device).scatter_add_(0, batch.src, log_size)
    size_sum = size_sum + torch.zeros(n, device=device).scatter_add_(0, batch.dst, log_size)
    degree = out_count + in_count
    mean_log_size = size_sum / torch.clamp_min(degree, 1.0)
    span = torch.as_tensor(batch.horizon, device=device, dtype=torch.float32)
    rate = degree / span
    return torch.stack(
        [torch.log1p(out_count), torch.log1p(in_count), mean_log_size / 8.0, torch.log1p(rate)],
        dim=-1,
    )


@dataclass(frozen=True)
class FitResult:
    loss_trace: tuple[float, ...]
    latency_ms: float
    final_elbo_per_event: float
    occupied_blocks: int


class HawkesFlowDSBM(nn.Module):
    """Amortized latent-block posterior with MAF geometry and Hawkes dynamics."""

    def __init__(
        self,
        blocks: int = 4,
        latent_dim: int = 6,
        hidden: int = 64,
        flow_layers: int = 2,
    ) -> None:
        super().__init__()
        if blocks < 2 or latent_dim < 2 or hidden < latent_dim or flow_layers < 1:
            raise ValueError("invalid model dimensions")
        self.blocks = blocks
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * latent_dim),
        )
        self.flow = MaskedAutoregressiveFlow(latent_dim, hidden, flow_layers)
        self.block_head = nn.Linear(latent_dim, blocks)
        self.hawkes = StableHawkesParameters(blocks * blocks)
        self.log_size_mean = nn.Parameter(torch.full((blocks * blocks,), math.log(700.0)))
        self.log_size_scale = nn.Parameter(torch.zeros(blocks * blocks))

    def posterior(
        self, batch: EventBatch, temperature: float = 0.7
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = aggregate_node_features(batch)
        mean, raw_log_scale = self.encoder(features).chunk(2, dim=-1)
        log_scale = 2.0 * torch.tanh(raw_log_scale / 2.0)
        base = mean + torch.exp(log_scale) * torch.randn_like(mean)
        latent, flow_log_det = self.flow.from_base(base)
        logits = self.block_head(latent)
        assignments = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
        log_two_pi = math.log(2.0 * math.pi)
        log_q_base = -0.5 * torch.sum(
            ((base - mean) * torch.exp(-log_scale)) ** 2 + 2.0 * log_scale + log_two_pi,
            dim=-1,
        )
        log_q = log_q_base - flow_log_det
        log_p = -0.5 * torch.sum(latent**2 + log_two_pi, dim=-1)
        return assignments, latent, log_q, log_p

    def elbo(self, batch: EventBatch, temperature: float = 0.7) -> tuple[torch.Tensor, dict[str, float]]:
        assignments, _latent, log_q, log_p = self.posterior(batch, temperature)
        labels = torch.argmax(assignments, dim=-1)
        marks = labels[batch.src] * self.blocks + labels[batch.dst]
        baseline, excitation, decay = self.hawkes.constrained()
        log_hawkes = exponential_hawkes_log_likelihood(
            batch.times, marks, baseline, excitation, decay, batch.horizon
        )
        observed_log_size = torch.log(batch.sizes.float())
        size_scale = F.softplus(self.log_size_scale[marks]) + torch.finfo(torch.float32).eps
        size_center = self.log_size_mean[marks]
        log_size = -0.5 * torch.sum(
            ((observed_log_size - size_center) / size_scale) ** 2
            + 2.0 * torch.log(size_scale)
            + math.log(2.0 * math.pi)
        )
        continuous_kl_sample = torch.sum(log_q - log_p)
        block_prob = torch.softmax(self.block_head(_latent), dim=-1)
        categorical_kl = torch.sum(
            block_prob * (torch.log(torch.clamp_min(block_prob, torch.finfo(block_prob.dtype).tiny)) + math.log(self.blocks))
        )
        total = log_hawkes + log_size - continuous_kl_sample - categorical_kl
        diagnostics = {
            "hawkes_log_likelihood": float(log_hawkes.detach()),
            "size_log_likelihood": float(log_size.detach()),
            "kl_sample": float((continuous_kl_sample + categorical_kl).detach()),
            "occupied_blocks": float(torch.unique(labels).numel()),
        }
        return total, diagnostics

    def fit_batch(
        self,
        batch: EventBatch,
        epochs: int = 80,
        learning_rate: float = 2e-3,
        seed: int = 2026,
    ) -> FitResult:
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        torch.manual_seed(seed)
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        trace: list[float] = []
        if batch.times.is_cuda:
            torch.cuda.synchronize(batch.times.device)
        started = time.perf_counter()
        final_diagnostics: dict[str, float] = {}
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            temperature = max(0.35, 1.0 - 0.65 * epoch / max(epochs - 1, 1))
            objective, final_diagnostics = self.elbo(batch, temperature)
            loss = -objective / batch.num_events
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite variational objective")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in self.parameters()
            ):
                raise FloatingPointError("non-finite variational gradient")
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            optimizer.step()
            trace.append(float(loss.detach()))
        if batch.times.is_cuda:
            torch.cuda.synchronize(batch.times.device)
        latency = (time.perf_counter() - started) * 1000.0
        return FitResult(
            tuple(trace),
            latency,
            -trace[-1],
            int(final_diagnostics["occupied_blocks"]),
        )

    @torch.no_grad()
    def link_scores(self, batch: EventBatch, samples: int = 8) -> torch.Tensor:
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("samples must be a positive integer")
        scores = torch.zeros(batch.num_nodes, batch.num_nodes, device=batch.sizes.device)
        baseline, _, _ = self.hawkes.constrained()
        block_rate = baseline.reshape(self.blocks, self.blocks)
        for _ in range(samples):
            features = aggregate_node_features(batch)
            mean, raw_log_scale = self.encoder(features).chunk(2, dim=-1)
            base = mean + torch.exp(2.0 * torch.tanh(raw_log_scale / 2.0)) * torch.randn_like(mean)
            latent, _ = self.flow.from_base(base)
            q = torch.softmax(self.block_head(latent), dim=-1)
            scores = scores + q @ block_rate @ q.T
        scores = scores / samples
        scores.fill_diagonal_(0.0)
        return scores
