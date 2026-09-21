"""A compact MADE-based masked autoregressive variational flow."""

from __future__ import annotations

import torch
from torch import nn


class MaskedLinear(nn.Linear):
    mask: torch.Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: torch.Tensor) -> None:
        if mask.shape != self.mask.shape:
            raise ValueError("mask shape does not match linear layer")
        self.mask.copy_(mask)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(values, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """Masked autoencoder whose output dimension ``d`` depends only on ``< d``."""

    def __init__(self, dimensions: int, hidden: int = 64) -> None:
        super().__init__()
        if dimensions < 2 or hidden < dimensions:
            raise ValueError("MADE requires dimensions >= 2 and a sufficiently wide hidden layer")
        self.dimensions = dimensions
        self.first = MaskedLinear(dimensions, hidden)
        self.second = MaskedLinear(hidden, hidden)
        self.output = MaskedLinear(hidden, 2 * dimensions)
        self.activation = nn.SiLU()

        input_degree = torch.arange(1, dimensions + 1)
        hidden_degree_1 = torch.arange(hidden) % (dimensions - 1) + 1
        hidden_degree_2 = torch.arange(hidden) % (dimensions - 1) + 1
        output_degree = torch.arange(1, dimensions + 1).repeat(2)
        self.first.set_mask((hidden_degree_1[:, None] >= input_degree[None, :]).float())
        self.second.set_mask((hidden_degree_2[:, None] >= hidden_degree_1[None, :]).float())
        self.output.set_mask((output_degree[:, None] > hidden_degree_2[None, :]).float())

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.activation(self.first(values))
        hidden = self.activation(self.second(hidden))
        shift, raw_scale = self.output(hidden).chunk(2, dim=-1)
        return shift, 1.8 * torch.tanh(raw_scale / 1.8)


class MaskedAutoregressiveFlow(nn.Module):
    """Stacked MAF transformations used as an amortized posterior family.

    ``from_base`` applies the sequential inverse of the density-estimation MAF,
    which is the direction required for variational sampling. ``to_base`` is
    parallel and provides an exact round-trip and log-Jacobian check.
    """

    permutations: torch.Tensor

    def __init__(self, dimensions: int, hidden: int = 64, layers: int = 2) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("at least one flow layer is required")
        self.dimensions = dimensions
        self.layers = nn.ModuleList([MADE(dimensions, hidden) for _ in range(layers)])
        permutations = []
        for index in range(layers):
            order = torch.arange(dimensions - 1, -1, -1) if index % 2 == 0 else torch.arange(dimensions)
            permutations.append(order)
        self.register_buffer("permutations", torch.stack(permutations))

    def from_base(self, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = base
        total_log_det = torch.zeros(base.shape[:-1], device=base.device, dtype=base.dtype)
        for layer_index, layer in enumerate(self.layers):
            permutation = self.permutations[layer_index]
            permuted = values[..., permutation]
            constructed: list[torch.Tensor] = []
            layer_log_det = torch.zeros_like(total_log_det)
            for dimension in range(self.dimensions):
                zeros = torch.zeros_like(permuted[..., dimension:])
                partial = torch.cat([*constructed, zeros], dim=-1)
                shift, log_scale = layer(partial)
                next_value = shift[..., dimension] + torch.exp(
                    log_scale[..., dimension]
                ) * permuted[..., dimension]
                constructed.append(next_value.unsqueeze(-1))
                layer_log_det = layer_log_det + log_scale[..., dimension]
            transformed = torch.cat(constructed, dim=-1)
            inverse_permutation = torch.argsort(permutation)
            values = transformed[..., inverse_permutation]
            total_log_det = total_log_det + layer_log_det
        return values, total_log_det

    def to_base(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = values
        total_inverse_log_det = torch.zeros(
            values.shape[:-1], device=values.device, dtype=values.dtype
        )
        for layer_index in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_index]
            permutation = self.permutations[layer_index]
            permuted_current = current[..., permutation]
            shift, log_scale = layer(permuted_current)
            base_permuted = (permuted_current - shift) * torch.exp(-log_scale)
            inverse_permutation = torch.argsort(permutation)
            current = base_permuted[..., inverse_permutation]
            total_inverse_log_det = total_inverse_log_det - torch.sum(log_scale, dim=-1)
        return current, total_inverse_log_det
