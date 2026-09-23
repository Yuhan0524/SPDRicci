"""SPD Ricci Flow model used for the Wisconsin experiment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def sym(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def from_eigh(values: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    return sym((vectors * values.unsqueeze(-2)) @ vectors.transpose(-1, -2))


def spd_project(matrix: torch.Tensor, floor: float) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(sym(matrix))
    return from_eigh(values.clamp_min(floor), vectors)


def spd_log(matrix: torch.Tensor, floor: float) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(sym(matrix))
    return from_eigh(values.clamp_min(floor).log(), vectors)


def spd_invsqrt(matrix: torch.Tensor, floor: float) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(sym(matrix))
    return from_eigh(values.clamp_min(floor).rsqrt(), vectors)


def spd_activation(
    matrix: torch.Tensor,
    activation: str,
    floor: float,
    tgrelu_delta: float,
) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(sym(matrix))
    name = activation.lower().replace("-", "_")
    if name in {"reeig", "mild_reeig"}:
        values = values.clamp_min(floor)
    elif name == "tgrelu":
        logged = values.clamp_min(floor).log()
        spacing = tgrelu_delta * torch.arange(
            logged.shape[-1], dtype=logged.dtype, device=logged.device
        )
        logged = torch.where(logged > 0.0, logged, spacing.expand_as(logged))
        values = logged.exp()
    else:
        raise ValueError(f"Unsupported SPD activation: {activation}")
    return from_eigh(values, vectors)


def guaranteed_invertible_bimap(
    weight: torch.Tensor, min_singular_value: float
) -> torch.Tensor:
    """Floor singular values only when a learned BiMap approaches singularity."""
    if min_singular_value <= 0.0:
        raise ValueError("BiMap minimum singular value must be positive")
    with torch.no_grad():
        is_safe = bool(
            torch.linalg.svdvals(weight.detach()).amin() >= min_singular_value
        )
    if is_safe:
        return weight
    left, singular_values, right_t = torch.linalg.svd(weight, full_matrices=False)
    singular_values = singular_values.clamp_min(min_singular_value)
    return (left * singular_values.unsqueeze(-2)) @ right_t


class FeatureToSPD(nn.Module):
    def __init__(self, input_dim: int, spd_dim: int, dropout: float) -> None:
        super().__init__()
        self.spd_dim = spd_dim
        self.dimred = nn.Linear(input_dim, spd_dim * spd_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tangent = self.dimred(features).reshape(-1, self.spd_dim, self.spd_dim)
        tangent = sym(self.dropout(sym(tangent)))
        return sym(torch.matrix_exp(tangent))


class InvertibleBiMap(nn.Module):
    def __init__(
        self,
        dim: int,
        max_delta: float,
        min_singular_value: float,
    ) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.min_singular_value = float(min_singular_value)
        self.raw_weight = nn.Parameter(torch.zeros(dim, dim))
        self.register_buffer("identity", torch.eye(dim))

    def weight(self) -> torch.Tensor:
        candidate = self.identity + self.max_delta * torch.tanh(self.raw_weight)
        return guaranteed_invertible_bimap(candidate, self.min_singular_value)

    def encode(self, matrices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.weight()
        return sym(weight.transpose(-1, -2) @ matrices @ weight), weight

    @staticmethod
    def decode(matrices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        inverse = torch.linalg.inv(weight)
        return sym(inverse.transpose(-1, -2) @ matrices @ inverse)


class SPDRicciLayer(nn.Module):
    def __init__(
        self,
        beta: float,
        epsilon: float,
        spd_floor: float,
        activation: str,
        tgrelu_delta: float,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.spd_floor = float(spd_floor)
        self.activation = activation
        self.tgrelu_delta = float(tgrelu_delta)

    def forward(
        self, matrices: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        source, target = edge_index
        source_matrices = matrices.index_select(0, source)
        target_matrices = matrices.index_select(0, target)

        inverse_sqrt = spd_invsqrt(source_matrices, self.spd_floor)
        relative = sym(inverse_sqrt @ target_matrices @ inverse_sqrt)
        relative_log = spd_log(relative, self.spd_floor)
        squared_distance = (relative_log * relative_log).sum(dim=(-1, -2))
        edge_weights = torch.exp(-self.beta * squared_distance)
        contributions = relative_log * edge_weights[:, None, None]

        neighborhood_sum = torch.zeros_like(matrices)
        neighborhood_sum.index_add_(0, source, contributions)
        ricci = sym(
            -0.5
            * (
                matrices @ neighborhood_sum
                + neighborhood_sum @ matrices
            )
        )
        updated = matrices - self.epsilon * ricci
        return spd_activation(
            updated,
            self.activation,
            self.spd_floor,
            self.tgrelu_delta,
        )


class LogSPDClassifier(nn.Module):
    def __init__(self, spd_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        rows, cols = torch.triu_indices(spd_dim, spd_dim)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(spd_dim * (spd_dim + 1) // 2, num_classes)

    def forward(self, matrices: torch.Tensor) -> torch.Tensor:
        logged = spd_log(matrices, floor=1.0e-6)
        vectors = logged[:, self.rows, self.cols]
        return self.projection(self.dropout(vectors))


@dataclass(frozen=True)
class ModelConfig:
    spd_dim: int = 16
    layers: int = 2
    feature_dropout: float = 0.0
    classifier_dropout: float = 0.5
    beta: float = 300.0
    epsilon: float = 0.1
    spd_floor: float = 1.0e-4
    activation: str = "tgrelu"
    tgrelu_delta: float = 0.2
    bimap_max_delta: float = 0.08
    bimap_min_singular_value: float = 0.1


class SPDRicci(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, config: ModelConfig) -> None:
        super().__init__()
        if config.layers < 1:
            raise ValueError("layers must be at least one")
        self.config = config
        self.feature_to_spd = FeatureToSPD(
            input_dim,
            config.spd_dim,
            config.feature_dropout,
        )
        self.bimaps = nn.ModuleList(
            InvertibleBiMap(
                config.spd_dim,
                config.bimap_max_delta,
                config.bimap_min_singular_value,
            )
            for _ in range(config.layers)
        )
        self.diffusion = nn.ModuleList(
            SPDRicciLayer(
                config.beta,
                config.epsilon,
                config.spd_floor,
                config.activation,
                config.tgrelu_delta,
            )
            for _ in range(config.layers)
        )
        self.output_head = LogSPDClassifier(
            config.spd_dim,
            num_classes,
            config.classifier_dropout,
        )

    def forward(
        self, features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        matrices = self.feature_to_spd(features)
        for bimap, layer in zip(self.bimaps, self.diffusion):
            latent, weight = bimap.encode(matrices)
            latent = layer(latent, edge_index)
            matrices = bimap.decode(latent, weight)
            matrices = spd_project(matrices, self.config.spd_floor)
        return self.output_head(matrices)


def build_model(
    input_dim: int, num_classes: int, config: ModelConfig
) -> SPDRicci:
    return SPDRicci(input_dim, num_classes, config)
