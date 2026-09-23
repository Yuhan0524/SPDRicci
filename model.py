"""simple1 diffusion with the official SPD4GNN input and output mappings.

The architecture is intentionally kept as:

    FeatureToSPD -> [BiMap -> simple1 SPD diffusion -> inverse BiMap] * L
                 -> SPD4GNN output head

The diffusion equations match ``Optical_AI/SPD/Diffusion/spd_diffusion``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


def guaranteed_invertible_bimap(
    weight: torch.Tensor, min_singular_value: float = 0.1
) -> torch.Tensor:
    """Floor singular values only when a learned BiMap approaches singularity."""
    if min_singular_value <= 0.0:
        raise ValueError("BiMap minimum singular value must be positive")

    # Preserve an already safe learned map exactly, including its gradient path.
    with torch.no_grad():
        is_safe = bool(
            torch.linalg.svdvals(weight.detach()).amin() >= min_singular_value
        )
    if is_safe:
        return weight

    left, singular_values, right_t = torch.linalg.svd(weight, full_matrices=False)
    singular_values = singular_values.clamp_min(min_singular_value)
    return (left * singular_values.unsqueeze(-2)) @ right_t


def from_eigh(values: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    return sym((vectors * values.unsqueeze(-2)) @ vectors.transpose(-1, -2))


def spd4gnn_eigh(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the upstream float32 path, with a float64 retry on solver failure."""
    try:
        return torch.linalg.eigh(x, UPLO="U")
    except torch._C._LinAlgError:
        values, vectors = torch.linalg.eigh(x.to(torch.float64), UPLO="U")
        return values.to(x.dtype), vectors.to(x.dtype)


def spd_project(x: torch.Tensor, floor: float) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(sym(x))
    return from_eigh(values.clamp_min(floor), vectors)


def spd_activation(
    x: torch.Tensor,
    activation: str,
    floor: float,
    beta: float = 20.0,
    jitter: float = 0.0,
    tgrelu_delta: float = 1.0e-2,
) -> torch.Tensor:
    """Match the activation family in the reference Simple1 SPDNet block."""
    values, vectors = torch.linalg.eigh(sym(x))
    normalized = activation.lower().replace("-", "_")
    if normalized in {"mild_reeig", "reeig", "mild"}:
        values = values.clamp_min(floor)
    elif normalized in {"smooth_reeig", "smooth"}:
        values = floor + torch.nn.functional.softplus(
            beta * (values - floor)
        ) / beta
    elif normalized in {"tgrelu", "tgreig", "tgreig_tgrelu", "tgreig/tgrelu"}:
        values = values.clamp_min(floor)
        logged = values.log()
        step = tgrelu_delta * torch.arange(
            logged.shape[-1], dtype=logged.dtype, device=logged.device
        )
        logged = torch.where(logged > 0.0, logged, step.expand_as(logged))
        values = logged.exp()
    else:
        raise ValueError(f"Unsupported SPD activation: {activation}")
    if jitter > 0:
        step = jitter * torch.arange(
            values.shape[-1], dtype=values.dtype, device=values.device
        )
        values = values + step.expand_as(values)
    return from_eigh(values, vectors)


def spd_log(x: torch.Tensor, floor: float) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(sym(x))
    return from_eigh(values.clamp_min(floor).log(), vectors)


def spd_invsqrt(x: torch.Tensor, floor: float) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(sym(x))
    return from_eigh(values.clamp_min(floor).rsqrt(), vectors)


def expmap_identity(x: torch.Tensor) -> torch.Tensor:
    """Mathematical Exp_I without SPD4GNN's implementation jitter."""
    return sym(torch.matrix_exp(sym(x)))


def spd4gnn_expmap_identity(x: torch.Tensor, jitter: float) -> torch.Tensor:
    """Match SPD4GNN ``sym_funcm(x, torch.exp)`` including its jitter."""
    if jitter <= 0:
        return expmap_identity(x)
    noisy = sym(x) + float(jitter) * torch.randn_like(x)
    values, vectors = spd4gnn_eigh(noisy)
    return from_eigh(values.exp(), vectors)


def spd4gnn_logmap_identity(
    x: torch.Tensor, jitter: float, floor: float = 1.0e-6
) -> torch.Tensor:
    """Match SPD4GNN ``sym_funcm(x, torch.log)`` including its jitter."""
    if jitter <= 0:
        return spd_log(x, floor)
    noisy = sym(x) + float(jitter) * torch.randn_like(x)
    values, vectors = spd4gnn_eigh(noisy)
    return from_eigh(values.clamp_min(floor).log(), vectors)


def spd4gnn_sqrtmap_identity(
    x: torch.Tensor, jitter: float, floor: float = 1.0e-6
) -> torch.Tensor:
    """Match SPD4GNN ``sym_funcm(x, torch.sqrt)`` including its jitter."""
    if jitter <= 0:
        values, vectors = torch.linalg.eigh(sym(x))
    else:
        noisy = sym(x) + float(jitter) * torch.randn_like(x)
        values, vectors = spd4gnn_eigh(noisy)
    return from_eigh(values.clamp_min(floor).sqrt(), vectors)


class SPD4GNNFeatureToSPD(nn.Module):
    """SPD4GNN Vec2SymMat variants and the rank-one input ablation."""

    def __init__(
        self,
        input_dim: int,
        spd_dim: int,
        dropout: float,
        mapper: str = "squared",
        gram_epsilon: float = 1.0e-4,
        spd4gnn_jitter: float = 0.0,
    ) -> None:
        super().__init__()
        self.spd_dim = spd_dim
        self.mapper = mapper.lower()
        self.gram_epsilon = float(gram_epsilon)
        self.spd4gnn_jitter = float(spd4gnn_jitter)
        if self.mapper == "squared":
            projection_dim = spd_dim * spd_dim
        elif self.mapper == "rank_one":
            projection_dim = spd_dim
        elif self.mapper in {"triangular", "gram"}:
            projection_dim = spd_dim * (spd_dim + 1) // 2
            rows, cols = torch.triu_indices(spd_dim, spd_dim)
            self.register_buffer("rows", rows)
            self.register_buffer("cols", cols)
        else:
            raise ValueError(f"Unsupported official SPD4GNN mapper: {mapper}")
        self.dimred = nn.Linear(input_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.gram_activation = nn.LeakyReLU(0.2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dimred(features)
        if self.mapper == "rank_one":
            # The ablation maps one learned d-vector per node directly to
            # uu^T + delta I; gram_epsilon is delta and is independent from
            # the Simple1 diffusion step size named epsilon in ModelConfig.
            vector = self.dropout(projected)
            matrices = vector.unsqueeze(-1) @ vector.unsqueeze(-2)
            identity = torch.eye(
                self.spd_dim, dtype=features.dtype, device=features.device
            )
            return sym(matrices) + self.gram_epsilon * identity
        if self.mapper == "squared":
            tangent = projected.reshape(-1, self.spd_dim, self.spd_dim)
            tangent = sym(tangent)
        else:
            if self.mapper == "gram":
                projected = self.gram_activation(projected)
            tangent = projected.new_zeros(
                (projected.shape[0], self.spd_dim, self.spd_dim)
            )
            tangent[:, self.rows, self.cols] = projected
            tangent[:, self.cols, self.rows] = projected
            if self.mapper == "gram":
                diagonal = torch.arange(self.spd_dim, device=features.device)
                tangent[:, diagonal, diagonal] *= 0.5
                matrices = sym(tangent @ tangent.transpose(-1, -2))
                identity = torch.eye(
                    self.spd_dim, dtype=features.dtype, device=features.device
                )
                return matrices + self.gram_epsilon * identity
        # SPD4GNN applies dropout to matrix entries and symmetrizes again.
        tangent = sym(self.dropout(tangent))
        return spd4gnn_expmap_identity(tangent, self.spd4gnn_jitter)


class TiedBiMap(nn.Module):
    """Bounded or unrestricted BiMap with a tied or untied decoder."""

    def __init__(
        self,
        dim: int,
        max_delta: float = 0.08,
        decoder: str = "tied_inverse",
        parameterization: str = "bounded",
        min_singular_value: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.min_singular_value = float(min_singular_value)
        if self.min_singular_value <= 0.0:
            raise ValueError("BiMap minimum singular value must be positive")
        self.decoder = decoder.lower()
        self.parameterization = parameterization.lower()
        if self.decoder not in {"tied_inverse", "untied"}:
            raise ValueError(f"Unsupported BiMap decoder: {decoder}")
        if self.parameterization not in {"bounded", "unrestricted"}:
            raise ValueError(
                f"Unsupported BiMap parameterization: {parameterization}"
            )
        initial = (
            torch.zeros(dim, dim)
            if self.parameterization == "bounded"
            else torch.eye(dim)
        )
        self.raw_weight = nn.Parameter(initial)
        self.raw_decoder = (
            nn.Parameter(initial.clone()) if self.decoder == "untied" else None
        )
        self.register_buffer("identity", torch.eye(dim))

    def weight(self) -> torch.Tensor:
        if self.parameterization == "unrestricted":
            candidate = self.raw_weight
        else:
            candidate = self.identity + self.max_delta * torch.tanh(
                self.raw_weight
            )
        return guaranteed_invertible_bimap(
            candidate, self.min_singular_value
        )

    def encode(self, matrices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.weight()
        return sym(weight.transpose(-1, -2) @ matrices @ weight), weight

    def decode(self, matrices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.raw_decoder is None:
            decoder = torch.linalg.inv(weight)
        elif self.parameterization == "unrestricted":
            decoder = guaranteed_invertible_bimap(
                self.raw_decoder, self.min_singular_value
            )
        else:
            candidate = self.identity + self.max_delta * torch.tanh(
                self.raw_decoder
            )
            decoder = guaranteed_invertible_bimap(
                candidate, self.min_singular_value
            )
        return sym(decoder.transpose(-1, -2) @ matrices @ decoder)


class SPD4GNNGCNLayer(nn.Module):
    """Upstream SPD4GNN ``SPDGCNConv`` with matched activation settings.

    The upstream QR transform projects an identity-initialized matrix through
    ``.data`` on every forward pass, so it remains a detached isometry. This
    intentionally preserves that behavior for a faithful pure SPD4GNN core.
    """

    def __init__(
        self,
        dim: int,
        spd_floor: float,
        activation: str,
        activation_beta: float,
        activation_jitter: float,
        tgrelu_delta: float,
        activation_floor: float | None,
        spd4gnn_jitter: float,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        self.spd_floor = float(spd_floor)
        self.activation = activation
        self.activation_beta = float(activation_beta)
        self.activation_jitter = float(activation_jitter)
        self.tgrelu_delta = float(tgrelu_delta)
        self.activation_floor = (
            float(spd_floor) if activation_floor is None else float(activation_floor)
        )
        self.spd4gnn_jitter = float(spd4gnn_jitter)
        self.isometry = nn.Parameter(torch.eye(dim))
        if has_bias:
            self.bias = nn.Parameter(torch.empty(1, dim, dim))
            nn.init.xavier_uniform_(self.bias)
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def _project_upstream_isometry(self) -> torch.Tensor:
        q, r = torch.linalg.qr(self.isometry)
        signs = torch.diagonal(r).sign()
        signs[signs == 0] = 1.0
        q = q * signs
        self.isometry.copy_(q)
        return self.isometry.detach()

    @staticmethod
    def _normalized_gcn_aggregate(
        tangent: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        source, target = edge_index
        node_count = tangent.shape[0]
        nodes = torch.arange(node_count, device=edge_index.device)
        source = torch.cat((source, nodes))
        target = torch.cat((target, nodes))
        if edge_weight is None:
            weights = torch.ones(
                edge_index.shape[1], dtype=tangent.dtype, device=tangent.device
            )
        else:
            if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
                raise ValueError("edge_weight must have one scalar per directed edge")
            weights = edge_weight.to(dtype=tangent.dtype)
        weights = torch.cat(
            (weights, torch.ones(node_count, dtype=tangent.dtype, device=tangent.device))
        )
        degree = torch.zeros(node_count, dtype=tangent.dtype, device=tangent.device)
        degree.index_add_(0, target, weights)
        inverse_sqrt = degree.clamp_min(1.0).rsqrt()
        normalized = weights * inverse_sqrt[source] * inverse_sqrt[target]
        messages = tangent.index_select(0, source) * normalized[:, None, None]
        output = torch.zeros_like(tangent)
        output.index_add_(0, target, messages)
        return output

    def forward(
        self,
        matrices: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = self._project_upstream_isometry()
        transformed = sym(weight @ matrices @ weight.transpose(-1, -2))
        tangent = spd4gnn_logmap_identity(
            transformed, self.spd4gnn_jitter, self.spd_floor
        )
        tangent = self._normalized_gcn_aggregate(tangent, edge_index, edge_weight)
        output = spd4gnn_expmap_identity(tangent, self.spd4gnn_jitter)
        if self.bias is not None:
            bias = spd4gnn_expmap_identity(sym(self.bias), self.spd4gnn_jitter)
            root = spd4gnn_sqrtmap_identity(
                output, self.spd4gnn_jitter, self.spd_floor
            )
            output = sym(root @ bias @ root)
        return spd_activation(
            output,
            self.activation,
            self.activation_floor,
            self.activation_beta,
            self.activation_jitter,
            self.tgrelu_delta,
        )


class Simple1DiffusionLayer(nn.Module):
    """One exact differentiable simple1 SPD diffusion step."""

    def __init__(
        self,
        beta: float,
        epsilon: float,
        spd_floor: float,
        activation: str = "mild_reeig",
        activation_beta: float = 20.0,
        activation_jitter: float = 0.0,
        tgrelu_delta: float = 1.0e-2,
        activation_floor: float | None = None,
        aggregation: str = "index_add",
        edge_chunk_size: int = 0,
        learnable_beta: bool = False,
        learnable_epsilon: bool = False,
        response: str = "ricci",
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.raw_beta = (
            nn.Parameter(torch.tensor(max(float(beta), 1.0e-8)).log())
            if learnable_beta
            else None
        )
        self.raw_epsilon = (
            nn.Parameter(torch.tensor(max(float(epsilon), 1.0e-8)).log())
            if learnable_epsilon
            else None
        )
        self.spd_floor = float(spd_floor)
        self.activation_floor = (
            float(spd_floor) if activation_floor is None else float(activation_floor)
        )
        self.activation = activation
        self.activation_beta = float(activation_beta)
        self.activation_jitter = float(activation_jitter)
        self.tgrelu_delta = float(tgrelu_delta)
        self.aggregation = aggregation.lower()
        self.edge_chunk_size = int(edge_chunk_size)
        self.response = response.lower()
        if self.aggregation not in {"index_add", "dense"}:
            raise ValueError(f"Unsupported diffusion aggregation: {aggregation}")
        if self.response not in {"ricci", "tangent"}:
            raise ValueError(f"Unsupported diffusion response: {response}")
        if self.edge_chunk_size < 0:
            raise ValueError("edge_chunk_size must be non-negative")

    def _edge_contributions(
        self,
        matrices: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        source_matrices = matrices.index_select(0, source)
        target_matrices = matrices.index_select(0, target)

        inverse_sqrt = spd_invsqrt(source_matrices, self.spd_floor)
        relative = sym(inverse_sqrt @ target_matrices @ inverse_sqrt)
        logged = spd_log(relative, self.spd_floor)
        squared_distance = (logged * logged).sum(dim=(-1, -2))
        beta = self.raw_beta.exp() if self.raw_beta is not None else self.beta
        weights = torch.exp(-beta * squared_distance)
        if edge_weight is not None:
            weights = weights * edge_weight.to(dtype=weights.dtype)
        return logged * weights[:, None, None]

    def forward(
        self,
        matrices: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source, target = edge_index
        epsilon = (
            self.raw_epsilon.exp() if self.raw_epsilon is not None else self.epsilon
        )
        if edge_weight is not None:
            if edge_weight.ndim != 1 or edge_weight.shape[0] != source.shape[0]:
                raise ValueError("edge_weight must have one scalar per directed edge")

        use_chunks = (
            self.edge_chunk_size > 0
            and source.shape[0] > self.edge_chunk_size
        )
        if use_chunks and self.aggregation == "dense":
            raise ValueError("edge_chunk_size is supported with index_add only")
        if use_chunks:
            laplacian = torch.zeros_like(matrices)
            for start in range(0, source.shape[0], self.edge_chunk_size):
                stop = min(start + self.edge_chunk_size, source.shape[0])
                chunk_source = source[start:stop]
                chunk_target = target[start:stop]
                chunk_weight = (
                    edge_weight[start:stop] if edge_weight is not None else None
                )
                if torch.is_grad_enabled() and matrices.requires_grad:
                    def calculate(
                        value,
                        selected_source=chunk_source,
                        selected_target=chunk_target,
                        selected_weight=chunk_weight,
                    ):
                        return self._edge_contributions(
                            value,
                            selected_source,
                            selected_target,
                            selected_weight,
                        )

                    contributions = checkpoint(
                        calculate, matrices, use_reentrant=False
                    )
                else:
                    contributions = self._edge_contributions(
                        matrices, chunk_source, chunk_target, chunk_weight
                    )
                laplacian = torch.index_add(
                    laplacian, 0, chunk_source, contributions
                )
        else:
            contributions = self._edge_contributions(
                matrices, source, target, edge_weight
            )
            if self.aggregation == "index_add":
                laplacian = torch.zeros_like(matrices)
                laplacian.index_add_(0, source, contributions)
            else:
                # This is algebraically identical to index_add, but avoids CUDA
                # atomic summation.  WebKB is small enough that its dense incidence
                # matrix is inexpensive and substantially more reproducible.
                incidence = torch.nn.functional.one_hot(
                    source, num_classes=matrices.shape[0]
                ).transpose(0, 1).to(dtype=matrices.dtype)
                flattened = incidence @ contributions.flatten(start_dim=1)
                laplacian = flattened.reshape_as(matrices)

        if self.response == "ricci":
            ricci = sym(-0.5 * (matrices @ laplacian + laplacian @ matrices))
            raw_next = matrices - epsilon * ricci
        else:
            raw_next = matrices + epsilon * sym(laplacian)
        return spd_activation(
            raw_next,
            self.activation,
            self.activation_floor,
            self.activation_beta,
            self.activation_jitter,
            self.tgrelu_delta,
        )


class SPD4GNNLinearHead(nn.Module):
    """Official Log_I, upper triangle, dropout, linear classifier."""

    def __init__(
        self,
        spd_dim: int,
        num_classes: int,
        dropout: float,
        spd4gnn_jitter: float = 0.0,
    ) -> None:
        super().__init__()
        rows, cols = torch.triu_indices(spd_dim, spd_dim)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        self.spd_floor = 1.0e-6
        self.spd4gnn_jitter = float(spd4gnn_jitter)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(spd_dim * (spd_dim + 1) // 2, num_classes)

    def forward(self, matrices: torch.Tensor) -> torch.Tensor:
        logged = spd4gnn_logmap_identity(
            matrices, self.spd4gnn_jitter, self.spd_floor
        )
        vectors = logged[:, self.rows, self.cols]
        return self.projection(self.dropout(vectors))


class SPD4GNNNCHead(nn.Module):
    """Official SPD4GNN class-conditional Mahalanobis (NC) head."""

    def __init__(
        self, spd_dim: int, num_classes: int, spd4gnn_jitter: float = 0.0
    ) -> None:
        super().__init__()
        rows, cols = torch.triu_indices(spd_dim, spd_dim)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        vector_dim = spd_dim * (spd_dim + 1) // 2
        self.means = nn.Parameter(torch.empty(num_classes, vector_dim))
        self.sigma = nn.Parameter(torch.empty(num_classes, vector_dim, vector_dim))
        self.bias = nn.Parameter(torch.ones(num_classes))
        self.spd_floor = 1.0e-6
        self.spd4gnn_jitter = float(spd4gnn_jitter)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        means_bound = (6.0 / (self.means.shape[-2] + self.means.shape[-1])) ** 0.5
        sigma_bound = (6.0 / (self.sigma.shape[-2] + self.sigma.shape[-1])) ** 0.5
        nn.init.uniform_(self.means, -means_bound, means_bound)
        nn.init.uniform_(self.sigma, -sigma_bound, sigma_bound)
        nn.init.ones_(self.bias)

    def forward(self, matrices: torch.Tensor) -> torch.Tensor:
        logged = spd4gnn_logmap_identity(
            matrices, self.spd4gnn_jitter, self.spd_floor
        )
        vectors = logged[:, self.rows, self.cols]
        offsets = vectors.unsqueeze(1) - self.means.unsqueeze(0)
        metrics = spd4gnn_expmap_identity(self.sigma, self.spd4gnn_jitter)
        distances = (
            offsets.unsqueeze(-2)
            @ metrics.unsqueeze(0)
            @ offsets.unsqueeze(-1)
        ).squeeze(-1).squeeze(-1)
        return -0.5 * distances + self.bias


@dataclass(frozen=True)
class ModelConfig:
    operator: str = "simple1"
    spd_dim: int = 16
    feature_mapper: str = "squared"
    feature_init: str = "random"
    freeze_feature_mapper: bool = False
    layers: int = 1
    feature_dropout: float = 0.0
    classifier_dropout: float = 0.5
    beta: float = 0.01
    epsilon: float = 0.01
    spd_floor: float = 1.0e-4
    # The reference Simple1 SPDNet keeps activation_eps independent of phi.
    # None preserves the historical/default behavior activation_eps == phi.
    activation_floor: float | None = None
    activation: str = "mild_reeig"
    activation_beta: float = 20.0
    activation_jitter: float = 0.0
    tgrelu_delta: float = 1.0e-2
    feature_spd_epsilon: float = 1.0e-4
    # The upstream SPD4GNN implementation uses 1e-3.  Zero preserves the
    # deterministic mathematical-map ablations produced before this audit.
    spd4gnn_jitter: float = 0.0
    bimap_max_delta: float = 0.08
    bimap_decoder: str = "tied_inverse"
    bimap_parameterization: str = "bounded"
    bimap_min_singular_value: float = 0.1
    classifier: str = "linear"
    diffusion_aggregation: str = "index_add"
    # Exact edge-wise diffusion can be checkpointed in chunks for large graphs.
    # Zero retains the original all-at-once implementation.
    edge_chunk_size: int = 0
    learnable_beta: bool = False
    learnable_epsilon: bool = False
    diffusion_response: str = "ricci"
    skip_diffusion: bool = False
    spd4gnn_has_bias: bool = True


class Simple1SPD4GNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, config: ModelConfig) -> None:
        super().__init__()
        if config.layers < 1:
            raise ValueError("layers must be at least one")
        self.config = config
        self.feature_to_spd = SPD4GNNFeatureToSPD(
            input_dim=input_dim,
            spd_dim=config.spd_dim,
            dropout=config.feature_dropout,
            mapper=config.feature_mapper,
            gram_epsilon=config.feature_spd_epsilon,
            spd4gnn_jitter=config.spd4gnn_jitter,
        )
        # The source simple1 SPDNet applies a separately parameterized BiMap,
        # diffusion update, activation, and inverse-BiMap for every layer.
        # L=1 is byte-for-byte the original one-block topology.
        self.bimaps = nn.ModuleList(
            TiedBiMap(
                config.spd_dim,
                config.bimap_max_delta,
                config.bimap_decoder,
                config.bimap_parameterization,
                config.bimap_min_singular_value,
            )
            for _ in range(config.layers)
        )
        self.diffusion = nn.ModuleList(
            Simple1DiffusionLayer(
                config.beta,
                config.epsilon,
                config.spd_floor,
                config.activation,
                config.activation_beta,
                config.activation_jitter,
                config.tgrelu_delta,
                config.activation_floor,
                config.diffusion_aggregation,
                config.edge_chunk_size,
                config.learnable_beta,
                config.learnable_epsilon,
                config.diffusion_response,
            )
            for _ in range(config.layers)
        )
        classifier = config.classifier.lower()
        if classifier == "linear":
            self.output_head = SPD4GNNLinearHead(
                spd_dim=config.spd_dim,
                num_classes=num_classes,
                dropout=config.classifier_dropout,
                spd4gnn_jitter=config.spd4gnn_jitter,
            )
        elif classifier == "nc":
            self.output_head = SPD4GNNNCHead(
                config.spd_dim, num_classes, config.spd4gnn_jitter
            )
        else:
            raise ValueError(f"Unsupported SPD4GNN classifier: {config.classifier}")

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        matrices = self.feature_to_spd(features)
        if self.config.skip_diffusion:
            return self.output_head(matrices)
        for bimap, layer in zip(self.bimaps, self.diffusion):
            latent, weight = bimap.encode(matrices)
            latent = layer(latent, edge_index, edge_weight)
            matrices = bimap.decode(latent, weight)
            matrices = spd_project(matrices, self.config.spd_floor)
        return self.output_head(matrices)


class PureSPD4GNN(nn.Module):
    """Matched pure SPD4GNN baseline using the upstream ``spdgcn`` operator."""

    def __init__(self, input_dim: int, num_classes: int, config: ModelConfig) -> None:
        super().__init__()
        if config.layers < 1:
            raise ValueError("layers must be at least one")
        self.config = config
        self.feature_to_spd = SPD4GNNFeatureToSPD(
            input_dim=input_dim,
            spd_dim=config.spd_dim,
            dropout=config.feature_dropout,
            mapper=config.feature_mapper,
            gram_epsilon=config.feature_spd_epsilon,
            spd4gnn_jitter=config.spd4gnn_jitter,
        )
        self.gcn_layers = nn.ModuleList(
            SPD4GNNGCNLayer(
                config.spd_dim,
                config.spd_floor,
                config.activation,
                config.activation_beta,
                config.activation_jitter,
                config.tgrelu_delta,
                config.activation_floor,
                config.spd4gnn_jitter,
                config.spd4gnn_has_bias,
            )
            for _ in range(config.layers)
        )
        classifier = config.classifier.lower()
        if classifier == "linear":
            self.output_head = SPD4GNNLinearHead(
                config.spd_dim,
                num_classes,
                config.classifier_dropout,
                config.spd4gnn_jitter,
            )
        elif classifier == "nc":
            self.output_head = SPD4GNNNCHead(
                config.spd_dim, num_classes, config.spd4gnn_jitter
            )
        else:
            raise ValueError(f"Unsupported SPD4GNN classifier: {config.classifier}")

    @property
    def bimaps(self) -> nn.ModuleList:
        """Expose propagation modules to the shared optimizer grouping."""
        return self.gcn_layers

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        matrices = self.feature_to_spd(features)
        for layer in self.gcn_layers:
            matrices = layer(matrices, edge_index, edge_weight)
        return self.output_head(matrices)


def build_model(
    input_dim: int, num_classes: int, config: ModelConfig
) -> Simple1SPD4GNN | PureSPD4GNN:
    operator = config.operator.lower().replace("-", "_")
    if operator == "simple1":
        return Simple1SPD4GNN(input_dim, num_classes, config)
    if operator in {"spd4gnn", "spd4gnn_gcn", "spdgcn"}:
        return PureSPD4GNN(input_dim, num_classes, config)
    raise ValueError(f"Unsupported SPD propagation operator: {config.operator}")


_PCA_CACHE: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}


@torch.no_grad()
def initialize_feature_mapper_pca(
    model: Simple1SPD4GNN | PureSPD4GNN, features: torch.Tensor
) -> None:
    """Initialize the unchanged official linear mapper with unlabeled PCA axes."""
    mapper = model.feature_to_spd
    key = (features.data_ptr(), features.shape[1], str(features.device))
    cached = _PCA_CACHE.get(key)
    if cached is None:
        mean = features.mean(dim=0)
        centered = features - mean
        _, _, right_vectors = torch.linalg.svd(centered, full_matrices=False)
        cached = (mean, right_vectors)
        _PCA_CACHE[key] = cached
    mean, right_vectors = cached

    projection = mapper.dimred
    # A tiny deterministic perturbation prevents repeated identity eigenspaces
    # in SPD dimensions beyond the empirical feature rank.
    projection.weight.normal_(mean=0.0, std=1.0e-5)
    projection.bias.normal_(mean=0.0, std=1.0e-5)
    if mapper.mapper == "squared":
        rows, cols = torch.triu_indices(mapper.spd_dim, mapper.spd_dim, device=features.device)
        capacity = len(rows)
        rank = min(capacity, right_vectors.shape[0])
        for index in range(rank):
            row = int(rows[index])
            col = int(cols[index])
            component = right_vectors[index]
            first = row * mapper.spd_dim + col
            projection.weight[first].copy_(component)
            projection.bias[first].copy_(-(component @ mean))
            if row != col:
                second = col * mapper.spd_dim + row
                projection.weight[second].copy_(component)
                projection.bias[second].copy_(-(component @ mean))
    else:
        rank = min(projection.out_features, right_vectors.shape[0])
        projection.weight[:rank].copy_(right_vectors[:rank])
        projection.bias[:rank].copy_(-(right_vectors[:rank] @ mean))


@torch.no_grad()
def initialize_feature_mapper_xavier(
    model: Simple1SPD4GNN | PureSPD4GNN,
) -> None:
    """Match the local Gram FeatureToSPD projection initialization exactly."""
    mapper = model.feature_to_spd
    if mapper.mapper != "gram":
        raise ValueError("Xavier FeatureToSPD initialization is only defined for Gram mapping")
    gain = nn.init.calculate_gain("leaky_relu", mapper.gram_activation.negative_slope)
    nn.init.xavier_uniform_(mapper.dimred.weight, gain=gain)
    nn.init.zeros_(mapper.dimred.bias)
