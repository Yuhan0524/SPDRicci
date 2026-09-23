"""Full-batch training with validation-only checkpoint selection."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import DEFAULT_DATA_ROOT, gnrf_reference_split, load_webkb
from model import (
    ModelConfig,
    build_model,
    initialize_feature_mapper_pca,
    initialize_feature_mapper_xavier,
)


PAPER_TARGETS = {
    "cornell": 87.28,
    "wisconsin": 88.00,
    "texas": 87.39,
    "roman-empire": 86.25,
    "tolokers": 83.96,
    "minesweeper": 95.03,
    "questions": 73.86,
    "amazon-ratings": 46.89,
    "cora-full": 72.12,
    "pubmed": 90.37,
    "dblp": 85.73,
    "cora-ml": 89.18,
}

ROC_AUC_DATASETS = {"tolokers", "minesweeper", "questions"}


def paper_standard_deviation(values) -> float:
    """Match released GNRF's Bessel-corrected ``torch.std`` reporting."""
    values = np.asarray(values)
    return float(values.std(ddof=1)) if values.size > 1 else 0.0


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 0.01
    weight_decay: float = 5.0e-4
    epochs: int = 300
    optimizer: str = "adamw"
    label_smoothing: float = 0.0
    scheduler: str = "none"
    minimum_lr_ratio: float = 0.01
    step_size: int = 100
    scheduler_gamma: float = 0.5
    gradient_clip: float = 5.0
    sam_rho: float = 0.0
    sam_adaptive: bool = False
    mixup_alpha: float = 0.0
    input_noise_std: float = 0.0
    edge_dropout: float = 0.0
    feature_lr_ratio: float = 1.0
    bimap_lr_ratio: float = 1.0
    class_weight_power: float = 0.0
    center_loss_weight: float = 0.0
    supervised_contrastive_weight: float = 0.0
    supervised_contrastive_temperature: float = 0.2
    auc_pairwise_weight: float = 0.0
    auc_pairwise_pairs: int = 8192
    auc_pairwise_margin: float = 0.0
    ema_decay: float = 0.0
    ema_start_epoch: int = 0
    momentum: float = 0.9
    head_finetune_epochs: int = 0
    head_finetune_lr: float = 0.001
    head_finetune_weight_decay: float = 0.0
    feature_weight_init_scale: float = 1.0
    feature_bias_init_scale: float = 1.0
    feature_mode: str = "original"
    feature_hops: int = 2
    feature_ppr_alpha: float = 0.1
    graph_mode: str = "original"
    graph_topk: int = 0
    graph_weight_mode: str = "none"
    graph_weight_temperature: float = 1.0
    eval_samples: int = 1
    eval_aggregation: str = "logits"
    deterministic: bool = False
    precision: str = "float32"


_EDGE_INDEX_CACHE: dict[tuple[int, int, str, str, int], torch.Tensor] = {}
_EDGE_WEIGHT_CACHE: dict[tuple[int, int, str, str, float], torch.Tensor | None] = {}
_FEATURE_CACHE: dict[tuple[int, int, str, str, int, float], torch.Tensor] = {}
GNRF_REWIRING_PATH = (
    Path(__file__).resolve().parent / "data" / "gnrf_rewiring.npz"
)


@torch.no_grad()
def model_input_features(
    data, mode: str = "original", hops: int = 2, ppr_alpha: float = 0.1
) -> torch.Tensor:
    """Return fixed, label-free graph features for the unchanged SPD model.

    ``original`` is the default training path. The concatenation modes
    are an explicitly separate preprocessing branch inspired by SIGN/FSGNN:
    they concatenate X with normalized-adjacency propagations and never inspect
    labels or any train/validation/test mask.
    """
    normalized = mode.lower()

    def normalized_base(kind: str) -> torch.Tensor:
        if kind == "signed":
            value = data.x
        elif kind == "absolute":
            value = data.x.abs()
        elif kind == "square":
            value = data.x.square()
        elif kind == "signed_sqrt":
            value = data.x.sign() * data.x.abs().sqrt()
        elif kind == "signed_log":
            value = data.x.sign() * data.x.abs().log1p()
        elif kind == "zscore":
            value = data.x - data.x.mean(dim=0, keepdim=True)
            value = value / data.x.std(
                dim=0, keepdim=True, unbiased=False
            ).clamp_min(1.0e-6)
        elif kind == "canonical":
            value = F.normalize(data.x, p=2.0, dim=1)
            dominant = value.abs().argmax(dim=1, keepdim=True)
            orientation = value.gather(1, dominant).sign()
            orientation[orientation == 0] = 1.0
            value = value * orientation
        else:
            raise ValueError(f"Unsupported base feature transform: {kind}")
        return F.normalize(value, p=2.0, dim=1)

    if normalized == "original":
        return data.x
    if normalized == "l2":
        return normalized_base("signed")
    if normalized == "abs_l2":
        return normalized_base("absolute")
    simple_transforms = {
        "square_l2": "square",
        "signed_sqrt_l2": "signed_sqrt",
        "signed_log_l2": "signed_log",
        "zscore_l2": "zscore",
        "canonical_l2": "canonical",
    }
    if normalized in simple_transforms:
        return normalized_base(simple_transforms[normalized])
    if normalized == "signed_abs_l2_concat":
        return torch.cat(
            (
                normalized_base("signed"),
                normalized_base("absolute"),
            ),
            dim=1,
        ).contiguous()
    supported = {
        "adjacency_concat",
        "adjacency_self_concat",
        "adjacency_l2_concat",
        "adjacency_self_l2_concat",
        "random_walk_l2_concat",
        "random_walk_self_l2_concat",
        "ppr_l2_concat",
        "ppr_self_l2_concat",
        "adjacency_abs_l2_concat",
        "adjacency_self_abs_l2_concat",
        "adjacency_signed_abs_l2_concat",
        "adjacency_self_signed_abs_l2_concat",
        "adjacency_self_canonical_l2_concat",
        "adjacency_self_canonical_abs_l2_concat",
        "adjacency_self_square_l2_concat",
        "adjacency_self_tfidf_l2_concat",
        "adjacency_self_signed_sqrt_l2_concat",
        "adjacency_self_signed_log_l2_concat",
        "adjacency_self_zscore_l2_concat",
        "adjacency_self_degree_features_l2_concat",
        "adjacency_self_rich_degree_features_l2_concat",
        "adjacency_self_signed_abs_degree_features_l2_concat",
        "adjacency_self_l2_laplacian16_concat",
        "laplacian_pe_concat",
        "laplacian_high_pe_concat",
        "rwse_concat",
        "role_concat",
    }
    if normalized not in supported:
        raise ValueError(f"Unsupported feature mode: {mode}")
    if hops < 1:
        raise ValueError("feature_hops must be positive for propagated features")
    if not 0.0 < ppr_alpha <= 1.0:
        raise ValueError("ppr_alpha must be in (0, 1]")
    key = (
        data.x.data_ptr(),
        data.edge_index.data_ptr(),
        str(data.x.device),
        normalized,
        int(hops),
        float(ppr_alpha),
    )
    cached = _FEATURE_CACHE.get(key)
    if cached is not None:
        return cached

    if normalized == "adjacency_self_l2_laplacian16_concat":
        propagated = model_input_features(
            data, "adjacency_self_l2_concat", hops, ppr_alpha
        )
        positional = model_input_features(
            data, "laplacian_pe_concat", 16, ppr_alpha
        )
        # laplacian_pe_concat begins with the normalized original feature
        # block; append only its structural coordinates to avoid duplication.
        structural = positional[:, data.x.shape[1] :]
        features = torch.cat((propagated, structural), dim=1).contiguous()
        _FEATURE_CACHE[key] = features
        return features

    if normalized in {"laplacian_pe_concat", "laplacian_high_pe_concat"} and data.num_nodes > 5_000:
        # The dense reference path below is exact and convenient for WebKB,
        # but O(N^2) storage is unnecessary on citation graphs.  Compute the
        # same normalized-Laplacian eigenspace with SciPy's sparse solver.
        from scipy import sparse
        from scipy.sparse.linalg import eigsh

        edge_index = data.edge_index.detach().cpu().numpy()
        values = np.ones(edge_index.shape[1], dtype=np.float64)
        adjacency = sparse.coo_matrix(
            (values, (edge_index[0], edge_index[1])),
            shape=(data.num_nodes, data.num_nodes),
        ).tocsr()
        adjacency = adjacency.maximum(adjacency.transpose()).tocsr()
        adjacency.setdiag(0.0)
        adjacency.eliminate_zeros()
        degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        inverse_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1.0))
        normalized_adjacency = (
            sparse.diags(inverse_sqrt)
            @ adjacency
            @ sparse.diags(inverse_sqrt)
        ).tocsr()
        width = min(int(hops), data.num_nodes - 2)
        if normalized == "laplacian_pe_concat":
            _, vectors = eigsh(
                normalized_adjacency,
                k=width + 1,
                which="LA",
                v0=np.ones(data.num_nodes, dtype=np.float64),
                tol=1.0e-5,
            )
            # Largest normalized-adjacency vector is the trivial lowest
            # Laplacian mode.  Retain the following low-frequency modes.
            values_rayleigh = np.sum(
                vectors * (normalized_adjacency @ vectors), axis=0
            )
            order = np.argsort(values_rayleigh)[::-1]
            structural_np = vectors[:, order[1 : width + 1]]
        else:
            _, vectors = eigsh(
                normalized_adjacency,
                k=width,
                which="SA",
                v0=np.ones(data.num_nodes, dtype=np.float64),
                tol=1.0e-5,
            )
            structural_np = vectors
        structural = torch.as_tensor(
            structural_np.copy(), dtype=data.x.dtype, device=data.x.device
        )
        largest = structural.abs().argmax(dim=0)
        column = torch.arange(width, device=data.x.device)
        signs = structural[largest, column].sign()
        signs[signs == 0] = 1.0
        structural = structural * signs
        structural = structural - structural.mean(dim=0, keepdim=True)
        structural = structural / structural.std(
            dim=0, keepdim=True, unbiased=False
        ).clamp_min(1.0e-6)
        structural = F.normalize(structural, p=2.0, dim=1)
        base = F.normalize(data.x, p=2.0, dim=1)
        features = torch.cat((base, structural), dim=1).contiguous()
        _FEATURE_CACHE[key] = features
        return features

    if normalized in {
        "laplacian_pe_concat",
        "laplacian_high_pe_concat",
        "rwse_concat",
        "role_concat",
    }:
        adjacency = torch.zeros(
            (data.num_nodes, data.num_nodes),
            dtype=data.x.dtype,
            device=data.x.device,
        )
        adjacency[data.edge_index[0], data.edge_index[1]] = 1.0
        adjacency.fill_diagonal_(0.0)
        raw_adjacency = adjacency
        degree = raw_adjacency.sum(dim=1)
        base = F.normalize(data.x, p=2.0, dim=1)
        if normalized in {"laplacian_pe_concat", "laplacian_high_pe_concat"}:
            inverse_sqrt = degree.clamp_min(1.0).rsqrt()
            normalized_adjacency = (
                inverse_sqrt[:, None]
                * raw_adjacency
                * inverse_sqrt[None, :]
            )
            laplacian = torch.eye(
                data.num_nodes, dtype=data.x.dtype, device=data.x.device
            ) - normalized_adjacency
            values, vectors = torch.linalg.eigh(laplacian)
            width = min(int(hops), data.num_nodes - 1)
            if normalized == "laplacian_pe_concat":
                structural = vectors[:, 1 : width + 1]
            else:
                structural = vectors[:, -width:]
            largest = structural.abs().argmax(dim=0)
            column = torch.arange(width, device=data.x.device)
            signs = structural[largest, column].sign()
            signs[signs == 0] = 1.0
            structural = structural * signs
        elif normalized == "rwse_concat":
            transition = raw_adjacency / degree.clamp_min(1.0)[:, None]
            current = transition
            returns = []
            for _ in range(int(hops)):
                returns.append(current.diagonal())
                current = current @ transition
            structural = torch.stack(returns, dim=1)
        else:
            neighbor_degree_sum = raw_adjacency @ degree
            neighbor_count = degree.clamp_min(1.0)
            neighbor_degree_mean = neighbor_degree_sum / neighbor_count
            neighbor_degree_square_mean = (
                raw_adjacency @ degree.square()
            ) / neighbor_count
            neighbor_degree_std = (
                neighbor_degree_square_mean - neighbor_degree_mean.square()
            ).clamp_min(0.0).sqrt()
            two_hop = (raw_adjacency @ raw_adjacency) > 0
            two_hop.fill_diagonal_(False)
            two_hop_count = two_hop.sum(dim=1).to(data.x.dtype)
            triangles = (
                (raw_adjacency @ raw_adjacency) * raw_adjacency
            ).sum(dim=1) / 2.0
            clustering = 2.0 * triangles / (
                degree * (degree - 1.0)
            ).clamp_min(1.0)
            pagerank = torch.full_like(degree, 1.0 / data.num_nodes)
            transition = raw_adjacency / degree.clamp_min(1.0)[:, None]
            for _ in range(max(10, int(hops) * 5)):
                pagerank = (
                    0.15 / data.num_nodes + 0.85 * transition.transpose(0, 1) @ pagerank
                )
            structural = torch.stack(
                (
                    degree,
                    degree.add(1.0).log(),
                    neighbor_degree_mean,
                    neighbor_degree_std,
                    two_hop_count,
                    triangles,
                    clustering,
                    pagerank,
                ),
                dim=1,
            )
        structural = structural - structural.mean(dim=0, keepdim=True)
        structural = structural / structural.std(
            dim=0, keepdim=True, unbiased=False
        ).clamp_min(1.0e-6)
        structural = F.normalize(structural, p=2.0, dim=1)
        features = torch.cat((base, structural), dim=1).contiguous()
        _FEATURE_CACHE[key] = features
        return features
    # Propagated feature blocks only require sparse matrix multiplication.
    # Avoid materializing an O(N^2) dense adjacency for the citation and
    # Heterophilous Graph Benchmark datasets.
    if normalized in {
        "adjacency_concat",
        "adjacency_self_concat",
        "adjacency_l2_concat",
        "adjacency_self_l2_concat",
        "random_walk_l2_concat",
        "random_walk_self_l2_concat",
        "ppr_l2_concat",
        "ppr_self_l2_concat",
        "adjacency_abs_l2_concat",
        "adjacency_self_abs_l2_concat",
        "adjacency_signed_abs_l2_concat",
        "adjacency_self_signed_abs_l2_concat",
        "adjacency_self_canonical_l2_concat",
        "adjacency_self_canonical_abs_l2_concat",
        "adjacency_self_square_l2_concat",
        "adjacency_self_tfidf_l2_concat",
        "adjacency_self_signed_sqrt_l2_concat",
        "adjacency_self_signed_log_l2_concat",
        "adjacency_self_zscore_l2_concat",
        "adjacency_self_degree_features_l2_concat",
        "adjacency_self_rich_degree_features_l2_concat",
        "adjacency_self_signed_abs_degree_features_l2_concat",
    }:
        source, target = data.edge_index
        if normalized in {
            "adjacency_self_concat",
            "adjacency_self_l2_concat",
            "random_walk_self_l2_concat",
            "ppr_self_l2_concat",
            "adjacency_self_abs_l2_concat",
            "adjacency_self_signed_abs_l2_concat",
            "adjacency_self_canonical_l2_concat",
            "adjacency_self_canonical_abs_l2_concat",
            "adjacency_self_square_l2_concat",
            "adjacency_self_tfidf_l2_concat",
            "adjacency_self_signed_sqrt_l2_concat",
            "adjacency_self_signed_log_l2_concat",
            "adjacency_self_zscore_l2_concat",
            "adjacency_self_degree_features_l2_concat",
            "adjacency_self_rich_degree_features_l2_concat",
            "adjacency_self_signed_abs_degree_features_l2_concat",
        }:
            nodes = torch.arange(data.num_nodes, device=data.edge_index.device)
            source = torch.cat((source, nodes))
            target = torch.cat((target, nodes))
        values = torch.ones(source.shape[0], dtype=data.x.dtype, device=data.x.device)
        degree = torch.zeros(data.num_nodes, dtype=data.x.dtype, device=data.x.device)
        degree.index_add_(0, source, values)
        if normalized.startswith("random_walk") or normalized.startswith("ppr"):
            values = values / degree.clamp_min(1.0).index_select(0, source)
        else:
            inverse_sqrt = degree.clamp_min(1.0).rsqrt()
            values = values * inverse_sqrt[source] * inverse_sqrt[target]
        propagation = torch.sparse_coo_tensor(
            torch.stack((source, target)),
            values,
            (data.num_nodes, data.num_nodes),
            dtype=data.x.dtype,
            device=data.x.device,
        ).coalesce()
        normalize_blocks = normalized in {
            "adjacency_l2_concat",
            "adjacency_self_l2_concat",
            "random_walk_l2_concat",
            "random_walk_self_l2_concat",
            "ppr_l2_concat",
            "ppr_self_l2_concat",
            "adjacency_abs_l2_concat",
            "adjacency_self_abs_l2_concat",
            "adjacency_signed_abs_l2_concat",
            "adjacency_self_signed_abs_l2_concat",
            "adjacency_self_canonical_l2_concat",
            "adjacency_self_canonical_abs_l2_concat",
            "adjacency_self_square_l2_concat",
            "adjacency_self_tfidf_l2_concat",
            "adjacency_self_signed_sqrt_l2_concat",
            "adjacency_self_signed_log_l2_concat",
            "adjacency_self_zscore_l2_concat",
            "adjacency_self_degree_features_l2_concat",
            "adjacency_self_rich_degree_features_l2_concat",
            "adjacency_self_signed_abs_degree_features_l2_concat",
        }
        signed_abs = "signed_abs" in normalized
        canonical_abs = "canonical_abs" in normalized
        absolute = "abs" in normalized and not (signed_abs or canonical_abs)
        if signed_abs:
            currents = [
                data.x,
                data.x.abs(),
            ]
        elif canonical_abs:
            currents = [
                normalized_base("canonical"),
                data.x.abs(),
            ]
        elif "canonical" in normalized:
            currents = [normalized_base("canonical")]
        elif "square" in normalized:
            currents = [data.x.square()]
        elif "tfidf" in normalized:
            document_frequency = (data.x != 0).sum(dim=0).to(data.x.dtype)
            inverse_document_frequency = (
                (1.0 + data.num_nodes) / (1.0 + document_frequency)
            ).log().add_(1.0)
            currents = [data.x * inverse_document_frequency]
        elif "signed_sqrt" in normalized:
            currents = [data.x.sign() * data.x.abs().sqrt()]
        elif "signed_log" in normalized:
            currents = [data.x.sign() * data.x.abs().log1p()]
        elif "zscore" in normalized:
            standardized = data.x - data.x.mean(dim=0, keepdim=True)
            standardized = standardized / data.x.std(
                dim=0, keepdim=True, unbiased=False
            ).clamp_min(1.0e-6)
            currents = [standardized]
        elif absolute:
            currents = [data.x.abs()]
        else:
            currents = [data.x]
        if "degree_features" in normalized:
            raw_source, raw_target = data.edge_index
            raw_degree = torch.zeros(
                data.num_nodes, dtype=data.x.dtype, device=data.x.device
            )
            raw_degree.index_add_(
                0, raw_source, torch.ones_like(raw_source, dtype=data.x.dtype)
            )
            neighbor_sum = torch.zeros_like(raw_degree)
            neighbor_square_sum = torch.zeros_like(raw_degree)
            neighbor_sum.index_add_(0, raw_source, raw_degree[raw_target])
            neighbor_square_sum.index_add_(
                0, raw_source, raw_degree[raw_target].square()
            )
            neighbor_mean = neighbor_sum / raw_degree.clamp_min(1.0)
            neighbor_variance = (
                neighbor_square_sum / raw_degree.clamp_min(1.0)
                - neighbor_mean.square()
            ).clamp_min(0.0)
            columns = [
                raw_degree,
                raw_degree.add(1.0).log(),
                neighbor_mean,
                neighbor_variance.sqrt(),
            ]
            if "rich_degree_features" in normalized:
                pagerank = torch.full_like(raw_degree, 1.0 / data.num_nodes)
                for _ in range(30):
                    updated = torch.full_like(raw_degree, 0.15 / data.num_nodes)
                    updated.index_add_(
                        0,
                        raw_source,
                        0.85
                        * pagerank[raw_target]
                        / raw_degree[raw_target].clamp_min(1.0),
                    )
                    pagerank = updated
                columns.extend(
                    (
                        raw_degree.sqrt(),
                        raw_degree.add(1.0).rsqrt(),
                        pagerank,
                    )
                )
            structural = torch.stack(columns, dim=1)
            structural = structural - structural.mean(dim=0, keepdim=True)
            structural = structural / structural.std(
                dim=0, keepdim=True, unbiased=False
            ).clamp_min(1.0e-6)
            currents.append(structural)
        blocks = [
            F.normalize(current, p=2.0, dim=1)
            if normalize_blocks
            else current
            for current in currents
        ]
        for _ in range(int(hops)):
            next_currents = []
            for current_index, current in enumerate(currents):
                current = torch.sparse.mm(propagation, current)
                if normalized.startswith("ppr"):
                    restart = data.x if current_index == 0 else data.x.abs()
                    current = (1.0 - ppr_alpha) * current + ppr_alpha * restart
                next_currents.append(current)
                blocks.append(
                    F.normalize(current, p=2.0, dim=1)
                    if normalize_blocks
                    else current
                )
            currents = next_currents
        features = torch.cat(blocks, dim=1).contiguous()
        _FEATURE_CACHE[key] = features
        return features
    raise AssertionError(f"Unhandled supported feature mode: {mode}")


def drop_undirected_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Drop whole undirected edge pairs for training-time augmentation."""
    if not 0.0 <= probability < 1.0:
        raise ValueError("edge_dropout must be in [0, 1)")
    if probability == 0.0 or edge_index.shape[1] == 0:
        return edge_index, edge_weight
    source, target = edge_index
    lower = torch.minimum(source, target)
    upper = torch.maximum(source, target)
    keys = lower * int(num_nodes) + upper
    _, inverse = torch.unique(keys, sorted=False, return_inverse=True)
    pair_count = int(inverse.max().item() + 1)
    pair_keep = torch.rand(pair_count, device=edge_index.device) >= probability
    keep = pair_keep.index_select(0, inverse)
    dropped_weight = edge_weight[keep] if edge_weight is not None else None
    return edge_index[:, keep], dropped_weight


@torch.no_grad()
def diffusion_edge_index(data, mode: str, topk: int = 0) -> torch.Tensor:
    """Build a label-free diffusion graph while leaving the model unchanged."""
    normalized = mode.lower()
    if normalized == "original":
        return data.edge_index
    supported = {
        "exact_two_hop",
        "within_two_hop",
        "exact_two_hop_topk",
        "within_two_hop_topk",
        "feature_knn",
        "feature_mutual_knn",
        "gnrf_sdrf",
        "gnrf_fosr",
        "gnrf_borf",
    }
    if normalized not in supported:
        raise ValueError(f"Unsupported graph mode: {mode}")
    if (
        normalized.endswith("_topk")
        or normalized in {"feature_knn", "feature_mutual_knn"}
    ) and topk < 1:
        raise ValueError("graph_topk must be positive for a top-k graph mode")
    key = (
        data.edge_index.data_ptr(),
        data.num_nodes,
        str(data.edge_index.device),
        normalized,
        int(topk),
    )
    cached = _EDGE_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    adjacency = torch.zeros(
        (data.num_nodes, data.num_nodes),
        dtype=torch.bool,
        device=data.edge_index.device,
    )
    adjacency[data.edge_index[0], data.edge_index[1]] = True
    adjacency.fill_diagonal_(False)
    if normalized.startswith("gnrf_"):
        if not data.name:
            raise ValueError("GNRF rewiring requires a named dataset")
        rewire_name = normalized.removeprefix("gnrf_")
        key_name = f"{rewire_name}_{data.name}"
        with np.load(GNRF_REWIRING_PATH) as rewired:
            if key_name not in rewired:
                raise ValueError(f"Released GNRF rewiring is unavailable: {key_name}")
            released = torch.as_tensor(
                rewired[key_name], dtype=torch.long, device=data.edge_index.device
            )
        selected = torch.zeros_like(adjacency)
        selected[released[0], released[1]] = True
        selected = selected | selected.transpose(0, 1)
        selected.fill_diagonal_(False)
        edge_index = selected.nonzero(as_tuple=False).transpose(0, 1).contiguous()
        _EDGE_INDEX_CACHE[key] = edge_index
        return edge_index
    if normalized in {"feature_knn", "feature_mutual_knn"}:
        normalized_features = data.x / data.x.norm(
            dim=1, keepdim=True
        ).clamp_min(1.0e-12)
        similarities = normalized_features @ normalized_features.transpose(0, 1)
        similarities.fill_diagonal_(float("-inf"))
        width = min(int(topk), data.num_nodes - 1)
        neighbors = similarities.topk(width, dim=1).indices
        selected = torch.zeros_like(adjacency)
        rows = torch.arange(data.num_nodes, device=data.edge_index.device)
        selected[rows.unsqueeze(1), neighbors] = True
        if normalized == "feature_knn":
            selected = selected | selected.transpose(0, 1)
        else:
            selected = selected & selected.transpose(0, 1)
        edge_index = selected.nonzero(as_tuple=False).transpose(0, 1).contiguous()
        _EDGE_INDEX_CACHE[key] = edge_index
        return edge_index
    common = adjacency.to(dtype=data.x.dtype) @ adjacency.to(dtype=data.x.dtype)
    two_hop = common > 0
    two_hop.fill_diagonal_(False)
    exact = two_hop & ~adjacency
    if normalized == "exact_two_hop":
        selected = exact
    elif normalized == "within_two_hop":
        selected = two_hop | adjacency
    else:
        candidates = exact if normalized == "exact_two_hop_topk" else two_hop
        node_index = torch.arange(data.num_nodes, device=data.edge_index.device)
        scores = common * float(data.num_nodes + 1) + (
            data.num_nodes - node_index
        ).to(common.dtype).unsqueeze(0)
        scores = scores.masked_fill(~candidates, float("-inf"))
        width = min(int(topk), data.num_nodes)
        top_scores, top_indices = scores.topk(width, dim=1)
        rows = node_index.unsqueeze(1).expand_as(top_indices)[torch.isfinite(top_scores)]
        cols = top_indices[torch.isfinite(top_scores)]
        selected = torch.zeros_like(adjacency)
        selected[rows, cols] = True
        selected = selected | selected.transpose(0, 1)
        if normalized == "within_two_hop_topk":
            selected |= adjacency
    edge_index = selected.nonzero(as_tuple=False).transpose(0, 1).contiguous()
    _EDGE_INDEX_CACHE[key] = edge_index
    return edge_index


@torch.no_grad()
def diffusion_edge_weight(
    data,
    edge_index: torch.Tensor,
    mode: str,
    temperature: float = 1.0,
) -> torch.Tensor | None:
    """Return label-free Simple1 edge multipliers.

    The upstream Simple1 equation multiplies its heat-kernel weight by an
    optional edge attribute.  ``none`` exactly preserves the unweighted
    released-dataset path.  All other modes below depend only on supplied node
    features or graph degree and are therefore safe for validation-only model
    selection.
    """
    normalized = mode.lower()
    if normalized == "none":
        return None
    supported = {"cosine", "degree", "cosine_degree"}
    if normalized not in supported:
        raise ValueError(f"Unsupported graph weight mode: {mode}")
    if temperature <= 0:
        raise ValueError("graph_weight_temperature must be positive")
    key = (
        data.x.data_ptr(),
        edge_index.data_ptr(),
        str(edge_index.device),
        normalized,
        float(temperature),
    )
    if key in _EDGE_WEIGHT_CACHE:
        return _EDGE_WEIGHT_CACHE[key]
    source, target = edge_index
    weights = torch.ones(source.shape[0], dtype=data.x.dtype, device=data.x.device)
    if normalized in {"cosine", "cosine_degree"}:
        unit = data.x / data.x.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
        similarity = (unit[source] * unit[target]).sum(dim=1).clamp_min(0.0)
        weights = similarity.pow(float(temperature))
    if normalized in {"degree", "cosine_degree"}:
        denominator = torch.zeros(
            data.num_nodes, dtype=weights.dtype, device=weights.device
        )
        denominator.index_add_(0, source, weights)
        weights = weights / denominator.index_select(0, source).clamp_min(1.0e-12)
    _EDGE_WEIGHT_CACHE[key] = weights
    return weights


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().item() * 100.0)


def binary_roc_auc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Binary ROC-AUC in percent, with average ranks for tied scores."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("binary ROC-AUC requires two-class logits")
    labels = labels.flatten()
    positive = labels == 1
    negative = labels == 0
    positive_count = int(positive.sum().item())
    negative_count = int(negative.sum().item())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("binary ROC-AUC requires both classes")

    # The class-1 logit margin induces the same ranking as its softmax score.
    scores = logits[:, 1] - logits[:, 0]
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    _, inverse, counts = torch.unique_consecutive(
        sorted_scores, return_inverse=True, return_counts=True
    )
    cumulative = counts.cumsum(dim=0)
    average_ranks = (
        cumulative.to(scores.dtype)
        + (cumulative - counts + 1).to(scores.dtype)
    ) / 2.0
    ranks = torch.empty_like(scores)
    ranks[order] = average_ranks[inverse]
    positive_rank_sum = ranks[positive].sum()
    mann_whitney = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return float(
        (mann_whitney / (positive_count * negative_count)).item() * 100.0
    )


def metric_name(dataset_name: str) -> str:
    return "roc_auc" if dataset_name in ROC_AUC_DATASETS else "accuracy"


def classification_score(
    dataset_name: str, logits: torch.Tensor, labels: torch.Tensor
) -> float:
    if dataset_name in ROC_AUC_DATASETS:
        return binary_roc_auc(logits, labels)
    return accuracy(logits, labels)


def pairwise_auc_surrogate(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_pairs: int = 8192,
    margin: float = 0.0,
) -> torch.Tensor:
    """Sampled logistic ranking surrogate for a binary ROC-AUC objective.

    Only training labels are supplied by the caller.  Sampling avoids the
    quadratic positive-by-negative matrix while retaining an unbiased pair
    distribution.  This changes the training loss, never the model forward
    architecture or evaluation metric.
    """
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("pairwise AUC surrogate requires two-class logits")
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    labels = labels.flatten()
    scores = logits[:, 1] - logits[:, 0]
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("pairwise AUC surrogate requires both classes")
    pair_count = min(int(max_pairs), max(positive.numel(), negative.numel()))
    positive_index = torch.randint(
        positive.numel(), (pair_count,), device=logits.device
    )
    negative_index = torch.randint(
        negative.numel(), (pair_count,), device=logits.device
    )
    differences = positive[positive_index] - negative[negative_index]
    return F.softplus(float(margin) - differences).mean()


def supervised_center_loss(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Compact same-class training logits without changing inference."""
    centers = logits.new_zeros((num_classes, logits.shape[-1]))
    counts = logits.new_zeros(num_classes)
    centers.index_add_(0, labels, logits)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=logits.dtype))
    centers = centers / counts.clamp_min(1.0).unsqueeze(-1)
    offsets = logits - centers.index_select(0, labels)
    return offsets.square().mean()


def supervised_contrastive_loss(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Class-balanced supervised contrastive regularizer on training logits.

    This changes only the training objective; inference and the fixed
    Simple1/SPD4GNN forward architecture are untouched.  Anchors whose class
    has no second example are excluded instead of producing a spurious loss.
    """
    if temperature <= 0:
        raise ValueError("supervised_contrastive_temperature must be positive")
    if logits.shape[0] < 2:
        return logits.sum() * 0.0
    normalized = F.normalize(logits, p=2.0, dim=-1)
    similarities = normalized @ normalized.transpose(0, 1)
    similarities = similarities / float(temperature)
    identity = torch.eye(
        logits.shape[0], dtype=torch.bool, device=logits.device
    )
    denominator_logits = similarities.masked_fill(identity, float("-inf"))
    log_probabilities = similarities - torch.logsumexp(
        denominator_logits, dim=1, keepdim=True
    )
    positives = labels[:, None].eq(labels[None, :]) & ~identity
    positive_count = positives.sum(dim=1)
    usable = positive_count > 0
    if not bool(usable.any()):
        return logits.sum() * 0.0
    per_anchor = -(
        log_probabilities.masked_fill(~positives, 0.0).sum(dim=1)
        / positive_count.clamp_min(1)
    )
    return per_anchor[usable].mean()


def averaged_eval_logits(
    model,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    samples: int,
    aggregation: str,
) -> torch.Tensor:
    """Average stochastic SPD4GNN Exp/Log evaluations without changing the model."""
    if samples < 1:
        raise ValueError("eval_samples must be positive")
    normalized = aggregation.lower()
    if normalized not in {"logits", "probabilities"}:
        raise ValueError(f"Unsupported eval aggregation: {aggregation}")
    draws = [model(features, edge_index, edge_weight) for _ in range(samples)]
    stacked = torch.stack(draws)
    if normalized == "logits":
        return stacked.mean(dim=0)
    # Log mean probabilities are valid inputs to cross_entropy and preserve
    # the predictive distribution for both loss and argmax accuracy.
    return torch.logsumexp(stacked.log_softmax(dim=-1), dim=0) - np.log(samples)


def train_seed(
    data,
    seed: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
    include_test: bool,
    initialization_seed: int | None = None,
    return_logits: bool = False,
) -> dict:
    if not 0.0 <= train_config.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if train_config.ema_start_epoch < 0:
        raise ValueError("ema_start_epoch must be non-negative")
    if train_config.sam_rho < 0:
        raise ValueError("sam_rho must be non-negative")
    if train_config.mixup_alpha < 0:
        raise ValueError("mixup_alpha must be non-negative")
    if train_config.input_noise_std < 0:
        raise ValueError("input_noise_std must be non-negative")
    if train_config.supervised_contrastive_weight < 0:
        raise ValueError("supervised_contrastive_weight must be non-negative")
    if train_config.supervised_contrastive_temperature <= 0:
        raise ValueError("supervised_contrastive_temperature must be positive")
    if train_config.auc_pairwise_weight < 0:
        raise ValueError("auc_pairwise_weight must be non-negative")
    if train_config.auc_pairwise_pairs < 1:
        raise ValueError("auc_pairwise_pairs must be positive")
    if train_config.auc_pairwise_weight > 0 and data.num_classes != 2:
        raise ValueError("pairwise AUC training is only defined for binary datasets")
    if not 0.0 <= train_config.edge_dropout < 1.0:
        raise ValueError("edge_dropout must be in [0, 1)")
    torch.use_deterministic_algorithms(train_config.deterministic)
    set_seed(seed)
    train_index, validation_index, test_index = gnrf_reference_split(data.num_nodes, seed)
    train_index = train_index.to(device)
    validation_index = validation_index.to(device)
    test_index = test_index.to(device)
    if initialization_seed is not None:
        set_seed(initialization_seed)

    model_edge_index = diffusion_edge_index(
        data, train_config.graph_mode, train_config.graph_topk
    )
    model_edge_weight = diffusion_edge_weight(
        data,
        model_edge_index,
        train_config.graph_weight_mode,
        train_config.graph_weight_temperature,
    )
    dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }.get(train_config.precision.lower())
    if dtype is None:
        raise ValueError(f"Unsupported precision: {train_config.precision}")
    features = model_input_features(
        data,
        train_config.feature_mode,
        train_config.feature_hops,
        train_config.feature_ppr_alpha,
    ).to(dtype=dtype)

    model = build_model(features.shape[1], data.num_classes, model_config).to(
        device=device, dtype=dtype
    )
    feature_init = model_config.feature_init.lower()
    if feature_init == "pca":
        initialize_feature_mapper_pca(model, features)
    elif feature_init == "xavier":
        initialize_feature_mapper_xavier(model)
    elif feature_init != "random":
        raise ValueError(f"Unsupported feature initialization: {model_config.feature_init}")
    # Initialization-scale tuning changes no layer or forward equation.  It is
    # useful for row-normalized WebKB inputs, whose official default Linear
    # initialization otherwise produces nearly indistinguishable SPD nodes.
    with torch.no_grad():
        model.feature_to_spd.dimred.weight.mul_(
            train_config.feature_weight_init_scale
        )
        if model.feature_to_spd.dimred.bias is not None:
            model.feature_to_spd.dimred.bias.mul_(train_config.feature_bias_init_scale)
    if model_config.freeze_feature_mapper:
        for parameter in model.feature_to_spd.parameters():
            parameter.requires_grad_(False)
    ema_model = None
    if train_config.ema_decay > 0:
        ema_model = copy.deepcopy(model)
        ema_model.requires_grad_(False)
    optimizer_name = train_config.optimizer.lower()
    optimizer_class = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "adamax": torch.optim.Adamax,
        "nadam": torch.optim.NAdam,
        "radam": torch.optim.RAdam,
        "sgd": torch.optim.SGD,
    }.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer: {train_config.optimizer}")
    if (
        train_config.feature_lr_ratio == 1.0
        and train_config.bimap_lr_ratio == 1.0
    ):
        optimizer_parameters = model.parameters()
    else:
        feature_parameters = [
            parameter
            for parameter in model.feature_to_spd.parameters()
            if parameter.requires_grad
        ]
        feature_ids = {id(parameter) for parameter in feature_parameters}
        bimap_parameters = [
            parameter
            for bimap in model.bimaps
            for parameter in bimap.parameters()
            if parameter.requires_grad
        ]
        bimap_ids = {id(parameter) for parameter in bimap_parameters}
        other_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
            and id(parameter) not in feature_ids
            and id(parameter) not in bimap_ids
        ]
        optimizer_parameters = [
            {
                "params": feature_parameters,
                "lr": train_config.learning_rate * train_config.feature_lr_ratio,
            },
            {
                "params": bimap_parameters,
                "lr": train_config.learning_rate * train_config.bimap_lr_ratio,
            },
            {"params": other_parameters, "lr": train_config.learning_rate},
        ]
    optimizer_kwargs = {
        "lr": train_config.learning_rate,
        "weight_decay": train_config.weight_decay,
    }
    if optimizer_name == "sgd":
        optimizer_kwargs["momentum"] = train_config.momentum
    optimizer = optimizer_class(optimizer_parameters, **optimizer_kwargs)
    scheduler_name = train_config.scheduler.lower()
    if scheduler_name == "none":
        scheduler = None
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config.epochs,
            eta_min=train_config.learning_rate * train_config.minimum_lr_ratio,
        )
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_config.step_size,
            gamma=train_config.scheduler_gamma,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {train_config.scheduler}")

    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    # Diagnostic only: retain the best task metric observed on validation
    # without changing the released GNRF loss-based checkpoint rule.
    max_validation_accuracy_seen = float("-inf")
    max_validation_accuracy_epoch = -1
    best_epoch = -1
    best_state = None
    stopped_nonfinite = False
    started = time.monotonic()
    class_weights = None
    if train_config.class_weight_power > 0:
        counts = torch.bincount(
            data.y[train_index], minlength=data.num_classes
        ).to(dtype=dtype)
        present = counts > 0
        class_weights = torch.zeros_like(counts)
        mean_count = counts[present].mean()
        class_weights[present] = (mean_count / counts[present]).pow(
            train_config.class_weight_power
        )

    def training_objective(
        logits: torch.Tensor,
        alternate_labels: torch.Tensor | None = None,
        mix_fraction: float = 1.0,
    ) -> torch.Tensor:
        primary = F.cross_entropy(
            logits[train_index],
            data.y[train_index],
            weight=class_weights,
            label_smoothing=train_config.label_smoothing,
        )
        objective = primary
        if alternate_labels is not None:
            alternate = F.cross_entropy(
                logits[train_index],
                alternate_labels,
                weight=class_weights,
                label_smoothing=train_config.label_smoothing,
            )
            objective = mix_fraction * primary + (1.0 - mix_fraction) * alternate
        if train_config.center_loss_weight > 0:
            primary_center = supervised_center_loss(
                logits[train_index], data.y[train_index], data.num_classes
            )
            center = primary_center
            if alternate_labels is not None:
                alternate_center = supervised_center_loss(
                    logits[train_index], alternate_labels, data.num_classes
                )
                center = (
                    mix_fraction * primary_center
                    + (1.0 - mix_fraction) * alternate_center
                )
            objective = objective + train_config.center_loss_weight * center
        if train_config.supervised_contrastive_weight > 0:
            primary_contrastive = supervised_contrastive_loss(
                logits[train_index],
                data.y[train_index],
                train_config.supervised_contrastive_temperature,
            )
            contrastive = primary_contrastive
            if alternate_labels is not None:
                alternate_contrastive = supervised_contrastive_loss(
                    logits[train_index],
                    alternate_labels,
                    train_config.supervised_contrastive_temperature,
                )
                contrastive = (
                    mix_fraction * primary_contrastive
                    + (1.0 - mix_fraction) * alternate_contrastive
                )
            objective = objective + (
                train_config.supervised_contrastive_weight * contrastive
            )
        if train_config.auc_pairwise_weight > 0:
            primary_auc = pairwise_auc_surrogate(
                logits[train_index],
                data.y[train_index],
                train_config.auc_pairwise_pairs,
                train_config.auc_pairwise_margin,
            )
            auc_objective = primary_auc
            if alternate_labels is not None:
                alternate_auc = pairwise_auc_surrogate(
                    logits[train_index],
                    alternate_labels,
                    train_config.auc_pairwise_pairs,
                    train_config.auc_pairwise_margin,
                )
                auc_objective = (
                    mix_fraction * primary_auc
                    + (1.0 - mix_fraction) * alternate_auc
                )
            objective = objective + (
                train_config.auc_pairwise_weight * auc_objective
            )
        return objective

    for epoch in range(train_config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        training_features = features
        alternate_labels = None
        mix_fraction = 1.0
        if train_config.mixup_alpha > 0:
            mix_fraction = float(
                np.random.beta(train_config.mixup_alpha, train_config.mixup_alpha)
            )
            permutation = torch.randperm(train_index.shape[0], device=device)
            alternate_index = train_index.index_select(0, permutation)
            alternate_labels = data.y.index_select(0, alternate_index)
            training_features = features.clone()
            training_features[train_index] = (
                mix_fraction * features[train_index]
                + (1.0 - mix_fraction) * features[alternate_index]
            )
        if train_config.input_noise_std > 0:
            if training_features is features:
                training_features = features.clone()
            training_features.add_(
                train_config.input_noise_std * torch.randn_like(training_features)
            )
        training_edge_index, training_edge_weight = drop_undirected_edges(
            model_edge_index,
            model_edge_weight,
            data.num_nodes,
            train_config.edge_dropout,
        )
        try:
            logits = model(
                training_features, training_edge_index, training_edge_weight
            )
        except torch._C._LinAlgError:
            if best_state is None:
                raise
            stopped_nonfinite = True
            break
        loss = training_objective(logits, alternate_labels, mix_fraction)
        if not torch.isfinite(loss):
            if best_state is None:
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            stopped_nonfinite = True
            break
        loss.backward()
        if train_config.sam_rho > 0:
            parameters_with_grad = [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            gradient_terms = [
                (
                    parameter.abs() if train_config.sam_adaptive else 1.0
                )
                * parameter.grad
                for parameter in parameters_with_grad
            ]
            gradient_norm = torch.linalg.vector_norm(
                torch.stack([term.norm(p=2) for term in gradient_terms]), ord=2
            )
            scale = train_config.sam_rho / (gradient_norm + 1.0e-12)
            perturbations = []
            with torch.no_grad():
                for parameter in parameters_with_grad:
                    multiplier = (
                        parameter.square() if train_config.sam_adaptive else 1.0
                    )
                    perturbation = multiplier * parameter.grad * scale
                    parameter.add_(perturbation)
                    perturbations.append((parameter, perturbation))
            optimizer.zero_grad(set_to_none=True)
            perturbed_logits = model(
                training_features, training_edge_index, training_edge_weight
            )
            perturbed_loss = training_objective(
                perturbed_logits, alternate_labels, mix_fraction
            )
            if not torch.isfinite(perturbed_loss):
                with torch.no_grad():
                    for parameter, perturbation in perturbations:
                        parameter.sub_(perturbation)
                if best_state is None:
                    raise FloatingPointError(
                        f"non-finite SAM loss at epoch {epoch}"
                    )
                stopped_nonfinite = True
                break
            perturbed_loss.backward()
            with torch.no_grad():
                for parameter, perturbation in perturbations:
                    parameter.sub_(perturbation)
        if train_config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=train_config.gradient_clip
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if ema_model is not None and epoch >= train_config.ema_start_epoch:
            with torch.no_grad():
                if epoch == train_config.ema_start_epoch:
                    ema_model.load_state_dict(model.state_dict())
                else:
                    live_parameters = dict(model.named_parameters())
                    for name, parameter in ema_model.named_parameters():
                        parameter.mul_(train_config.ema_decay).add_(
                            live_parameters[name], alpha=1.0 - train_config.ema_decay
                        )
                    live_buffers = dict(model.named_buffers())
                    for name, buffer in ema_model.named_buffers():
                        buffer.copy_(live_buffers[name])

        evaluation_model = (
            ema_model
            if ema_model is not None and epoch >= train_config.ema_start_epoch
            else model
        )
        evaluation_model.eval()
        with torch.no_grad():
            try:
                logits = averaged_eval_logits(
                    evaluation_model,
                    features,
                    model_edge_index,
                    model_edge_weight,
                    train_config.eval_samples,
                    train_config.eval_aggregation,
                )
            except torch._C._LinAlgError:
                if best_state is None:
                    raise
                stopped_nonfinite = True
                break
            validation_loss = F.cross_entropy(
                logits[validation_index], data.y[validation_index]
            ).item()
            validation_accuracy = classification_score(
                data.name, logits[validation_index], data.y[validation_index]
            )
        if validation_accuracy > max_validation_accuracy_seen:
            max_validation_accuracy_seen = validation_accuracy
            max_validation_accuracy_epoch = epoch
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(evaluation_model.state_dict())

    if best_state is None:
        raise FloatingPointError(
            "no finite validation checkpoint was produced; use a numerically "
            "stable feature scale, SPD jitter, or optimizer setting"
        )
    if train_config.head_finetune_epochs > 0:
        model.load_state_dict(best_state)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.output_head.parameters():
            parameter.requires_grad_(True)
        finetune_optimizer = torch.optim.Adam(
            model.output_head.parameters(),
            lr=train_config.head_finetune_lr,
            weight_decay=train_config.head_finetune_weight_decay,
        )
        for finetune_epoch in range(train_config.head_finetune_epochs):
            model.train()
            finetune_optimizer.zero_grad(set_to_none=True)
            logits = model(features, model_edge_index, model_edge_weight)
            loss = F.cross_entropy(
                logits[train_index],
                data.y[train_index],
                weight=class_weights,
                label_smoothing=train_config.label_smoothing,
            )
            if train_config.center_loss_weight > 0:
                loss = loss + train_config.center_loss_weight * supervised_center_loss(
                    logits[train_index], data.y[train_index], data.num_classes
                )
            if train_config.supervised_contrastive_weight > 0:
                loss = loss + (
                    train_config.supervised_contrastive_weight
                    * supervised_contrastive_loss(
                        logits[train_index],
                        data.y[train_index],
                        train_config.supervised_contrastive_temperature,
                    )
                )
            if not torch.isfinite(loss):
                stopped_nonfinite = True
                break
            loss.backward()
            if train_config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.output_head.parameters(), max_norm=train_config.gradient_clip
                )
            finetune_optimizer.step()
            model.eval()
            with torch.no_grad():
                logits = averaged_eval_logits(
                    model,
                    features,
                    model_edge_index,
                    model_edge_weight,
                    train_config.eval_samples,
                    train_config.eval_aggregation,
                )
                validation_loss = F.cross_entropy(
                    logits[validation_index], data.y[validation_index]
                ).item()
                validation_accuracy = classification_score(
                    data.name, logits[validation_index], data.y[validation_index]
                )
            if validation_accuracy > max_validation_accuracy_seen:
                max_validation_accuracy_seen = validation_accuracy
                max_validation_accuracy_epoch = train_config.epochs + finetune_epoch
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_validation_accuracy = validation_accuracy
                best_epoch = train_config.epochs + finetune_epoch
                best_state = copy.deepcopy(model.state_dict())

    result = {
        "seed": seed,
        "metric": metric_name(data.name),
        "best_epoch": best_epoch,
        "validation_loss": best_validation_loss,
        "validation_accuracy": best_validation_accuracy,
        "max_validation_accuracy_seen": max_validation_accuracy_seen,
        "max_validation_accuracy_epoch": max_validation_accuracy_epoch,
        "seconds": time.monotonic() - started,
        "stopped_nonfinite": stopped_nonfinite,
        "initialization_seed": initialization_seed,
    }
    if include_test or return_logits:
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits = averaged_eval_logits(
                model,
                features,
                model_edge_index,
                model_edge_weight,
                train_config.eval_samples,
                train_config.eval_aggregation,
            )
        if return_logits:
            result["_validation_logits"] = logits[validation_index].detach().cpu()
            if include_test:
                result["_test_logits"] = logits[test_index].detach().cpu()
    if include_test:
        with torch.no_grad():
            result["test_accuracy"] = classification_score(
                data.name, logits[test_index], data.y[test_index]
            )
    return result


def parse_dataclass_config(cls, values: dict):
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def run(args: argparse.Namespace) -> dict:
    with Path(args.config).open() as handle:
        raw_config = json.load(handle)
    model_config = parse_dataclass_config(ModelConfig, raw_config.get("model", {}))
    train_config = parse_dataclass_config(TrainConfig, raw_config.get("training", {}))
    if args.feature_mapper_override is not None:
        model_config = replace(
            model_config, feature_mapper=args.feature_mapper_override
        )
    if args.precision_override is not None:
        train_config = replace(train_config, precision=args.precision_override)
    if args.operator_override is not None:
        model_config = replace(model_config, operator=args.operator_override)
    if model_config.layers != 1 and not args.allow_deeper:
        raise ValueError("Initial experiments must have exactly one layer")

    device = torch.device(args.device)
    data = load_webkb(args.dataset, Path(args.data_root)).to(device)
    input_features = model_input_features(
        data,
        train_config.feature_mode,
        train_config.feature_hops,
        train_config.feature_ppr_alpha,
    )
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    include_test = args.phase == "final"
    results = []
    for seed in seeds:
        result = train_seed(
            data, seed, model_config, train_config, device, include_test=include_test
        )
        results.append(result)
        fields_to_log = (
            f"val={result['validation_accuracy']:.2f}%"
            + (f" test={result['test_accuracy']:.2f}%" if include_test else "")
        )
        print(
            f"[{args.dataset}] seed={seed} epoch={result['best_epoch']} {fields_to_log}",
            flush=True,
        )

    validation_values = np.asarray([item["validation_accuracy"] for item in results])
    output = {
        "dataset": args.dataset,
        "metric": metric_name(data.name),
        "phase": args.phase,
        "protocol": "GNRF_new exact released 60/20/20 split implementation",
        "seeds": seeds,
        "model": asdict(model_config),
        "training": asdict(train_config),
        "parameter_count": sum(
            parameter.numel()
            for parameter in build_model(
                input_features.shape[1], data.num_classes, model_config
            ).parameters()
        ),
        "runs": results,
        "validation_mean": float(validation_values.mean()),
        "validation_std": paper_standard_deviation(validation_values),
    }
    if include_test:
        test_values = np.asarray([item["test_accuracy"] for item in results])
        output.update(
            {
                "test_mean": float(test_values.mean()),
                "test_std": paper_standard_deviation(test_values),
                "paper_target": PAPER_TARGETS[args.dataset],
                "margin": float(test_values.mean() - PAPER_TARGETS[args.dataset]),
            }
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    temporary_path.replace(output_path)
    print(json.dumps({key: output[key] for key in output if key.endswith("mean") or key == "margin"}))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(PAPER_TARGETS), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("validation", "final"), default="validation")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--allow-deeper", action="store_true")
    parser.add_argument(
        "--feature-mapper-override",
        choices=("squared", "triangular", "gram", "rank_one"),
        default=None,
        help="Override only FeatureToSPD while retaining the supplied config.",
    )
    parser.add_argument(
        "--precision-override",
        choices=("float32", "float64"),
        default=None,
        help="Override only arithmetic precision while retaining the supplied config.",
    )
    parser.add_argument(
        "--operator-override",
        choices=("simple1", "spd4gnn_gcn"),
        default=None,
        help="Replace only the propagation operator while retaining shared parameters.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
