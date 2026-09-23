"""Wisconsin WebKB loading and node splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.datasets import WebKB
from torch_geometric.utils import remove_self_loops, to_undirected


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"


@dataclass
class NodeData:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    name: str

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item() + 1)

    def to(self, device: torch.device) -> "NodeData":
        return NodeData(
            self.x.to(device),
            self.edge_index.to(device),
            self.y.to(device),
            self.name,
        )


def load_node_dataset(
    name: str = "wisconsin", root: Path = DEFAULT_DATA_ROOT
) -> NodeData:
    if name.lower() != "wisconsin":
        raise ValueError("This repository provides the Wisconsin experiment")
    graph = WebKB(root=str(root), name="wisconsin")[0]
    edge_index = remove_self_loops(graph.edge_index)[0]
    edge_index = to_undirected(edge_index)

    features = graph.x.clone()
    features[features.isnan()] = 0.0
    row_sum = features.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0.0] = 1.0
    features = features / row_sum
    return NodeData(features, edge_index, graph.y.flatten(), "wisconsin")


def random_node_split(num_nodes: int, seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_size = int(num_nodes * 0.6)
    validation_size = int(num_nodes * 0.2)
    indices = torch.randperm(num_nodes)
    train = indices[:train_size]
    validation = indices[train_size + 1 : train_size + validation_size]
    test = indices[train_size + validation_size + 1 :]
    return train, validation, test
