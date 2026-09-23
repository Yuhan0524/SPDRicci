"""GNRF benchmark loading and the exact released split protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.datasets import (
    CitationFull,
    HeterophilousGraphDataset,
    Planetoid,
    WebKB,
)
from torch_geometric.utils import remove_self_loops, to_undirected


# PyTorch Geometric downloads Wisconsin into this repository-local directory.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"


@dataclass
class NodeData:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    name: str = ""

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
            self.x.to(device), self.edge_index.to(device), self.y.to(device), self.name
        )


DATASET_ALIASES = {
    "cornell": "cornell",
    "wisconsin": "wisconsin",
    "texas": "texas",
    "roman-empire": "roman_empire",
    "roman_empire": "roman_empire",
    "tolokers": "tolokers",
    "minesweeper": "minesweeper",
    "questions": "questions",
    "amazon-ratings": "amazon_ratings",
    "amazon_ratings": "amazon_ratings",
    "cora-full": "cora_full",
    "cora_full": "cora_full",
    "pubmed": "pubmed",
    "dblp": "dblp",
    "cora-ml": "cora_ml",
    "cora_ml": "cora_ml",
}


def canonical_dataset_name(name: str) -> str:
    normalized = name.lower().replace(" ", "-")
    if normalized not in DATASET_ALIASES:
        raise ValueError(f"Unsupported GNRF benchmark dataset: {name}")
    return DATASET_ALIASES[normalized]


def load_node_dataset(name: str, root: Path = DEFAULT_DATA_ROOT) -> NodeData:
    """Load every node dataset reported in the GNRF main table.

    Dataset classes and preprocessing mirror the released GNRF loader:
    WebKB for the three small web graphs, the Heterophilous Graph Benchmark
    class for its five datasets, Planetoid for PubMed, and CitationFull for
    Cora Full, DBLP, and Cora ML.  Every graph is made undirected, self-loops
    are removed, and supplied features are row-sum normalized.
    """
    canonical = canonical_dataset_name(name)
    if canonical in {"cornell", "wisconsin", "texas"}:
        dataset = WebKB(root=str(root), name=canonical)
    elif canonical in {
        "roman_empire",
        "tolokers",
        "minesweeper",
        "questions",
        "amazon_ratings",
    }:
        dataset = HeterophilousGraphDataset(
            root=str(root), name=canonical.replace("_", "-")
        )
    elif canonical == "pubmed":
        dataset = Planetoid(root=str(root), name="PubMed")
    else:
        citation_name = {
            "cora_full": "Cora",
            "dblp": "DBLP",
            "cora_ml": "Cora_ML",
        }[canonical]
        dataset = CitationFull(root=str(root), name=citation_name)
    graph = dataset[0]
    edge_index = remove_self_loops(graph.edge_index)[0]
    edge_index = to_undirected(edge_index)
    features = graph.x.clone()
    features[features.isnan()] = 0.0
    row_sum = features.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0.0] = 1.0
    features = features / row_sum
    return NodeData(features, edge_index, graph.y.flatten(), canonical)


def load_webkb(name: str, root: Path = DEFAULT_DATA_ROOT) -> NodeData:
    """Backward-compatible entry point retained for existing tuning scripts."""
    return load_node_dataset(name, root)


def gnrf_reference_split(num_nodes: int, seed: int) -> tuple[torch.Tensor, ...]:
    """Match the released GNRF code exactly for direct report comparison."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    n_train = int(num_nodes * 0.6)
    n_val = int(num_nodes * 0.2)
    full_index = torch.randperm(num_nodes)
    # The +1 offsets intentionally mirror GNRF_new/NodeDataset.py.
    train = full_index[:n_train]
    validation = full_index[n_train + 1 : n_train + n_val]
    test = full_index[n_train + n_val + 1 :]
    return train, validation, test
