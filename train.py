"""Full-batch training with validation-loss checkpoint selection."""

from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import DEFAULT_DATA_ROOT, load_node_dataset, random_node_split
from model import ModelConfig, build_model


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 0.03
    weight_decay: float = 5.0e-4
    epochs: int = 300
    optimizer: str = "adam"
    gradient_clip: float = 5.0
    precision: str = "float32"


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == labels).float().mean().item() * 100.0)


def population_standard_deviation(values: list[float]) -> float:
    return float(np.asarray(values).std(ddof=0)) if values else 0.0


def evaluation_logits(
    model, features: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    # Retain the reduction used by the reported Wisconsin checkpoint rule.
    return torch.stack((model(features, edge_index),)).mean(dim=0)


def train_seed(
    data,
    seed: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
    include_test: bool,
) -> dict:
    set_seed(seed)
    train_index, validation_index, test_index = random_node_split(
        data.num_nodes, seed
    )
    train_index = train_index.to(device)
    validation_index = validation_index.to(device)
    test_index = test_index.to(device)

    dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }.get(train_config.precision.lower())
    if dtype is None:
        raise ValueError(f"Unsupported precision: {train_config.precision}")

    features = data.x.to(dtype=dtype)
    model = build_model(
        features.shape[1], data.num_classes, model_config
    ).to(device=device, dtype=dtype)
    if train_config.optimizer.lower() != "adam":
        raise ValueError("This experiment uses the Adam optimizer")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_epoch = -1
    best_state = None
    stopped_nonfinite = False
    started = time.monotonic()

    for epoch in range(train_config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        try:
            logits = model(features, data.edge_index)
        except torch._C._LinAlgError:
            if best_state is None:
                raise
            stopped_nonfinite = True
            break

        loss = F.cross_entropy(
            logits[train_index],
            data.y[train_index],
            weight=None,
            label_smoothing=0.0,
        )
        if not torch.isfinite(loss):
            if best_state is None:
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            stopped_nonfinite = True
            break
        loss.backward()
        if train_config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=train_config.gradient_clip
            )
        optimizer.step()

        model.eval()
        with torch.no_grad():
            try:
                logits = evaluation_logits(model, features, data.edge_index)
            except torch._C._LinAlgError:
                if best_state is None:
                    raise
                stopped_nonfinite = True
                break
            validation_loss = F.cross_entropy(
                logits[validation_index], data.y[validation_index]
            ).item()
            validation_accuracy = accuracy(
                logits[validation_index], data.y[validation_index]
            )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise FloatingPointError("no finite validation checkpoint was produced")

    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_loss": best_validation_loss,
        "validation_accuracy": best_validation_accuracy,
        "seconds": time.monotonic() - started,
        "stopped_nonfinite": stopped_nonfinite,
    }
    if include_test:
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits = evaluation_logits(model, features, data.edge_index)
        result["test_accuracy"] = accuracy(
            logits[test_index], data.y[test_index]
        )
    return result


def parse_config(cls, values: dict):
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def run(args) -> dict:
    with Path(args.config).open() as handle:
        raw_config = json.load(handle)
    model_config = parse_config(ModelConfig, raw_config.get("model", {}))
    train_config = parse_config(TrainConfig, raw_config.get("training", {}))

    device = torch.device(args.device)
    data = load_node_dataset(args.dataset, Path(args.data_root)).to(device)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    include_test = args.phase == "final"

    runs = []
    for seed in seeds:
        result = train_seed(
            data,
            seed,
            model_config,
            train_config,
            device,
            include_test,
        )
        runs.append(result)
        test_text = (
            f" test={result['test_accuracy']:.2f}%" if include_test else ""
        )
        print(
            f"[{args.dataset}] seed={seed} epoch={result['best_epoch']} "
            f"val={result['validation_accuracy']:.2f}%{test_text}",
            flush=True,
        )

    validation_values = [run["validation_accuracy"] for run in runs]
    output = {
        "dataset": args.dataset,
        "metric": "accuracy",
        "phase": args.phase,
        "protocol": "random 60/20/20 node splits",
        "seeds": seeds,
        "model": asdict(model_config),
        "training": asdict(train_config),
        "parameter_count": sum(
            parameter.numel()
            for parameter in build_model(
                data.num_features, data.num_classes, model_config
            ).parameters()
        ),
        "runs": runs,
        "validation_mean": float(np.mean(validation_values)),
        "validation_std": population_standard_deviation(validation_values),
    }
    if include_test:
        test_values = [run["test_accuracy"] for run in runs]
        output.update(
            {
                "test_mean": float(np.mean(test_values)),
                "test_std": population_standard_deviation(test_values),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    temporary_path.replace(output_path)
    print(
        json.dumps(
            {
                key: output[key]
                for key in output
                if key.endswith("mean") or key.endswith("std")
            }
        )
    )
    return output
