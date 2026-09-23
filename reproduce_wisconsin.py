#!/usr/bin/env python3
"""Reproduce the clean Wisconsin original-graph SPD Ricci Flow result.

This is a narrow GitHub-facing entry point for the Wisconsin row:

    89.80 +/- 3.52 accuracy over GNRF split seeds 0,...,9.

It intentionally rejects configs that use graph rewiring, edge weights,
multi-hop feature propagation, or training-time augmentation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    ModelConfig,
    TrainConfig,
    parse_dataclass_config,
    run,
)


DEFAULT_CONFIGS = {
    "tgrelu": ROOT / "configs" / "wisconsin_tgrelu.json",
    "reeig": ROOT / "configs" / "wisconsin_reeig.json",
}
DEFAULT_OUTPUTS = {
    name: ROOT / "results" / f"wisconsin_{name}_reproduce.json"
    for name in DEFAULT_CONFIGS
}
EXPECTED_RESULTS = {
    # Table-facing values use the population standard deviation over seeds.
    "tgrelu": (89.79999780654907, 3.5156792042266205),
    "reeig": (87.99999713897705, 2.366432175190927),
}


def load_and_validate_config(path: Path) -> tuple[ModelConfig, TrainConfig]:
    with path.open() as handle:
        raw = json.load(handle)
    model = parse_dataclass_config(ModelConfig, raw.get("model", {}))
    training = parse_dataclass_config(TrainConfig, raw.get("training", {}))

    clean_checks = {
        "operator=simple1": model.operator == "simple1",
        "feature_mapper=squared": model.feature_mapper == "squared",
        "feature_init=random": model.feature_init == "random",
        "feature_mode=original": training.feature_mode == "original",
        "graph_mode=original": training.graph_mode == "original",
        "graph_topk=0": training.graph_topk == 0,
        "graph_weight_mode=none": training.graph_weight_mode == "none",
        "edge_dropout=0": training.edge_dropout == 0.0,
        "input_noise_std=0": training.input_noise_std == 0.0,
        "mixup_alpha=0": training.mixup_alpha == 0.0,
        "label_smoothing=0": training.label_smoothing == 0.0,
        "class_weight_power=0": training.class_weight_power == 0.0,
        "auc_pairwise_weight=0": training.auc_pairwise_weight == 0.0,
        "center_loss_weight=0": training.center_loss_weight == 0.0,
        "supervised_contrastive_weight=0": training.supervised_contrastive_weight == 0.0,
        "ema_decay=0": training.ema_decay == 0.0,
        "sam_rho=0": training.sam_rho == 0.0,
        "head_finetune_epochs=0": training.head_finetune_epochs == 0,
        "eval_samples=1": training.eval_samples == 1,
    }
    failed = [name for name, ok in clean_checks.items() if not ok]
    if failed:
        raise ValueError(
            "Config is not the clean Wisconsin original-graph protocol; "
            f"failed checks: {', '.join(failed)}"
        )
    return model, training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the clean Wisconsin SPD Ricci Flow result."
    )
    parser.add_argument(
        "--activation",
        choices=tuple(DEFAULT_CONFIGS),
        default="tgrelu",
        help=(
            "Activation preset. TGReLU is the validation-selected Wisconsin "
            "default; reeig runs the classic hard eigenvalue floor ablation."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config override; otherwise --activation selects the config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path; defaults to an activation-specific result file.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate that the config is the clean original-graph protocol and exit.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.25,
        help="Allowed absolute deviation from the recorded mean accuracy.",
    )
    return parser


def summarize_table_statistics(result: dict) -> None:
    """Add the table-facing mean/std values used by the Wisconsin row."""
    test_values = [
        float(run["test_accuracy"])
        for run in result.get("runs", [])
        if "test_accuracy" in run
    ]
    if not test_values:
        return
    mean = sum(test_values) / len(test_values)
    population_variance = sum((value - mean) ** 2 for value in test_values) / len(
        test_values
    )
    population_std = math.sqrt(population_variance)
    sample_std = (
        math.sqrt(
            sum((value - mean) ** 2 for value in test_values)
            / (len(test_values) - 1)
        )
        if len(test_values) > 1
        else 0.0
    )
    result["reported_test_mean"] = mean
    result["reported_test_std"] = population_std
    result["test_std_population"] = population_std
    result["test_std_sample"] = sample_std


def main() -> None:
    args = build_parser().parse_args()
    config_path = (args.config or DEFAULT_CONFIGS[args.activation]).resolve()
    model, _ = load_and_validate_config(config_path)
    preset = (
        "tgrelu"
        if model.activation == "tgrelu"
        else "reeig" if model.activation in {"mild_reeig", "reeig", "mild"} else None
    )
    output_path = (
        args.output or DEFAULT_OUTPUTS[preset or args.activation]
    ).resolve()

    print(f"Clean Wisconsin config: {config_path}")
    print(f"SPD activation: {model.activation}")
    print("Protocol checks passed: original features, original graph, no edge weights.")
    if args.check_only:
        return

    result = run(
        SimpleNamespace(
            dataset="wisconsin",
            config=str(config_path),
            output=str(output_path),
            phase="final",
            seeds=args.seeds,
            device=args.device,
            data_root=str(args.data_root),
            allow_deeper=True,
            feature_mapper_override=None,
            precision_override=None,
            operator_override=None,
        )
    )
    summarize_table_statistics(result)
    with output_path.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    mean = result.get("test_mean")
    std = result.get("reported_test_std", result.get("test_std"))
    print(f"Saved result: {output_path}")
    print(f"Wisconsin table accuracy: {mean:.2f} +/- {std:.2f}")
    if preset is None:
        print("No recorded reference is available for this custom activation.")
        return
    expected_mean, expected_std = EXPECTED_RESULTS[preset]
    if abs(mean - expected_mean) > args.tolerance:
        raise RuntimeError(
            f"Mean accuracy {mean:.4f} differs from recorded {expected_mean:.4f} "
            f"by more than tolerance {args.tolerance}."
        )
    if abs(std - expected_std) > args.tolerance:
        raise RuntimeError(
            f"Reported std {std:.4f} differs from recorded {expected_std:.4f} "
            f"by more than tolerance {args.tolerance}."
        )


if __name__ == "__main__":
    main()
