#!/usr/bin/env python3
"""Train and evaluate SPD Ricci Flow on a node-classification dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import DEFAULT_DATA_ROOT, run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SPD Ricci Flow.")
    parser.add_argument("--dataset", default="wisconsin")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "wisconsin.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "wisconsin_run.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--phase",
        choices=("validation", "final"),
        default="final",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        SimpleNamespace(
            dataset=args.dataset,
            config=str(args.config.resolve()),
            output=str(args.output.resolve()),
            phase=args.phase,
            seeds=args.seeds,
            device=args.device,
            data_root=str(args.data_root.resolve()),
            allow_deeper=True,
            feature_mapper_override=None,
            precision_override=None,
            operator_override=None,
        )
    )


if __name__ == "__main__":
    main()
