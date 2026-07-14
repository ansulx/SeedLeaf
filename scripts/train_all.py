#!/usr/bin/env python3
"""Train all dataset × backbone combinations (3 × 2)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import build_dataloaders
from src.eval.metrics import aggregate_leaderboard, evaluate_and_export
from src.models.backbones import create_model
from src.train.seed import set_seed
from src.train.trainer import two_phase_train
from src.utils import get_device, load_config, run_name, save_json


DATASETS = ["plantvillage", "tomato", "cassava"]
BACKBONES = ["efficientnet_b3", "resnet50"]


def parse_args():
    p = argparse.ArgumentParser(description="SeedLeaf full experiment grid")
    p.add_argument("--config", default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=DATASETS,
    )
    p.add_argument(
        "--backbones",
        nargs="+",
        default=BACKBONES,
        choices=BACKBONES,
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have summary.json",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(prefer_cuda=not args.cpu)
    print(f"Device: {device}")

    results_root = Path(cfg["paths"]["results"])
    for dataset in args.datasets:
        for backbone in args.backbones:
            name = run_name(dataset, backbone)
            summary_path = results_root / name / "summary.json"
            if args.skip_existing and summary_path.exists():
                from src.utils import load_json

                summary = load_json(summary_path)
                if summary.get("demo"):
                    print(f"Replacing demo results for: {name}")
                else:
                    print(f"Skipping existing run: {name}")
                    continue

            print("\n" + "=" * 60)
            print(f"Training {name}")
            print("=" * 60)
            set_seed(cfg["seed"])

            train_loader, val_loader, test_loader, classes = build_dataloaders(
                dataset, cfg
            )
            model = create_model(
                backbone,
                num_classes=len(classes),
                pretrained=cfg["models"]["pretrained"],
                dropout=cfg["models"]["dropout"],
            )
            result = two_phase_train(
                model,
                train_loader,
                val_loader,
                device,
                cfg,
                dataset=dataset,
                backbone=backbone,
                classes=classes,
            )
            evaluate_and_export(
                model,
                test_loader,
                classes,
                device,
                result["results_dir"],
                history=result["history"],
                use_amp=cfg["train"]["amp"],
                title_prefix=name,
            )

    board = aggregate_leaderboard(results_root)
    save_json(board, results_root / "leaderboard.json")
    print("\nLeaderboard:")
    for row in board:
        print(
            f"  {row['dataset']:15s} {row['backbone']:16s} "
            f"acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f}"
        )


if __name__ == "__main__":
    main()
