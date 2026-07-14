#!/usr/bin/env python3
"""Evaluate a checkpoint on the test split and refresh result artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import build_dataloaders
from src.eval.metrics import aggregate_leaderboard, evaluate_and_export
from src.models.backbones import load_checkpoint
from src.train.seed import set_seed
from src.utils import get_device, load_config, load_json, run_name, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["plantvillage", "tomato", "cassava"])
    p.add_argument("--backbone", required=True, choices=["efficientnet_b3", "resnet50"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(prefer_cuda=not args.cpu)

    name = run_name(args.dataset, args.backbone)
    results_dir = Path(cfg["paths"]["results"]) / name
    ckpt = args.checkpoint or Path(cfg["paths"]["checkpoints"]) / f"{name}_best.pt"

    _, _, test_loader, classes = build_dataloaders(args.dataset, cfg)
    model = load_checkpoint(str(ckpt), len(classes), args.backbone, device)

    history = None
    hist_path = results_dir / "history.json"
    if hist_path.exists():
        history = load_json(hist_path)

    evaluate_and_export(
        model,
        test_loader,
        classes,
        device,
        results_dir,
        history=history,
        use_amp=cfg["train"]["amp"],
        title_prefix=name,
    )

    board = aggregate_leaderboard(cfg["paths"]["results"])
    save_json(board, Path(cfg["paths"]["results"]) / "leaderboard.json")


if __name__ == "__main__":
    main()
