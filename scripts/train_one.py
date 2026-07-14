#!/usr/bin/env python3
"""Train a single dataset × backbone combination."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import build_dataloaders
from src.eval.metrics import evaluate_and_export
from src.models.backbones import create_model
from src.train.seed import set_seed
from src.train.trainer import two_phase_train
from src.utils import get_device, load_config, run_name


def parse_args():
    p = argparse.ArgumentParser(description="SeedLeaf single-run trainer")
    p.add_argument(
        "--dataset",
        required=True,
        choices=["plantvillage", "tomato", "cassava"],
    )
    p.add_argument(
        "--backbone",
        required=True,
        choices=["efficientnet_b3", "resnet50"],
    )
    p.add_argument("--config", default=None, help="Path to YAML config")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(prefer_cuda=not args.cpu)
    print(f"Device: {device}")
    if device.type == "cuda":
        import torch

        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader, classes = build_dataloaders(
        args.dataset, cfg
    )
    print(f"Classes ({len(classes)}): {classes[:8]}{'...' if len(classes) > 8 else ''}")

    model = create_model(
        args.backbone,
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
        dataset=args.dataset,
        backbone=args.backbone,
        classes=classes,
    )

    # Reload best checkpoint weights already in model via trainer
    evaluate_and_export(
        model,
        test_loader,
        classes,
        device,
        result["results_dir"],
        history=result["history"],
        use_amp=cfg["train"]["amp"],
        title_prefix=run_name(args.dataset, args.backbone),
    )
    print(f"Done. Checkpoint: {result['checkpoint']}")


if __name__ == "__main__":
    main()
