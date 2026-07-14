#!/usr/bin/env python3
"""Prepare datasets (download / organize / split) without training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import prepare_dataset
from src.train.seed import set_seed
from src.utils import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["plantvillage", "tomato", "cassava"],
        choices=["plantvillage", "tomato", "cassava"],
    )
    p.add_argument("--config", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    for name in args.datasets:
        path = prepare_dataset(name, cfg)
        print(f"Ready: {path}")


if __name__ == "__main__":
    main()
