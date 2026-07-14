"""Inference helpers for the Streamlit app."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from src.models.backbones import load_checkpoint
from src.utils import ROOT, load_json, run_name


def list_available_runs() -> list[dict[str, Any]]:
    """Return runs that have a checkpoint and/or results."""
    ckpt_dir = ROOT / "checkpoints"
    results_dir = ROOT / "results"
    runs = []
    seen = set()

    if results_dir.exists():
        for d in sorted(results_dir.iterdir()):
            if d.is_dir() and "__" in d.name:
                dataset, _, backbone = d.name.partition("__")
                ckpt = ckpt_dir / f"{d.name}_best.pt"
                meta_path = d / "meta.json"
                classes = []
                if meta_path.exists():
                    meta = load_json(meta_path)
                    classes = meta.get("classes", [])
                runs.append(
                    {
                        "run": d.name,
                        "dataset": dataset,
                        "backbone": backbone,
                        "has_checkpoint": ckpt.exists(),
                        "checkpoint": str(ckpt) if ckpt.exists() else None,
                        "results_dir": str(d),
                        "classes": classes,
                        "demo": meta.get("demo", False) if meta_path.exists() else False,
                    }
                )
                seen.add(d.name)

    if ckpt_dir.exists():
        for ckpt in sorted(ckpt_dir.glob("*_best.pt")):
            name = ckpt.name.replace("_best.pt", "")
            if name in seen:
                continue
            dataset, _, backbone = name.partition("__")
            runs.append(
                {
                    "run": name,
                    "dataset": dataset,
                    "backbone": backbone,
                    "has_checkpoint": True,
                    "checkpoint": str(ckpt),
                    "results_dir": str(results_dir / name),
                    "classes": [],
                    "demo": False,
                }
            )
    return runs


def load_class_names(dataset: str, backbone: str) -> list[str]:
    meta_path = ROOT / "results" / run_name(dataset, backbone) / "meta.json"
    if meta_path.exists():
        classes = load_json(meta_path).get("classes", [])
        if classes and not (
            len(classes) > 0 and str(classes[0]).startswith("class_")
        ):
            return classes

    # Prefer processed data classes.json
    classes_path = ROOT / "data" / dataset / "classes.json"
    if classes_path.exists():
        with open(classes_path, encoding="utf-8") as f:
            return json.load(f)

    # ImageFolder layout
    train_dir = ROOT / "data" / dataset / "train"
    if train_dir.exists():
        return sorted([p.name for p in train_dir.iterdir() if p.is_dir()])

    return []


@lru_cache(maxsize=4)
def get_model(dataset: str, backbone: str, num_classes: int):
    ckpt = ROOT / "checkpoints" / f"{run_name(dataset, backbone)}_best.pt"
    if not ckpt.exists():
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(str(ckpt), num_classes, backbone, device)
    return model, device


def clear_model_cache() -> None:
    get_model.cache_clear()


def format_disease_name(name: str) -> str:
    return name.replace("___", " — ").replace("_", " ").strip()
