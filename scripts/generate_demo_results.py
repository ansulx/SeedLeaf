#!/usr/bin/env python3
"""Generate demo/placeholder metrics so the Streamlit dashboard works before training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, save_json

# Illustrative placeholders — replaced when you run real training
DEMO = {
    ("plantvillage", "efficientnet_b3"): {
        "accuracy": 0.9912,
        "f1_macro": 0.9898,
        "precision_macro": 0.9901,
        "recall_macro": 0.9895,
        "num_classes": 38,
        "num_test_samples": 8142,
    },
    ("plantvillage", "resnet50"): {
        "accuracy": 0.9864,
        "f1_macro": 0.9841,
        "precision_macro": 0.9848,
        "recall_macro": 0.9835,
        "num_classes": 38,
        "num_test_samples": 8142,
    },
    ("tomato", "efficientnet_b3"): {
        "accuracy": 0.9945,
        "f1_macro": 0.9938,
        "precision_macro": 0.9940,
        "recall_macro": 0.9936,
        "num_classes": 10,
        "num_test_samples": 2715,
    },
    ("tomato", "resnet50"): {
        "accuracy": 0.9910,
        "f1_macro": 0.9901,
        "precision_macro": 0.9904,
        "recall_macro": 0.9898,
        "num_classes": 10,
        "num_test_samples": 2715,
    },
    ("cassava", "efficientnet_b3"): {
        "accuracy": 0.8820,
        "f1_macro": 0.8512,
        "precision_macro": 0.8605,
        "recall_macro": 0.8440,
        "num_classes": 5,
        "num_test_samples": 3209,
    },
    ("cassava", "resnet50"): {
        "accuracy": 0.8645,
        "f1_macro": 0.8310,
        "precision_macro": 0.8398,
        "recall_macro": 0.8245,
        "num_classes": 5,
        "num_test_samples": 3209,
    },
}


def fake_history(seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    history = []
    # phase 1
    for e in range(1, 9):
        history.append(
            {
                "phase": "phase1_frozen",
                "epoch": e,
                "train_loss": float(1.8 / e + rng.normal(0, 0.02)),
                "train_acc": float(min(0.95, 0.55 + 0.05 * e + rng.normal(0, 0.01))),
                "val_loss": float(1.9 / e + rng.normal(0, 0.03)),
                "val_acc": float(min(0.93, 0.52 + 0.048 * e + rng.normal(0, 0.01))),
                "lr": 1e-3,
                "seconds": 40.0,
            }
        )
    # phase 2
    for e in range(1, 21):
        history.append(
            {
                "phase": "phase2_finetune",
                "epoch": e,
                "train_loss": float(0.35 / (1 + 0.15 * e) + rng.normal(0, 0.01)),
                "train_acc": float(min(0.995, 0.90 + 0.004 * e + rng.normal(0, 0.005))),
                "val_loss": float(0.40 / (1 + 0.12 * e) + rng.normal(0, 0.015)),
                "val_acc": float(min(0.99, 0.88 + 0.0045 * e + rng.normal(0, 0.005))),
                "lr": 1e-4,
                "seconds": 55.0,
            }
        )
    return history


def write_curves(history: list[dict], out: Path) -> None:
    epochs = list(range(1, len(history) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], color="#1b5e20", label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], color="#66bb6a", label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [h["train_acc"] for h in history], color="#1b5e20", label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], color="#66bb6a", label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    phases = [h["phase"] for h in history]
    if "phase2_finetune" in phases:
        boundary = phases.index("phase2_finetune") + 0.5
        for ax in axes:
            ax.axvline(boundary, color="#2e7d32", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()


def write_cm(n: int, out: Path, title: str) -> list[list[int]]:
    rng = np.random.default_rng(42 + n)
    cm = np.eye(n, dtype=int) * 80
    cm += rng.integers(0, 5, size=(n, n))
    np.fill_diagonal(cm, np.diag(cm) + rng.integers(10, 40, size=n))
    plt.figure(figsize=(max(6, n * 0.4), max(5, n * 0.35)))
    sns.heatmap(cm, cmap="Greens", annot=n <= 12, fmt="d")
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    return cm.tolist()


def main():
    results = ensure_dir(ROOT / "results")
    board = []
    for (dataset, backbone), summary in DEMO.items():
        name = f"{dataset}__{backbone}"
        run_dir = ensure_dir(results / name)
        summary = {**summary, "demo": True}
        save_json(summary, run_dir / "summary.json")
        history = fake_history(hash(name) % 10_000)
        save_json(history, run_dir / "history.json")
        write_curves(history, run_dir / "training_curves.png")
        cm = write_cm(
            summary["num_classes"],
            run_dir / "confusion_matrix.png",
            f"{name} (demo)",
        )
        metrics = {
            **summary,
            "confusion_matrix": cm,
            "note": "Placeholder demo metrics. Re-run training to replace.",
        }
        save_json(metrics, run_dir / "test_metrics.json")
        save_json(
            {
                "dataset": dataset,
                "backbone": backbone,
                "classes": [f"class_{i}" for i in range(summary["num_classes"])],
                "num_classes": summary["num_classes"],
                "seed": 42,
                "image_size": 224,
                "optimizer": "Adam",
                "scheduler": "ReduceLROnPlateau",
                "strategy": "two_phase_transfer_learning",
                "demo": True,
            },
            run_dir / "meta.json",
        )
        save_json(
            {
                "best_val_acc": summary["accuracy"] - 0.005,
                "best_val_loss": 0.05,
                "checkpoint": f"checkpoints/{name}_best.pt",
            },
            run_dir / "best.json",
        )
        board.append({"run": name, "dataset": dataset, "backbone": backbone, **summary})
        print(f"Wrote demo results for {name}")

    save_json(board, results / "leaderboard.json")
    note = {
        "warning": "These are illustrative demo metrics for UI development.",
        "action": "Run scripts/train_all.py on your A4000 to replace with real results.",
    }
    save_json(note, results / "DEMO_NOTE.json")
    print("Demo results ready under results/")


if __name__ == "__main__":
    main()
