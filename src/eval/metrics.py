"""Metrics export: accuracy, F1, confusion matrix, training curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

from src.utils import ensure_dir, save_json


def _autocast(device: torch.device, use_amp: bool):
    enabled = use_amp and device.type == "cuda"
    try:
        return autocast("cuda", enabled=enabled)
    except TypeError:
        return autocast(enabled=enabled)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with _autocast(device, use_amp):
            logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    cm = np.asarray(cm)
    n = len(class_names)
    fig_w = max(8, min(24, n * 0.45))
    fig_h = max(6, min(22, n * 0.4))
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(
        cm,
        annot=n <= 20,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names if n <= 30 else False,
        yticklabels=class_names if n <= 30 else False,
        cbar=True,
    )
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(history: list[dict], out_path: Path) -> None:
    if not history:
        return
    epochs = list(range(1, len(history) + 1))
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    phases = [h.get("phase", "") for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train", color="#1b5e20")
    axes[0].plot(epochs, val_loss, label="val", color="#66bb6a")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch (global)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_acc, label="train", color="#1b5e20")
    axes[1].plot(epochs, val_acc, label="val", color="#66bb6a")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch (global)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Mark phase boundary
    if "phase2_finetune" in phases:
        boundary = phases.index("phase2_finetune") + 0.5
        for ax in axes:
            ax.axvline(boundary, color="#2e7d32", linestyle="--", alpha=0.6, label="phase2")

    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_and_export(
    model: nn.Module,
    test_loader,
    class_names: list[str],
    device: torch.device,
    results_dir: str | Path,
    history: list[dict] | None = None,
    use_amp: bool = True,
    title_prefix: str = "",
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    ensure_dir(results_dir)

    y_true, y_pred = collect_predictions(model, test_loader, device, use_amp)
    metrics = compute_metrics(y_true, y_pred, class_names)
    save_json(metrics, results_dir / "test_metrics.json")

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        results_dir / "confusion_matrix.png",
        title=f"{title_prefix} Confusion Matrix".strip(),
    )

    if history:
        plot_training_curves(history, results_dir / "training_curves.png")
        save_json(history, results_dir / "history.json")

    # Compact summary for dashboard aggregation
    summary = {
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "num_classes": len(class_names),
        "num_test_samples": int(len(y_true)),
    }
    save_json(summary, results_dir / "summary.json")
    print(
        f"Test accuracy={summary['accuracy']:.4f}  "
        f"F1(macro)={summary['f1_macro']:.4f}"
    )
    return metrics


def aggregate_leaderboard(results_root: str | Path) -> list[dict[str, Any]]:
    """Scan results/*/summary.json into a comparison table."""
    results_root = Path(results_root)
    rows = []
    if not results_root.exists():
        return rows
    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        meta_path = run_dir / "meta.json"
        if not summary_path.exists():
            continue
        from src.utils import load_json

        summary = load_json(summary_path)
        meta = load_json(meta_path) if meta_path.exists() else {}
        dataset, _, backbone = run_dir.name.partition("__")
        rows.append(
            {
                "run": run_dir.name,
                "dataset": meta.get("dataset", dataset),
                "backbone": meta.get("backbone", backbone),
                **summary,
            }
        )
    return rows
