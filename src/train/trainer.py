"""Two-phase transfer learning trainer (Adam + ReduceLROnPlateau + AMP)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.models.backbones import count_trainable, freeze_backbone, unfreeze_backbone
from src.utils import ensure_dir, run_name, save_json


class EarlyStopping:
    def __init__(self, patience: int = 7):
        self.patience = patience
        self.best = float("inf")
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - 1e-6:
            self.best = val_loss
            self.bad_epochs = 0
            return True  # improved
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.should_stop = True
        return False


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device, use_amp: bool = True):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast(device, use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def _autocast(device: torch.device, use_amp: bool):
    enabled = use_amp and device.type == "cuda"
    try:
        return autocast("cuda", enabled=enabled)
    except TypeError:
        return autocast(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device,
    scaler: GradScaler | None,
    use_amp: bool,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="train", leave=True, file=sys.stdout, dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(total, 1), correct / max(total, 1)


def _run_phase(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    weight_decay: float,
    use_amp: bool,
    reduce_lr_cfg: dict,
    early_patience: int,
    ckpt_path: Path,
    history: list[dict],
    phase_name: str,
    best_state: dict,
) -> dict:
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=reduce_lr_cfg.get("factor", 0.5),
        patience=reduce_lr_cfg.get("patience", 3),
        min_lr=reduce_lr_cfg.get("min_lr", 1e-7),
    )
    try:
        scaler = GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    except TypeError:
        scaler = GradScaler(enabled=use_amp and device.type == "cuda")
    early = EarlyStopping(patience=early_patience)

    trainable, total = count_trainable(model)
    print(
        f"[{phase_name}] epochs={epochs} lr={lr} "
        f"trainable={trainable:,}/{total:,}"
    )

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step(val_loss)
        improved = early.step(val_loss)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        record = {
            "phase": phase_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "seconds": elapsed,
        }
        history.append(record)
        print(
            f"  {phase_name} epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
            f"lr={current_lr:.2e}  ({elapsed:.1f}s)"
        )
        if improved:
            best_state["val_loss"] = val_loss
            best_state["val_acc"] = val_acc
            best_state["epoch"] = len(history)
            best_state["model_state_dict"] = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            torch.save(
                {
                    "model_state_dict": best_state["model_state_dict"],
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "history_len": len(history),
                },
                ckpt_path,
            )
        if early.should_stop:
            print(f"  Early stopping at {phase_name} epoch {epoch}")
            break

    # Restore best weights from this or previous phase
    if best_state.get("model_state_dict"):
        model.load_state_dict(best_state["model_state_dict"])
    return best_state


def two_phase_train(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: dict,
    dataset: str,
    backbone: str,
    classes: list[str],
) -> dict[str, Any]:
    """
    Phase 1: freeze backbone, train head.
    Phase 2: unfreeze all, fine-tune at lower LR.
    """
    name = run_name(dataset, backbone)
    ckpt_dir = ensure_dir(Path(cfg["paths"]["checkpoints"]))
    results_dir = ensure_dir(Path(cfg["paths"]["results"]) / name)
    ckpt_path = ckpt_dir / f"{name}_best.pt"

    use_amp = bool(cfg["train"].get("amp", True))
    history: list[dict] = []
    best_state: dict[str, Any] = {}

    meta = {
        "dataset": dataset,
        "backbone": backbone,
        "classes": classes,
        "num_classes": len(classes),
        "seed": cfg["seed"],
        "image_size": cfg["image_size"],
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "strategy": "two_phase_transfer_learning",
    }
    save_json(meta, results_dir / "meta.json")

    # Phase 1 — frozen backbone
    freeze_backbone(model)
    model.to(device)
    _run_phase(
        model,
        train_loader,
        val_loader,
        device,
        epochs=cfg["train"]["phase1_epochs"],
        lr=cfg["train"]["phase1_lr"],
        weight_decay=cfg["train"]["weight_decay"],
        use_amp=use_amp,
        reduce_lr_cfg=cfg["train"]["reduce_lr"],
        early_patience=cfg["train"]["early_stopping_patience"],
        ckpt_path=ckpt_path,
        history=history,
        phase_name="phase1_frozen",
        best_state=best_state,
    )

    # Phase 2 — fine-tune
    unfreeze_backbone(model)
    _run_phase(
        model,
        train_loader,
        val_loader,
        device,
        epochs=cfg["train"]["phase2_epochs"],
        lr=cfg["train"]["phase2_lr"],
        weight_decay=cfg["train"]["weight_decay"],
        use_amp=use_amp,
        reduce_lr_cfg=cfg["train"]["reduce_lr"],
        early_patience=cfg["train"]["early_stopping_patience"],
        ckpt_path=ckpt_path,
        history=history,
        phase_name="phase2_finetune",
        best_state=best_state,
    )

    save_json(history, results_dir / "history.json")
    save_json(
        {
            "best_val_loss": best_state.get("val_loss"),
            "best_val_acc": best_state.get("val_acc"),
            "checkpoint": str(ckpt_path),
        },
        results_dir / "best.json",
    )

    return {
        "checkpoint": str(ckpt_path),
        "results_dir": str(results_dir),
        "history": history,
        "best": best_state,
        "meta": meta,
    }
