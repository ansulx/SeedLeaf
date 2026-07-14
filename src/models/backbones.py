"""Transfer-learning backbones: EfficientNet-B3 and ResNet50."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import timm

BackboneName = Literal["efficientnet_b3", "resnet50"]


def create_model(
    backbone: BackboneName,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Create a classification model with a replaceable head."""
    if backbone not in {"efficientnet_b3", "resnet50"}:
        raise ValueError(f"Unsupported backbone: {backbone}")

    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classifier head."""
    for param in model.parameters():
        param.requires_grad = False

    # timm models expose classifier via get_classifier / reset_classifier
    head = model.get_classifier()
    if isinstance(head, nn.Module):
        for param in head.parameters():
            param.requires_grad = True
    elif head is not None:
        # Rare: classifier is a Parameter
        pass

    # Also unfreeze common head attribute names
    for attr in ("classifier", "fc", "head"):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Module):
            for param in module.parameters():
                param.requires_grad = True


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_target_layer(model: nn.Module, backbone: str) -> nn.Module:
    """Return a late convolutional layer suitable for Grad-CAM."""
    if backbone == "resnet50":
        return model.layer4[-1]
    if backbone == "efficientnet_b3":
        # EfficientNet: last block in conv_head / blocks
        if hasattr(model, "conv_head"):
            return model.conv_head
        if hasattr(model, "blocks"):
            return model.blocks[-1]
    # Fallback: last Conv2d
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM")
    return last_conv


def load_checkpoint(
    path: str,
    num_classes: int,
    backbone: str,
    device: torch.device | None = None,
) -> nn.Module:
    device = device or torch.device("cpu")
    model = create_model(backbone, num_classes, pretrained=False)
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
