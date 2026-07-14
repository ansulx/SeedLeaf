"""Grad-CAM visualization for leaf disease predictions."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_eval_transforms
from src.models.backbones import get_target_layer


class GradCAM:
    """Minimal Grad-CAM using forward/backward hooks on a target conv layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = []
        self._handles.append(
            target_layer.register_forward_hook(self._forward_hook)
        )
        self._handles.append(
            target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients  # (1, C, H, W)
        acts = self.activations
        if grads is None or acts is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_cam_on_image(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay heatmap on RGB uint8 image."""
    h, w = image_rgb.shape[:2]
    heatmap = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (
        alpha * heatmap_color.astype(np.float32)
        + (1 - alpha) * image_rgb.astype(np.float32)
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def predict_with_gradcam(
    model: nn.Module,
    backbone: str,
    pil_image: Image.Image,
    class_names: list[str],
    device: torch.device,
    image_size: int = 224,
    top_k: int = 5,
) -> dict:
    """Run inference + Grad-CAM for a single PIL image."""
    rgb = pil_image.convert("RGB")
    original = np.array(rgb)
    transform = get_eval_transforms(image_size)
    tensor = transform(rgb).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_k = min(top_k, len(class_names))
    top_idx = probs.argsort()[::-1][:top_k]
    predictions = [
        {"class": class_names[i], "confidence": float(probs[i]), "index": int(i)}
        for i in top_idx
    ]
    pred_idx = int(top_idx[0])

    target = get_target_layer(model, backbone)
    cam_engine = GradCAM(model, target)
    try:
        cam = cam_engine(tensor, class_idx=pred_idx)
        overlay = overlay_cam_on_image(original, cam)
    finally:
        cam_engine.remove_hooks()

    return {
        "predictions": predictions,
        "predicted_class": predictions[0]["class"],
        "confidence": predictions[0]["confidence"],
        "original_image": original,
        "gradcam_overlay": overlay,
        "cam": cam,
    }


def tensor_to_display(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized CHW tensor to RGB uint8."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (image_tensor.cpu() * std + mean).clamp(0, 1)
    arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return arr
