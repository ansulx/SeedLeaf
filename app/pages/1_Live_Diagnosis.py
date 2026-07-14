"""Live leaf disease diagnosis + Grad-CAM."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.common import page_setup, sidebar_nav_note
from src.inference import (
    format_disease_name,
    list_available_runs,
    load_class_names,
)

page_setup("Live Diagnosis")
sidebar_nav_note()

st.markdown('<p class="sl-brand" style="font-size:2.4rem">Live Diagnosis</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sl-tagline">Upload a leaf photo. The selected checkpoint returns '
    "class probabilities and a Grad-CAM attention map.</p>",
    unsafe_allow_html=True,
)

runs = [r for r in list_available_runs() if r["has_checkpoint"]]
if not runs:
    st.warning(
        "No trained checkpoints found under `checkpoints/`. "
        "Train with `python scripts/train_one.py --dataset tomato --backbone efficientnet_b3` "
        "or place `*_best.pt` files in `checkpoints/`.\n\n"
        "You can still browse **Research Results** (demo metrics are available)."
    )
    st.stop()

labels = {
    f"{r['dataset']} · {r['backbone']}": r for r in runs
}
choice = st.selectbox("Model", list(labels.keys()))
run = labels[choice]
dataset, backbone = run["dataset"], run["backbone"]

classes = load_class_names(dataset, backbone)
if not classes:
    st.error(
        f"Could not resolve class names for `{dataset}`. "
        "Prepare data (`python scripts/prepare_data.py`) or ensure `results/.../meta.json` has real class labels."
    )
    st.stop()

uploaded = st.file_uploader(
    "Leaf image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Clear, well-lit leaf photos work best.",
)

col_l, col_r = st.columns(2)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    with col_l:
        st.markdown("#### Input")
        st.image(image, use_container_width=True)

    with st.spinner("Running inference + Grad-CAM..."):
        import torch

        from src.explain.gradcam import predict_with_gradcam
        from src.inference import get_model

        packed = get_model(dataset, backbone, len(classes))
        if packed is None:
            st.error("Failed to load checkpoint.")
            st.stop()
        model, device = packed
        result = predict_with_gradcam(
            model,
            backbone,
            image,
            classes,
            device,
            image_size=224,
            top_k=min(5, len(classes)),
        )

    with col_r:
        st.markdown("#### Grad-CAM")
        st.image(result["gradcam_overlay"], use_container_width=True)

    pred = format_disease_name(result["predicted_class"])
    conf = result["confidence"]
    st.markdown(
        f"""
<div class="sl-panel">
  <div class="sl-muted">Predicted class</div>
  <div class="sl-metric">{pred}</div>
  <div>Confidence: <strong>{conf:.1%}</strong> · device: <code>{device}</code></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("#### Top predictions")
    for p in result["predictions"]:
        st.progress(
            min(float(p["confidence"]), 1.0),
            text=f"{format_disease_name(p['class'])} — {p['confidence']:.1%}",
        )
else:
    st.info("Upload a leaf image to begin.")
