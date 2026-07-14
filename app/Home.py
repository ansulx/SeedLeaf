"""SeedLeaf — Home."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

from app.common import page_setup, sidebar_nav_note

page_setup("Home")
sidebar_nav_note()

st.markdown('<p class="sl-brand">SeedLeaf</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sl-tagline">Transfer-learning baselines for leaf disease recognition '
    "across PlantVillage, Tomato, and Cassava — trained with a reproducible "
    "two-phase protocol, explained with Grad-CAM.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    """
<span class="sl-chip">Adam</span>
<span class="sl-chip">Two-phase transfer learning</span>
<span class="sl-chip">ReduceLROnPlateau</span>
<span class="sl-chip">seed = 42</span>
<span class="sl-chip">224 × 224</span>
<span class="sl-chip">RTX A4000 ready</span>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
<div class="sl-panel">
<h3>PlantVillage</h3>
<p class="sl-muted">Broad multi-crop leaf disease benchmark. Strong transfer-learning ceiling; used as the primary large-scale corpus.</p>
</div>
""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
<div class="sl-panel">
<h3>Tomato</h3>
<p class="sl-muted">Focused crop study drawn from Tomato___ classes in PlantVillage — a clean single-species evaluation slice.</p>
</div>
""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
<div class="sl-panel">
<h3>Cassava</h3>
<p class="sl-muted">Harder, field-like Kaggle disease set. The more credible stress test versus near-saturated PlantVillage scores.</p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("## Methodology")
st.markdown(
    """
**Backbones.** EfficientNet-B3 and ResNet50, ImageNet-pretrained, classifier head replaced per dataset.

**Phase 1 — frozen backbone.** Train only the classification head with Adam (`lr=1e-3`).

**Phase 2 — fine-tune.** Unfreeze the full network and continue with Adam at a lower learning rate (`lr=1e-4`).

**Schedule.** `ReduceLROnPlateau` monitors validation loss (factor 0.5, patience 3).

**Reproducibility.** Global seed **42** across Python, NumPy, and PyTorch; images resized to **224×224** with ImageNet normalization.
"""
)

st.markdown("## Explore")
a, b = st.columns(2)
with a:
    st.info("**Live Diagnosis** — upload a leaf image, get top-k predictions and Grad-CAM.")
with b:
    st.info("**Research Results** — dataset × backbone comparison, curves, confusion matrices.")

st.caption(
    "Scores on PlantVillage are often near ceiling with modern CNNs; "
    "SeedLeaf reports strong reproducible baselines — not absolute SOTA claims."
)
