"""About & reproducibility."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.common import page_setup, sidebar_nav_note

page_setup("About")
sidebar_nav_note()

st.markdown(
    '<p class="sl-brand" style="font-size:2.4rem">About & Reproducibility</p>',
    unsafe_allow_html=True,
)

st.markdown(
    """
SeedLeaf is a portfolio-ready research demo for **plant leaf disease classification**.
It packages a fixed experimental protocol, offline GPU training, and a Streamlit
interface for inference and paper-style comparison.
"""
)

st.markdown("### Hardware target")
st.code("NVIDIA RTX A4000 · mixed precision (AMP) · batch size 32", language="text")

st.markdown("### Environment")
st.code(
    """python -m venv .venv
# Windows:
.venv\\Scripts\\activate
pip install -r requirements.txt
""",
    language="bash",
)

st.markdown("### Data")
st.markdown(
    """
1. Configure Kaggle credentials (`~/.kaggle/kaggle.json`) for Cassava / optional PlantVillage.
2. Prepare splits:

```bash
python scripts/prepare_data.py --datasets plantvillage tomato cassava
```

Tomato is filtered from PlantVillage `Tomato___*` classes. Splits use **seed 42**.
"""
)

st.markdown("### Training (A4000)")
st.code(
    """# Single run
python scripts/train_one.py --dataset tomato --backbone efficientnet_b3

# Full 3×2 grid
python scripts/train_all.py

# Optional: skip finished runs
python scripts/train_all.py --skip-existing
""",
    language="bash",
)

st.markdown("### Launch Streamlit")
st.code("streamlit run app/Home.py", language="bash")

st.markdown("### Fixed experimental protocol")
st.markdown(
    """
| Setting | Value |
|---------|-------|
| Seed | 42 |
| Input size | 224 × 224 |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 3) |
| Phase 1 | Freeze backbone, train head (lr=1e-3) |
| Phase 2 | Unfreeze, fine-tune (lr=1e-4) |
| Backbones | EfficientNet-B3, ResNet50 |
| Datasets | PlantVillage, Tomato, Cassava |
"""
)

st.markdown("### Project layout")
st.code(
    """SeedLeaf/
  configs/default.yaml
  src/data  models  train  eval  explain
  scripts/train_one.py  train_all.py  evaluate.py
  app/Home.py  pages/
  checkpoints/  results/  data/
""",
    language="text",
)

st.caption(
    "Not a SOTA claim — a clear, reproducible baseline stack designed for demos and reports."
)
