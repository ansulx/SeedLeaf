"""Paper-style research results dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.common import page_setup, sidebar_nav_note
from src.eval.metrics import aggregate_leaderboard
from src.utils import load_json

page_setup("Research Results")
sidebar_nav_note()

st.markdown(
    '<p class="sl-brand" style="font-size:2.4rem">Research Results</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sl-tagline">Fair comparison across three datasets and two backbones '
    "under a fixed protocol (seed 42, Adam, two-phase TL, ReduceLROnPlateau).</p>",
    unsafe_allow_html=True,
)

results_root = ROOT / "results"
board = aggregate_leaderboard(results_root)

demo_note = results_root / "DEMO_NOTE.json"
if demo_note.exists():
    st.warning(
        "Showing **demo / placeholder metrics** until you run real training on your A4000. "
        "Replace them with `python scripts/train_all.py`."
    )

if not board:
    st.info(
        "No results yet. Generate placeholders with "
        "`python scripts/generate_demo_results.py` or train models first."
    )
    st.stop()

df = pd.DataFrame(board)
show_cols = [
    c
    for c in [
        "dataset",
        "backbone",
        "accuracy",
        "f1_macro",
        "precision_macro",
        "recall_macro",
        "num_classes",
        "num_test_samples",
        "demo",
    ]
    if c in df.columns
]
view = df[show_cols].copy()
for col in ["accuracy", "f1_macro", "precision_macro", "recall_macro"]:
    if col in view.columns:
        view[col] = view[col].map(lambda x: f"{x:.4f}")

st.markdown("### Leaderboard")
st.dataframe(view, use_container_width=True, hide_index=True)

# Highlight best per dataset
st.markdown("### Best backbone per dataset")
cols = st.columns(3)
for i, dataset in enumerate(["plantvillage", "tomato", "cassava"]):
    sub = df[df["dataset"] == dataset]
    with cols[i]:
        if sub.empty:
            st.markdown(f"**{dataset}** — no runs")
            continue
        best = sub.loc[sub["accuracy"].idxmax()]
        st.markdown(
            f"""
<div class="sl-panel">
  <h3>{dataset}</h3>
  <div class="sl-metric">{best['accuracy']:.2%}</div>
  <div class="sl-muted">{best['backbone']} · F1 {best['f1_macro']:.4f}</div>
</div>
""",
            unsafe_allow_html=True,
        )

st.markdown("### Run detail")
run_names = sorted(df["run"].tolist()) if "run" in df.columns else sorted(
    (results_root / r).name for r in df.apply(lambda r: f"{r['dataset']}__{r['backbone']}", axis=1)
)
# Ensure run column
if "run" not in df.columns:
    df["run"] = df["dataset"] + "__" + df["backbone"]
    run_names = sorted(df["run"].tolist())

selected = st.selectbox("Select run", run_names)
run_dir = results_root / selected

c1, c2 = st.columns(2)
curves = run_dir / "training_curves.png"
cm = run_dir / "confusion_matrix.png"
with c1:
    st.markdown("#### Training curves")
    if curves.exists():
        st.image(str(curves), use_container_width=True)
    else:
        st.caption("No training_curves.png")
with c2:
    st.markdown("#### Confusion matrix")
    if cm.exists():
        st.image(str(cm), use_container_width=True)
    else:
        st.caption("No confusion_matrix.png")

meta_path = run_dir / "meta.json"
summary_path = run_dir / "summary.json"
if meta_path.exists() or summary_path.exists():
    with st.expander("Raw meta / summary JSON"):
        if meta_path.exists():
            st.json(load_json(meta_path))
        if summary_path.exists():
            st.json(load_json(summary_path))

st.markdown("---")
st.caption(
    "Protocol held constant: input 224×224, Adam, ReduceLROnPlateau, "
    "phase-1 frozen head training then phase-2 full fine-tune, seed 42."
)
