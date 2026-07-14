"""Shared Streamlit helpers."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
STYLES = Path(__file__).resolve().parent / "styles.css"


def inject_css() -> None:
    if STYLES.exists():
        st.markdown(f"<style>{STYLES.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_setup(title: str, icon: str = "🌿") -> None:
    st.set_page_config(
        page_title=f"SeedLeaf · {title}",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    # Ensure project root is on path
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def sidebar_nav_note() -> None:
    st.sidebar.markdown("### SeedLeaf")
    st.sidebar.caption(
        "Research-grade plant disease classification · "
        "EfficientNet-B3 + ResNet50 · seed 42"
    )
