#!/usr/bin/env python3
"""Run full training with line-buffered logging for live watching."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONUNBUFFERED"] = "1"

# Force tqdm / print flush
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

from scripts.train_all import main  # noqa: E402

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("SeedLeaf training started — watch this terminal / logs/training.log", flush=True)
    print("=" * 60, flush=True)
    main()
