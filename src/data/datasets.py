"""Dataset download, organization, and train/val/test splits (seed 42)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from src.data.transforms import get_eval_transforms, get_train_transforms
from src.train.seed import make_generator, seed_worker
from src.utils import ROOT, ensure_dir, save_json


TOMATO_PREFIXES = ("Tomato___", "Tomato_")


class LeafImageDataset(Dataset):
    """Thin wrapper around ImageFolder with optional transform override."""

    def __init__(self, root: str | Path, transform=None):
        self.folder = ImageFolder(str(root), transform=transform)
        self.classes = self.folder.classes
        self.class_to_idx = self.folder.class_to_idx

    def __len__(self) -> int:
        return len(self.folder)

    def __getitem__(self, idx: int):
        return self.folder[idx]


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _copy_or_link(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def collect_class_images(source_root: Path) -> dict[str, list[Path]]:
    """Collect images from class-subdirectory layout."""
    class_map: dict[str, list[Path]] = {}
    if not source_root.exists():
        return class_map

    # Prefer direct class folders
    for child in sorted(source_root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            images = [p for p in child.rglob("*") if p.is_file() and _is_image(p)]
            if images:
                class_map[child.name] = images

    # Flatten nested PlantVillage-style layouts (color/grayscale/segmented)
    if not class_map:
        for img in source_root.rglob("*"):
            if img.is_file() and _is_image(img):
                cls = img.parent.name
                class_map.setdefault(cls, []).append(img)
    return class_map


def split_and_materialize(
    class_map: dict[str, list[Path]],
    out_root: Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """Stratified split per class into train/val/test folders."""
    ensure_dir(out_root)
    summary = {"classes": {}, "counts": {"train": 0, "val": 0, "test": 0}}

    for cls, paths in sorted(class_map.items()):
        paths = sorted(paths)
        if len(paths) < 3:
            # Too few for 3-way split — put all in train
            for p in paths:
                _copy_or_link(p, out_root / "train" / cls / p.name)
            summary["classes"][cls] = {"train": len(paths), "val": 0, "test": 0}
            summary["counts"]["train"] += len(paths)
            continue

        train_val, test = train_test_split(
            paths, test_size=test_ratio, random_state=seed
        )
        relative_val = val_ratio / (1.0 - test_ratio)
        train, val = train_test_split(
            train_val, test_size=relative_val, random_state=seed
        )

        for split_name, split_paths in (
            ("train", train),
            ("val", val),
            ("test", test),
        ):
            for p in split_paths:
                _copy_or_link(p, out_root / split_name / cls / p.name)
            summary["classes"][cls] = summary["classes"].get(cls, {})
            summary["classes"][cls][split_name] = len(split_paths)
            summary["counts"][split_name] += len(split_paths)

    save_json(summary, out_root / "split_summary.json")
    with open(out_root / "classes.json", "w", encoding="utf-8") as f:
        json.dump(sorted(class_map.keys()), f, indent=2)
    return summary


def download_plantvillage_hf(raw_dir: Path) -> Path:
    """Download PlantVillage from Hugging Face datasets."""
    ensure_dir(raw_dir)
    marker = raw_dir / ".download_complete"
    if marker.exists():
        return raw_dir

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Install `datasets` and `huggingface_hub` to download PlantVillage."
        ) from e

    print("Downloading PlantVillage from Hugging Face (this may take a while)...")
    sys.stdout.flush()
    # Official HF port of Mohanty et al. PlantVillage (color images)
    try:
        ds_dict = load_dataset("mohanty/PlantVillage", "color")
        # Merge official splits; we re-split with seed 42 ourselves
        from datasets import concatenate_datasets

        parts = [ds_dict[k] for k in ds_dict.keys()]
        ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    except Exception as primary_err:
        print(f"  mohanty/PlantVillage failed ({primary_err}); trying geraldmc/plantvillage-full...")
        ds = load_dataset("geraldmc/plantvillage-full", split="train")
    print(f"Loaded {len(ds)} samples — writing class folders...", flush=True)

    # Materialize to class folders
    out = raw_dir / "images"
    ensure_dir(out)
    for i, row in enumerate(ds):
        label = (
            row.get("label_name")
            or row.get("class_label")
            or row.get("label")
            or "unknown"
        )
        if isinstance(label, int):
            names = None
            if "label" in getattr(ds, "features", {}):
                feat = ds.features["label"]
                names = getattr(feat, "names", None)
            label = names[label] if names else str(label)
        label = str(label).replace("/", "_").replace(" ", "_")
        img = row["image"]
        cls_dir = out / label
        ensure_dir(cls_dir)
        dest = cls_dir / f"{i:06d}.jpg"
        if not dest.exists():
            if hasattr(img, "save"):
                img.convert("RGB").save(dest, quality=95)
            else:
                Image.open(img).convert("RGB").save(dest, quality=95)
        if (i + 1) % 2000 == 0:
            print(f"  wrote {i + 1}/{len(ds)} images...", flush=True)

    marker.write_text("ok", encoding="utf-8")
    return out


def _kaggle_download_dataset(dataset_id: str, raw_dir: Path) -> None:
    """Download a Kaggle dataset using the Python API (CLI may not be on PATH)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise ImportError(
            "Install `kaggle` and place credentials at %USERPROFILE%\\.kaggle\\kaggle.json"
        ) from e

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading Kaggle dataset: {dataset_id}")
    api.dataset_download_files(dataset_id, path=str(raw_dir), unzip=True)


def download_plantvillage_kaggle(raw_dir: Path) -> Path:
    """Download PlantVillage via Kaggle API if HF fails."""
    ensure_dir(raw_dir)
    marker = raw_dir / ".download_complete"
    if marker.exists():
        color = list(raw_dir.rglob("color"))
        if color:
            return color[0]
        return raw_dir

    print("Downloading PlantVillage from Kaggle...")
    _kaggle_download_dataset("abdallahalidev/plantvillage-dataset", raw_dir)
    marker.write_text("ok", encoding="utf-8")
    color = list(raw_dir.rglob("color"))
    return color[0] if color else raw_dir


def download_cassava_hf(raw_dir: Path) -> Path:
    """Download Cassava from Hugging Face when Kaggle credentials are missing."""
    ensure_dir(raw_dir)
    marker = raw_dir / ".download_complete"
    if marker.exists():
        return raw_dir

    from datasets import concatenate_datasets, load_dataset

    # Prefer larger / competition-style mirrors when HF_TOKEN can access gated data.
    candidates = [
        "pufanyi/cassava-leaf-disease-classification",
        "dpdl-benchmark/cassava",
        "cayala/cassava",
        "andrewkatumba/cassava_leaf_diseases_dsa_2023",
    ]
    label_map = {
        0: "Cassava_Bacterial_Blight",
        1: "Cassava_Brown_Streak_Disease",
        2: "Cassava_Green_Mottle",
        3: "Cassava_Mosaic_Disease",
        4: "Healthy",
        "cbsd": "Cassava_Brown_Streak_Disease",
        "cmd": "Cassava_Mosaic_Disease",
        "cgm": "Cassava_Green_Mottle",
        "cbb": "Cassava_Bacterial_Blight",
        "healthy": "Healthy",
    }

    ds = None
    used = None
    last_err: Exception | None = None
    for name in candidates:
        try:
            print(f"Downloading Cassava from Hugging Face: {name} ...", flush=True)
            loaded = load_dataset(name)
            parts = [loaded[k] for k in loaded.keys()]
            ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
            used = name
            print(f"  loaded {len(ds)} samples from {name}", flush=True)
            break
        except Exception as e:
            last_err = e
            print(f"  failed {name}: {e}", flush=True)

    if ds is None:
        raise FileNotFoundError(
            f"No public Cassava HF dataset available ({last_err}). "
            "Place class folders or train.csv + train_images/ under data/raw/cassava."
        )

    out = raw_dir / "images"
    ensure_dir(out)
    names = None
    if "label" in getattr(ds, "features", {}):
        names = getattr(ds.features["label"], "names", None)

    for i, row in enumerate(ds):
        label = row.get("label", 0)
        if isinstance(label, str):
            cls = label_map.get(label.lower(), label.replace(" ", "_"))
        else:
            idx = int(label)
            if names:
                raw_name = names[idx]
                cls = label_map.get(str(raw_name).lower(), str(raw_name).replace(" ", "_"))
            else:
                cls = label_map.get(idx, f"class_{idx}")
        img = row["image"] if "image" in row else row.get("img")
        cls_dir = out / cls
        ensure_dir(cls_dir)
        dest = cls_dir / f"{i:06d}.jpg"
        if not dest.exists():
            if hasattr(img, "save"):
                img.convert("RGB").save(dest, quality=95)
            else:
                Image.open(img).convert("RGB").save(dest, quality=95)
        if (i + 1) % 500 == 0:
            print(f"  wrote {i + 1}/{len(ds)} cassava images...", flush=True)

    (raw_dir / ".hf_source").write_text(used or "", encoding="utf-8")
    marker.write_text("ok", encoding="utf-8")
    return out


def download_cassava_kaggle(raw_dir: Path) -> Path:
    """Download Cassava leaf disease dataset from Kaggle."""
    ensure_dir(raw_dir)
    marker = raw_dir / ".download_complete"
    if marker.exists():
        return raw_dir

    print("Downloading Cassava from Kaggle...")
    try:
        _kaggle_download_dataset(
            "agranamanyaa/cassava-leaf-disease-classification", raw_dir
        )
    except Exception as e:
        print(f"Primary Cassava dataset failed ({e}); trying alternate...")
        try:
            _kaggle_download_dataset(
                "nirmalsankalana/cassava-leaf-disease-classification", raw_dir
            )
        except Exception as e2:
            print(f"Kaggle Cassava failed ({e2}); trying Hugging Face...")
            return download_cassava_hf(raw_dir)
    marker.write_text("ok", encoding="utf-8")
    return raw_dir


def organize_cassava(raw_dir: Path, processed_dir: Path, cfg: dict) -> Path:
    """Organize cassava into class folders then split."""
    if (processed_dir / "split_summary.json").exists():
        return processed_dir

    # HF downloads land in raw/cassava/images/<class>/ — prefer that over
    # treating the parent "images" folder as a single class.
    search_roots = []
    images_root = raw_dir / "images"
    if images_root.is_dir():
        search_roots.append(images_root)
    search_roots.append(raw_dir)

    class_map: dict[str, list[Path]] = {}
    for root in search_roots:
        class_map = collect_class_images(root)
        # Ignore a single dump folder mistakenly named "images"
        if set(class_map.keys()) == {"images"}:
            class_map = {}
            continue
        if class_map:
            break

    if not class_map:
        # Case B: train.csv + train_images/
        csv_path = next(raw_dir.rglob("train.csv"), None)
        img_dir = next(
            (p for p in raw_dir.rglob("train_images") if p.is_dir()),
            None,
        )
        if csv_path and img_dir:
            import pandas as pd

            df = pd.read_csv(csv_path)
            label_map = {
                0: "Cassava_Bacterial_Blight",
                1: "Cassava_Brown_Streak_Disease",
                2: "Cassava_Green_Mottle",
                3: "Cassava_Mosaic_Disease",
                4: "Healthy",
            }
            class_map = {}
            for _, row in df.iterrows():
                img_id = row["image_id"]
                label = int(row["label"])
                cls = label_map.get(label, f"class_{label}")
                src = img_dir / img_id
                if src.exists():
                    class_map.setdefault(cls, []).append(src)

    if not class_map:
        raise FileNotFoundError(
            f"Could not find cassava images under {raw_dir}. "
            "Place class folders or train.csv + train_images/ there."
        )

    print(
        f"[cassava] organizing {sum(len(v) for v in class_map.values())} images "
        f"across {len(class_map)} classes: {sorted(class_map.keys())}",
        flush=True,
    )
    split_and_materialize(
        class_map,
        processed_dir,
        val_ratio=cfg["train"]["val_ratio"],
        test_ratio=cfg["train"]["test_ratio"],
        seed=cfg["seed"],
    )
    return processed_dir


def filter_tomato(class_map: dict[str, list[Path]]) -> dict[str, list[Path]]:
    tomato = {}
    for cls, paths in class_map.items():
        if cls.startswith(TOMATO_PREFIXES) or "Tomato" in cls or "tomato" in cls.lower():
            tomato[cls] = paths
    return tomato


def prepare_dataset(name: str, cfg: dict | None = None) -> Path:
    """
    Ensure processed train/val/test layout exists for a dataset.

    Returns path to processed root: data/<name>/{train,val,test}/
    """
    from src.utils import load_config

    cfg = cfg or load_config()
    data_root = ROOT / cfg["data"]["root"]
    processed = data_root / name
    if (processed / "split_summary.json").exists():
        print(f"[{name}] already prepared at {processed}")
        return processed

    if name == "plantvillage":
        raw = data_root / "raw" / "plantvillage"
        try:
            images_root = download_plantvillage_hf(raw)
        except Exception as e:
            print(f"HF download failed ({e}); trying Kaggle...")
            images_root = download_plantvillage_kaggle(raw)
        # Prefer color folder if nested
        color = list(Path(images_root).rglob("color")) if Path(images_root).is_dir() else []
        source = color[0] if color else Path(images_root)
        class_map = collect_class_images(source)
        if not class_map:
            raise FileNotFoundError(f"No PlantVillage class folders under {source}")
        split_and_materialize(
            class_map,
            processed,
            val_ratio=cfg["train"]["val_ratio"],
            test_ratio=cfg["train"]["test_ratio"],
            seed=cfg["seed"],
        )

    elif name == "tomato":
        # Reuse PlantVillage raw/processed
        pv_processed = data_root / "plantvillage"
        if not (pv_processed / "split_summary.json").exists():
            prepare_dataset("plantvillage", cfg)
        # Build tomato from plantvillage train+val+test OR from raw
        raw_pv = data_root / "raw" / "plantvillage"
        source_candidates = [
            data_root / "raw" / "plantvillage" / "images",
            *list(raw_pv.rglob("color")),
        ]
        class_map = {}
        for cand in source_candidates:
            if cand.exists():
                class_map = collect_class_images(cand)
                if class_map:
                    break
        if not class_map and pv_processed.exists():
            # Collect from all splits
            for split in ("train", "val", "test"):
                split_dir = pv_processed / split
                if split_dir.exists():
                    for cls, paths in collect_class_images(split_dir).items():
                        class_map.setdefault(cls, []).extend(paths)
        tomato_map = filter_tomato(class_map)
        if not tomato_map:
            raise FileNotFoundError(
                "No tomato classes found. Prepare plantvillage first, "
                "or place Tomato___* folders under data/raw/plantvillage."
            )
        split_and_materialize(
            tomato_map,
            processed,
            val_ratio=cfg["train"]["val_ratio"],
            test_ratio=cfg["train"]["test_ratio"],
            seed=cfg["seed"],
        )

    elif name == "cassava":
        raw = data_root / "raw" / "cassava"
        # Allow manual placement without download
        if not (raw / ".download_complete").exists() and not any(raw.glob("*")):
            ensure_dir(raw)
            try:
                download_cassava_kaggle(raw)
            except Exception as e:
                raise FileNotFoundError(
                    f"Cassava download failed ({e}). "
                    f"Manually place the dataset under {raw} "
                    "(class folders or train.csv + train_images/), then re-run."
                ) from e
        organize_cassava(raw, processed, cfg)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    print(f"[{name}] prepared at {processed}")
    return processed


def build_dataloaders(
    dataset_name: str,
    cfg: dict | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train/val/test loaders for a prepared dataset."""
    from src.utils import load_config

    cfg = cfg or load_config()
    root = prepare_dataset(dataset_name, cfg)
    image_size = cfg["image_size"]
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg.get("num_workers", 4)
    seed = cfg["seed"]

    train_ds = LeafImageDataset(root / "train", transform=get_train_transforms(image_size))
    val_ds = LeafImageDataset(root / "val", transform=get_eval_transforms(image_size))
    test_ds = LeafImageDataset(root / "test", transform=get_eval_transforms(image_size))

    # Align class order to train
    classes = train_ds.classes
    g = make_generator(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cfg.get("pin_memory", True),
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cfg.get("pin_memory", True),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cfg.get("pin_memory", True),
    )
    return train_loader, val_loader, test_loader, classes
