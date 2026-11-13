"""Shared paths and helpers for the pipeline scripts."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
RAW_TRAIN_IMAGES = RAW_DIR / "stage_2_train_images"
RAW_LABELS_CSV = RAW_DIR / "stage_2_train_labels.csv"
RAW_CLASS_INFO_CSV = RAW_DIR / "stage_2_detailed_class_info.csv"

YOLO_DIR = ROOT / "data" / "yolo"
SPLITS_DIR = ROOT / "splits"
RUNS_DIR = ROOT / "runs" / "detect"
MODELS_DIR = ROOT / "models"

SPLITS = ("train", "val", "test")


def dcm_path(patient_id: str) -> Path:
    return RAW_TRAIN_IMAGES / f"{patient_id}.dcm"


def load_split(split: str) -> pd.DataFrame:
    path = SPLITS_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python src/make_splits.py` first.")
    return pd.read_csv(path)


def load_boxes() -> pd.DataFrame:
    """RSNA box rows (pixel coords, top-left origin). Only rows with Target == 1 carry boxes."""
    df = pd.read_csv(RAW_LABELS_CSV)
    return df[df["Target"] == 1].reset_index(drop=True)


def splits_fingerprint() -> str:
    """Short hash of the committed split files, logged with every run to prove which split was used."""
    h = hashlib.sha256()
    for split in SPLITS:
        h.update((SPLITS_DIR / f"{split}.csv").read_bytes())
    return h.hexdigest()[:12]


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a path from a config file relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
