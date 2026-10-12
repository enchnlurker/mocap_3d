"""Data contract (DESIGN 5): one npz per stage in data/<clip>/.

Every file carries schema_version, fps and joint_names.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = DATA / "models"


class Clip:
    """Paths for one clip directory."""

    def __init__(self, clip: str | Path):
        p = Path(clip)
        self.dir = p if p.is_dir() or p.is_absolute() or p.parent != Path(".") else DATA / p
        self.name = self.dir.name

    def __truediv__(self, name: str) -> Path:
        return self.dir / name

    @property
    def video(self) -> Path:
        for name in ("source.mp4", "source.mov", "source.MOV", "source.m4v"):
            if (self.dir / name).exists():
                return self.dir / name
        return self.dir / "source.mp4"

    @property
    def capture(self) -> Path:
        return self.dir / "capture_mediapipe.npz"

    @property
    def depth(self) -> Path:
        return self.dir / "depth.npz"

    @property
    def depth_joints(self) -> Path:
        return self.dir / "depth_joints.npz"

    @property
    def clean(self) -> Path:
        """clean_depth.npz when stage 2 used depth, else clean.npz."""
        d = self.dir / "clean_depth.npz"
        return d if d.exists() else self.dir / "clean.npz"


def save(path: str | Path, fps: float, joint_names, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, schema_version=SCHEMA_VERSION, fps=float(fps),
                        joint_names=np.array(list(joint_names)), **arrays)
    os.replace(tmp, path)
    return path


def load(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as f:
        out = {k: f[k] for k in f.files}
    v = int(out.get("schema_version", -1))
    if v != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {v}, expected {SCHEMA_VERSION}")
    out["fps"] = float(out["fps"])
    return out


def update(path: str | Path, **arrays) -> Path:
    """Add or replace keys in an existing stage file (e.g. 2c adding to clean)."""
    d = load(path)
    d.update(arrays)
    fps, names = d.pop("fps"), d.pop("joint_names")
    d.pop("schema_version")
    return save(path, fps, names, **d)


def scalar(d: dict, key: str, default=None):
    if key not in d:
        return default
    v = d[key]
    return v.item() if np.ndim(v) == 0 else v
