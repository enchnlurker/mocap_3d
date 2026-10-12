"""Check images: before/after cleanup, 2D overlay of fits on the footage, 3D views.

    python -m core.plots clip01      # camera_check.jpg + fullbody_check.jpg for one clip
    python -m core.plots --all       # data/camera_check_all.jpg across clips
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np

from core import schema, video
from core.skeleton import BONE_PAIRS, J

PLOT_JOINTS = ["head", "l_wrist", "r_wrist", "l_ankle", "r_ankle"]
COLORS = {"raw": "#9aa0a6", "clean": "#1a73e8", "2b": "#e37400", "fit2d": "#188038"}


def clean_before_after(out: dict, fps: float, path) -> None:
    raw, pos = out["positions_raw"], out["positions"]
    inf = out["inferred"]
    t = np.arange(len(pos)) / fps
    fig, axes = plt.subplots(len(PLOT_JOINTS), 3, figsize=(14, 2.2 * len(PLOT_JOINTS)), sharex=True)
    for r, name in enumerate(PLOT_JOINTS):
        j = J[name]
        for c, ax_name in enumerate("XYZ"):
            ax = axes[r, c]
            ax.plot(t, raw[:, j, c], color=COLORS["raw"], lw=0.8, label="raw")
            cl = np.where(inf[:, j], np.nan, pos[:, j, c])
            ax.plot(t, cl, color=COLORS["clean"], lw=1.2, label="clean")
            ax.plot(t, np.where(inf[:, j], pos[:, j, c], np.nan), color=COLORS["clean"], lw=1.2, ls=":",
                    label="inferred")
            ax.set_ylabel(f"{name} {ax_name} (m)", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("s")
    fig.suptitle("stage 2: raw MediaPipe (grey) vs clean (blue), world frame, Z up", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _draw(img, uv, color, thick=2):
    for a, b in BONE_PAIRS:
        if np.isfinite(uv[[a, b]]).all():
            cv2.line(img, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])), color, thick, cv2.LINE_AA)


def camera_check(clip: schema.Clip, n: int = 6, width: int = 480) -> np.ndarray:
    """Frames with MediaPipe 2D (white) and the fit2d projection (green)."""
    clean = schema.load(clip.clean)
    if "positions_fit2d_cam" not in clean:
        raise SystemExit(f"{clip.name}: run core.fit2d first")
    W, H = int(clean["width"]), int(clean["height"])
    uv = clean["uv_2d"] * np.array([W, H])
    P = clean["positions_fit2d_cam"]
    f = float(clean["fit2d_focal_px"])
    proj = f * P[..., :2] / P[..., 2:3] + np.array([(W - 1) / 2, (H - 1) / 2])
    idx = video.evenly_spaced(len(P), n)
    tiles = []
    for i, rgb in video.frames(clip.video, idx):
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _draw(img, uv[i], (255, 255, 255), 3)
        _draw(img, proj[i], (60, 200, 60), 2)
        cv2.putText(img, f"{clip.name} f{i}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        tiles.append(cv2.resize(img, (width, int(width * H / W))))
    return np.hstack(tiles) if tiles else np.zeros((10, 10, 3), np.uint8)


def fullbody_check(clip: schema.Clip, n: int = 6, path=None) -> None:
    """Side and front views of 2b (orange) vs fit2d (green) on sampled frames."""
    clean = schema.load(clip.clean)
    keys = [("positions_proportional", "2b"), ("positions_fit2d", "fit2d")]
    keys = [(k, c) for k, c in keys if k in clean]
    T = len(clean["positions"])
    idx = video.evenly_spaced(T, n)
    fig, axes = plt.subplots(2, len(idx), figsize=(2.6 * len(idx), 6), squeeze=False)
    for c, t in enumerate(idx):
        for r, (a, b, lab) in enumerate([(0, 2, "front X-Z"), (1, 2, "side Y-Z")]):
            ax = axes[r, c]
            for k, col in keys:
                P = clean[k][t]
                for i, j in BONE_PAIRS:
                    ax.plot(P[[i, j], a], P[[i, j], b], color=COLORS[col], lw=1.5)
            ax.axhline(0, color="#ccc", lw=0.8)
            ax.set_aspect("equal")
            ax.set_title(f"f{t} {lab}", fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"{clip.name}: 2b (orange) vs fit2d (green)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path or clip / "fullbody_check.jpg", dpi=100)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    if a.all:
        rows = []
        for d in sorted(p for p in schema.DATA.iterdir() if p.is_dir() and p.name != "models"):
            c = schema.Clip(d)
            if c.clean.exists() and c.video.exists():
                rows.append(camera_check(c))
        if rows:
            w = max(r.shape[1] for r in rows)
            rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
            cv2.imwrite(str(schema.DATA / "camera_check_all.jpg"), np.vstack(rows))
            print(f"-> {schema.DATA / 'camera_check_all.jpg'}")
        return
    clip = schema.Clip(a.clip)
    cv2.imwrite(str(clip / "camera_check.jpg"), camera_check(clip))
    fullbody_check(clip)
    print(f"-> {clip / 'camera_check.jpg'}, {clip / 'fullbody_check.jpg'}")


if __name__ == "__main__":
    main()
