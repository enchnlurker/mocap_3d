"""Export one Depth Pro frame as a coloured point cloud (PLY) in the clip's world
frame, lined up with the solved camera, to check where a seat or a bed really was.

    python -m core.depth_surface walk01 [--frame 90] [--max-dist 12]

Uses depth.npz (1/4-res maps on 1c's frames; the nearest 1c frame to --frame).
2c's camera space has its own scale, so the depth map is scaled to it: the ratio
of 2c's torso joint depths to the map's depths at their pixels. Then
world = camera point @ floor_R^T + fit2d_camera_pos[t], the same transform
core.blender_view uses for the camera. Import the PLY in Blender
(File > Import > Stanford PLY) or open it with core.blender_view --ply.

Writes data/<clip>/depth_surface_f<frame>.ply
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from core import schema, video
from core.depth import backproject
from core.skeleton import J

TORSO = [J[n] for n in ("pelvis", "neck", "l_shoulder", "r_shoulder", "l_hip", "r_hip")]


def nearest_depth_frame(depth: dict, frame: int | None) -> int:
    f = depth["frames"]
    return len(f) // 2 if frame is None else int(np.argmin(np.abs(f - frame)))


def fit_scale(clean: dict, depth_small: np.ndarray, quarter: int, t: int) -> float:
    """2c camera-space depth / Depth Pro depth, from the torso joints' pixels."""
    P = clean["positions_fit2d_cam"][t]
    W, H = int(clean["width"]), int(clean["height"])
    f = float(clean["fit2d_focal_px"])
    uv = f * P[:, :2] / P[:, 2:3] + np.array([(W - 1) / 2, (H - 1) / 2])
    h, w = depth_small.shape
    ratios = []
    for j in TORSO:
        u, v = (uv[j] / quarter).astype(int)
        if 0 <= u < w and 0 <= v < h and depth_small[v, u] > 0:
            ratios.append(P[j, 2] / depth_small[v, u])
    return float(np.median(ratios)) if ratios else 1.0


def points_world(clip: schema.Clip, frame: int | None = None, max_dist: float = 12.0, colors: bool = True):
    """World points [N,3] (and RGB [N,3] uint8) of the 1c frame nearest `frame`."""
    clean = schema.load(clip.clean)
    if "fit2d_camera_pos" not in clean:
        raise SystemExit(f"{clip.name}: run core.fit2d first")
    depth = schema.load(clip.depth)
    k = nearest_depth_frame(depth, frame)
    t = int(depth["frames"][k])
    q = int(depth["quarter"])
    dm = depth["depth_maps"][k].astype(np.float32)
    W, H = int(clean["width"]), int(clean["height"])
    f = float(clean["fit2d_focal_px"])
    s = fit_scale(clean, dm, q, t)
    pts = backproject(dm * s, f / q, (W - 1) / 2 / q, (H - 1) / 2 / q).reshape(-1, 3)
    ok = np.isfinite(pts).all(1) & (pts[:, 2] > 0.1) & (pts[:, 2] < max_dist * s)
    R = np.asarray(clean["floor_R"], float)
    world = pts @ R.T + clean["fit2d_camera_pos"][t]
    rgb = None
    if colors:
        _, img = next(video.frames(clip.video, [t]))
        rgb = cv2.resize(img, dm.shape[::-1], interpolation=cv2.INTER_AREA).reshape(-1, 3)[ok]
    return world[ok], rgb, dict(frame=t, scale=s, person=depth["person_masks"][k].ravel()[ok]
                                if "person_masks" in depth else None)


def write_ply(path, pts: np.ndarray, rgb: np.ndarray | None = None) -> None:
    n = len(pts)
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
            "property float x", "property float y", "property float z"]
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if rgb is not None:
        head += ["property uchar red", "property uchar green", "property uchar blue"]
        dt += [("r", "u1"), ("g", "u1"), ("b", "u1")]
    head.append("end_header")
    arr = np.zeros(n, dt)
    arr["x"], arr["y"], arr["z"] = pts.T
    if rgb is not None:
        arr["r"], arr["g"], arr["b"] = rgb.T
    with open(path, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode())
        fh.write(arr.tobytes())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--frame", type=int, help="video frame (nearest 1c frame is used; default: middle)")
    ap.add_argument("--max-dist", type=float, default=12.0, help="drop points farther than this (m)")
    a = ap.parse_args(argv)
    clip = schema.Clip(a.clip)
    pts, rgb, info = points_world(clip, a.frame, a.max_dist)
    path = clip / f"depth_surface_f{info['frame']}.ply"
    write_ply(path, pts, rgb)
    print(f"{clip.name}: frame {info['frame']}, {len(pts)} points, depth x{info['scale']:.3f} -> {path}")


if __name__ == "__main__":
    main()
