"""Stage 1c: Depth Pro on a few frames -> floor plane, camera pitch/height,
body distance, torso surface samples, focal length.

    python -m core.depth clip01 [--n 8 | --stride 1]

Depth Pro (Apple) estimates metric depth and focal length from one image.
Default: 8 evenly spaced frames (~35 s per clip on MPS); --stride 1 for every frame.

Camera frame: x right, y down, z forward (OpenCV). Floor normal points up.
camera_pitch_deg > 0 means the camera looks down.

Floor search (DESIGN 4): over the whole image minus the person, pixels whose
local normal is level; the lowest parallel surface is the floor (sofa seats
and table tops lose because they are higher). The clip floor is valid only if
>= 50% of sampled frames find one. Stage 2 uses it for floor *direction* only.
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import cv2
import numpy as np

from core import schema, video
from core.skeleton import J, map_mediapipe

QUARTER = 4               # depth maps are stored and searched at 1/4 resolution
LEVEL_CONE_DEG = 50       # candidate floor normals: within this of camera-up
LEVEL_TOL_DEG = 8         # "level" = within this of the dominant up direction
FLOOR_BAND_M = 0.04
N_TORSO_SAMPLES = 64


def load_depth_pro(device: str | None = None):
    import depth_pro
    import torch
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    ckpt = schema.MODELS / "depth_pro.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} missing; see README (models)")
    cfg = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(ckpt))
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    precision = torch.half if device != "cpu" else torch.float32
    model, transform = depth_pro.create_model_and_transforms(
        config=cfg, device=torch.device(device), precision=precision)
    model.eval()

    def infer(rgb: np.ndarray, f_px: float | None = None):
        import torch

        with torch.no_grad():
            pred = model.infer(transform(rgb), f_px=f_px)
        return pred["depth"].float().cpu().numpy(), float(pred["focallength_px"])

    return infer


def backproject(depth: np.ndarray, f: float, cx: float, cy: float) -> np.ndarray:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    return np.stack([(u - cx) / f * depth, (v - cy) / f * depth, depth], -1)


def normals(points: np.ndarray) -> np.ndarray:
    """Per-pixel unit normals from central differences, oriented camera-up (-y)."""
    n = np.full_like(points, np.nan)
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    c = np.cross(dx, dy)
    c /= np.maximum(np.linalg.norm(c, axis=-1, keepdims=True), 1e-9)
    c *= np.where(c[..., 1:2] > 0, -1, 1)
    n[1:-1, 1:-1] = c
    return n


def find_floor(points: np.ndarray, person: np.ndarray, min_pixels: int = 150) -> dict | None:
    """Lowest level surface among non-person pixels. Returns plane n.p + d = 0 (n up)."""
    nrm = normals(points)
    ok = ~person & np.isfinite(nrm).all(-1) & (points[..., 2] > 0.1) & (points[..., 2] < 30)
    up_cam = np.array([0.0, -1.0, 0.0])
    cand = ok & (nrm @ up_cam > np.cos(np.radians(LEVEL_CONE_DEG)))
    if cand.sum() < min_pixels:
        return None
    nc, pc = nrm[cand], points[cand]
    m = up_cam
    for _ in range(8):  # mean-shift to the dominant level direction
        sel = nc @ m > np.cos(np.radians(15))
        if sel.sum() < min_pixels:
            return None
        m = nc[sel].mean(0)
        m /= np.linalg.norm(m)
    level = nc @ m > np.cos(np.radians(LEVEL_TOL_DEG))
    if level.sum() < min_pixels:
        return None
    pl = pc[level]
    below = -(pl @ m)  # distance below the camera along up
    bins = np.arange(below.min(), below.max() + 0.02, 0.02)
    if len(bins) < 2:
        bins = np.array([below.min() - 0.01, below.max() + 0.01])
    hist, edges = np.histogram(below, bins)
    support = max(min_pixels // 3, int(0.03 * len(pl)))
    strong = np.nonzero(hist >= support)[0]
    if not len(strong):
        return None
    h0 = 0.5 * (edges[strong[-1]] + edges[strong[-1] + 1])  # lowest strong surface
    inl = pl[np.abs(below - h0) < FLOOR_BAND_M]
    if len(inl) < min_pixels:
        return None
    c = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    n = vt[-1] * (1 if vt[-1] @ m > 0 else -1)
    d = -float(n @ c)
    return dict(normal=n, d=d, n_inliers=len(inl), pitch_deg=float(-np.degrees(np.arcsin(n[2]))))


def plane_basis(n_cam: np.ndarray) -> np.ndarray:
    """Rotation camera -> Z-up world (X right, Y forward along the camera's heading)."""
    n = n_cam / np.linalg.norm(n_cam)
    fwd = np.array([0.0, 0.0, 1.0]) - n[2] * n
    if np.linalg.norm(fwd) < 1e-6:  # camera looking straight down
        fwd = np.array([0.0, -1.0, 0.0]) - n[1] * n
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, n)
    return np.stack([x, fwd, n])


def torso_samples(points_full_res_fn, uv20: np.ndarray, person: np.ndarray, n: int, rng) -> np.ndarray:
    """Back-projected person pixels inside the shoulder-hip quad, [n,3] NaN-padded."""
    out = np.full((n, 3), np.nan)
    quad = uv20[[J["l_shoulder"], J["r_shoulder"], J["r_hip"], J["l_hip"]]]
    if not np.isfinite(quad).all():
        return out
    m = np.zeros(person.shape, np.uint8)
    cv2.fillConvexPoly(m, np.round(quad).astype(np.int32), 1)
    vs, us = np.nonzero((m > 0) & person)
    if not len(us):
        return out
    pick = rng.choice(len(us), size=min(n, len(us)), replace=False)
    out[: len(pick)] = points_full_res_fn(us[pick], vs[pick])
    return out


def run(clip: schema.Clip, n: int = 8, stride: int | None = None, device: str | None = None) -> dict:
    cap = schema.load(clip.capture)
    W, H, T = int(cap["width"]), int(cap["height"]), int(cap["n_frames"])
    uv20 = map_mediapipe(cap["landmarks_2d"])
    frames = np.arange(0, T, stride) if stride else video.evenly_spaced(T, n)
    infer = load_depth_pro(device)
    masker = video.PersonMasker()
    rng = np.random.default_rng(0)
    cx, cy = (W - 1) / 2, (H - 1) / 2

    rows = dict(floor_normal_cam=[], floor_d=[], camera_pitch_deg=[], floor_valid=[],
                torso_points_cam=[], pelvis_distance_m=[], focal_px=[], depth_maps=[])
    t0 = time.time()
    for i, rgb in video.frames(clip.video, frames):
        depth, f = infer(rgb)
        person = masker(rgb)
        small = cv2.resize(depth, (W // QUARTER, H // QUARTER), interpolation=cv2.INTER_AREA)
        psmall = cv2.resize(person.astype(np.uint8), small.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
        pts = backproject(small, f / QUARTER, cx / QUARTER, cy / QUARTER)
        fl = find_floor(pts, psmall)
        rows["floor_valid"].append(fl is not None)
        rows["floor_normal_cam"].append(fl["normal"] if fl else np.full(3, np.nan))
        rows["floor_d"].append(fl["d"] if fl else np.nan)
        rows["camera_pitch_deg"].append(fl["pitch_deg"] if fl else np.nan)

        def full_pts(us, vs, depth=depth, f=f):
            z = depth[vs, us]
            return np.stack([(us - cx) / f * z, (vs - cy) / f * z, z], -1)

        rows["torso_points_cam"].append(torso_samples(full_pts, uv20[i], person, N_TORSO_SAMPLES, rng))
        rows["pelvis_distance_m"].append(_body_distance(depth, person, uv20[i]))
        rows["focal_px"].append(f)
        rows["depth_maps"].append(small.astype(np.float16))
        print(f"  frame {i}: f={f:.0f}px floor={'yes' if fl else 'no'}"
              + (f" pitch={fl['pitch_deg']:+.1f} h={fl['d']:.2f}m" if fl else ""))
    masker.close()
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["frames"] = frames[: len(out["focal_px"])]
    valid = out["floor_valid"]
    out["floor_found"] = bool(valid.mean() >= 0.5) if len(valid) else False
    if valid.any():
        nn = out["floor_normal_cam"][valid].mean(0)
        out["floor_normal_clip"] = nn / np.linalg.norm(nn)
        out["floor_d_clip"] = float(np.median(out["floor_d"][valid]))
    else:
        out["floor_normal_clip"] = np.full(3, np.nan)
        out["floor_d_clip"] = np.nan
    out.update(width=W, height=H, quarter=QUARTER)
    print(f"{clip.name}: {len(out['frames'])} frames in {time.time() - t0:.0f}s, "
          f"floor in {valid.sum()}/{len(valid)} -> {'valid' if out['floor_found'] else 'NOT valid'}")
    return out


def _body_distance(depth: np.ndarray, person: np.ndarray, uv20: np.ndarray) -> float:
    """Median person-surface depth near the pelvis pixel (falls back to the neck)."""
    for j in ("pelvis", "neck"):
        u, v = uv20[J[j]]
        if not np.isfinite([u, v]).all():
            continue
        h, w = depth.shape
        r = max(6, int(0.03 * np.hypot(h, w)))
        u0, v0 = int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))
        sl = (slice(max(0, v0 - r), v0 + r + 1), slice(max(0, u0 - r), u0 + r + 1))
        patch, pm = depth[sl], person[sl]
        if pm.sum() > 10:
            return float(np.median(patch[pm]))
    return float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--n", type=int, default=8, help="evenly spaced frames (default 8)")
    ap.add_argument("--stride", type=int, help="every Nth frame instead (1 = per-frame)")
    ap.add_argument("--device")
    a = ap.parse_args(argv)
    clip = schema.Clip(a.clip)
    out = run(clip, a.n, a.stride, a.device)
    schema.save(clip.depth, schema.load(clip.capture)["fps"], [], **out)


if __name__ == "__main__":
    main()
