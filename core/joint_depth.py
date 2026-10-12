"""Stage 1d: Depth Pro on every 2nd frame -> nearest-surface depth at each landmark.

    python -m core.joint_depth clip01 [--stride 2]

For each MediaPipe landmark, read the nearest surface in a small patch around
its pixel, preferring pixels on the person. That's skin, not the joint centre;
if something covers the joint it's the covering object, which is why 2c treats
some readings as "at least this far" only. Leaves depth.npz alone.
(~4.5 min per clip on MPS.)

depth_joints.npz:
    frames     [N]      video frame indices
    z_surface  [N,33]   metres along the camera axis (NaN off-screen)
    on_mask    [N,33]   landmark pixel lies on the person mask
    in_frame   [N,33]
    focal_px   [N]
    depth_maps [N,h/4,w/4] float16
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from core import schema, video
from core.depth import QUARTER, load_depth_pro

PATCH_FRAC = 0.008   # patch radius as a fraction of the image diagonal
NEAR_PCT = 10        # "nearest" = 10th percentile, robust to single bad pixels


def surface_depth(depth: np.ndarray, person: np.ndarray, uv: np.ndarray):
    """z_surface, on_mask, in_frame for [K,2] pixel positions."""
    h, w = depth.shape
    r = max(3, int(round(PATCH_FRAC * np.hypot(h, w))))
    K = len(uv)
    z = np.full(K, np.nan, np.float32)
    on = np.zeros(K, bool)
    inside = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    for k in np.nonzero(inside)[0]:
        u, v = int(uv[k, 0]), int(uv[k, 1])
        sl = (slice(max(0, v - r), v + r + 1), slice(max(0, u - r), u + r + 1))
        patch, pm = depth[sl], person[sl]
        on[k] = bool(person[v, u])
        vals = patch[pm] if pm.any() else patch.ravel()
        z[k] = np.percentile(vals, NEAR_PCT)
    return z, on, inside


def run(clip: schema.Clip, stride: int = 2, device: str | None = None) -> dict:
    cap = schema.load(clip.capture)
    T = int(cap["n_frames"])
    W, H = int(cap["width"]), int(cap["height"])
    frames = np.arange(0, T, stride)
    infer = load_depth_pro(device)
    masker = video.PersonMasker()
    rows = dict(z_surface=[], on_mask=[], in_frame=[], focal_px=[], depth_maps=[], frames=[])
    t0 = time.time()
    for i, rgb in video.frames(clip.video, frames):
        depth, f = infer(rgb)
        person = masker(rgb)
        z, on, inside = surface_depth(depth, person, cap["landmarks_2d"][i])
        rows["z_surface"].append(z)
        rows["on_mask"].append(on)
        rows["in_frame"].append(inside)
        rows["focal_px"].append(f)
        rows["frames"].append(i)
        rows["depth_maps"].append(cv2.resize(depth, (W // QUARTER, H // QUARTER),
                                             interpolation=cv2.INTER_AREA).astype(np.float16))
        k = len(rows["frames"])
        if k % 20 == 0:
            print(f"  {k}/{len(frames)} frames, {time.time() - t0:.0f}s")
    masker.close()
    out = {k: np.asarray(v) for k, v in rows.items()}
    print(f"{clip.name}: {len(out['frames'])} frames in {time.time() - t0:.0f}s")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--device")
    a = ap.parse_args(argv)
    clip = schema.Clip(a.clip)
    out = run(clip, a.stride, a.device)
    schema.save(clip.depth_joints, schema.load(clip.capture)["fps"], [f"mp{k}" for k in range(33)], **out)


if __name__ == "__main__":
    main()
