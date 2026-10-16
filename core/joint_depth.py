"""Stage 1d: Sapiens2 on every 4th frame -> nearest-surface depth at each landmark.

    python -m core.joint_depth clip01 [--stride 4]

For each MediaPipe landmark, read the nearest surface in a small patch around
its pixel, using only pixels on the person. That's skin, not the joint centre;
if a body part covers the joint it's that part, which is why 2c treats some
readings as "at least this far" only. Sapiens2 is trained on people, so its
depth off the body isn't trusted: a patch with no person pixels (a cup or a
newspaper in front of the hand) has no Sapiens2 reading. On 1c's frames those
joints get Depth Pro's reading instead (z_fallback), which 2c counts at half
weight. Every 2nd frame gives the same 2c result as every 4th, at twice the time.

Sapiens2 (Meta, facebook/sapiens2-pointmap-0.4b via transformers) predicts a
per-pixel 3D point map. Only its Z channel is used, and its own metric scale
output is ignored: one scale factor per clip comes from Depth Pro on stage 1c's
frames (median body depth vs Sapiens2's median on the same frames). Focal
length also comes from 1c, so run core.depth first. (~5.5 s/frame on M1 MPS, fp16.)

Frames where MediaPipe found nobody are skipped, and 1c's frames are always
included. A re-run keeps the frames of the earlier file (same 1c frames, so the
same scale) and computes only the new ones. Leaves depth.npz alone.

depth_joints.npz:
    frames       [N]      video frame indices
    z_surface    [N,33]   metres along the camera axis (NaN off-screen / off the person)
    z_fallback   [N,33]   Depth Pro's reading where z_surface is NaN, on 1c's frames only
    on_mask      [N,33]   landmark pixel lies on the person mask
    in_frame     [N,33]
    focal_px     [N]      Depth Pro's clip focal from 1c
    scale        []       Depth Pro / Sapiens2 depth ratio for this clip
    scale_frames [M]      1c frames the scale came from
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from core import schema, video
from core.depth import QUARTER, detected_frames

SAPIENS2 = "facebook/sapiens2-pointmap-0.4b"
PATCH_FRAC = 0.008   # patch radius as a fraction of the image diagonal
NEAR_PCT = 10        # "nearest" = 10th percentile, robust to single bad pixels


def load_sapiens2(device: str | None = None):
    import torch
    from transformers import AutoImageProcessor, AutoModelForPointmapEstimation

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.half if device != "cpu" else torch.float32
    cache = schema.MODELS / "hf"
    # do_pad: the whole frame, letterboxed. Resizing to the native 1024x768
    # stretches the image and throws off the depth.
    processor = AutoImageProcessor.from_pretrained(SAPIENS2, cache_dir=cache, do_pad=True)
    model = AutoModelForPointmapEstimation.from_pretrained(SAPIENS2, cache_dir=cache, dtype=dtype)
    model.to(device).eval()

    def infer(rgb: np.ndarray) -> np.ndarray:
        """Unscaled Z [H,W] for one RGB frame."""
        inputs = processor(images=rgb, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            out = model(**inputs)
        out.scales = None   # canonical space; the clip scale comes from Depth Pro
        pm = processor.post_process_pointmap_estimation(out, source_sizes=[rgb.shape[:2]])[0]["pointmap"]
        return pm[2].float().cpu().numpy()

    return infer


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
        if pm.any():
            z[k] = np.percentile(patch[pm], NEAR_PCT)
    return z, on, inside


def clip_scale(ref_maps: np.ndarray, s2_maps: np.ndarray, masks: np.ndarray) -> float:
    """One Depth Pro / Sapiens2 factor per clip: the median over frames of each
    frame's body median ratio. Frames with too few person pixels are skipped."""
    ratios = []
    for ref, s2, m in zip(ref_maps, s2_maps, masks):
        ok = m & np.isfinite(ref) & np.isfinite(s2) & (s2 > 0)
        if ok.sum() >= 20:
            ratios.append(np.median(ref[ok]) / np.median(s2[ok]))
    if not ratios:
        raise RuntimeError("no 1c frame with enough person pixels to scale Sapiens2")
    return float(np.median(ratios))


def depth_pro_map(clip: schema.Clip, dp: dict, k: int, size: tuple[int, int]) -> np.ndarray:
    """Depth Pro depth for 1c frame k at full resolution: the cached frame if there
    is one, else the 1/4-res map from depth.npz scaled up."""
    path = clip / "cache" / "depth_pro" / f"f{int(dp['frames'][k]):06d}.npz"
    if path.exists():
        with np.load(path) as f:
            return f["depth"].astype(np.float32)
    return cv2.resize(dp["depth_maps"][k].astype(np.float32), size, interpolation=cv2.INTER_NEAREST)


def fallback_depth(depth: np.ndarray, uv: np.ndarray, need: np.ndarray) -> np.ndarray:
    """Nearest surface in each needed landmark's patch, any pixel (Depth Pro)."""
    everywhere = np.ones(depth.shape, bool)
    z, _, _ = surface_depth(depth, everywhere, np.where(need[:, None], uv, np.nan))
    return z


def _previous(clip: schema.Clip, dp: dict) -> dict | None:
    """Earlier depth_joints.npz if it was scaled on the same 1c frames."""
    if not clip.depth_joints.exists():
        return None
    old = schema.load(clip.depth_joints)
    if "scale_frames" not in old or not np.array_equal(old["scale_frames"], dp["frames"]):
        return None
    return old


def run(clip: schema.Clip, stride: int = 4, device: str | None = None) -> dict:
    cap = schema.load(clip.capture)
    if not clip.depth.exists():
        raise FileNotFoundError(f"{clip.depth} missing; run core.depth (1c) first for scale and focal")
    dp = schema.load(clip.depth)
    T = int(cap["n_frames"])
    W, H = int(cap["width"]), int(cap["height"])
    small = (W // QUARTER, H // QUARTER)
    focal = float(np.nanmedian(dp["focal_px"]))
    dp_frames = dp["frames"].astype(int)
    frames = np.union1d(np.intersect1d(np.arange(0, T, stride), detected_frames(cap)), dp_frames)
    old = _previous(clip, dp)
    model, masker = [], []

    def infer(rgb):
        if not model:
            model.append(load_sapiens2(device))
        return model[0](rgb)

    def mask(rgb):
        if not masker:
            masker.append(video.PersonMasker())
        return masker[0](rgb)

    rows = {}
    if old is not None:
        scale = float(old["scale"])
        for k, i in enumerate(old["frames"]):
            rows[int(i)] = (old["z_surface"][k], old["on_mask"][k], old["in_frame"][k])
        print(f"  reusing {len(rows)} frames and scale {scale:.3f} from the earlier run")
    else:
        # scale from 1c's frames (older depth.npz files have no person masks; recompute those)
        s2_ref, masks = [], []
        for k, (i, rgb) in enumerate(video.frames(clip.video, dp_frames)):
            s2_ref.append(cv2.resize(infer(rgb), small, interpolation=cv2.INTER_AREA))
            if "person_masks" in dp:
                masks.append(dp["person_masks"][k])
            else:
                masks.append(cv2.resize(mask(rgb).astype(np.uint8), small, interpolation=cv2.INTER_NEAREST) > 0)
        scale = clip_scale(dp["depth_maps"].astype(np.float32), np.asarray(s2_ref), np.asarray(masks))
        print(f"  scale {scale:.3f} from {len(s2_ref)} 1c frames, focal {focal:.0f}px")

    todo = [int(i) for i in frames if int(i) not in rows]
    t0 = time.time()
    for n, (i, rgb) in enumerate(video.frames(clip.video, todo), 1):
        z, on, inside = surface_depth(scale * infer(rgb), mask(rgb), cap["landmarks_2d"][i])
        rows[i] = (z, on, inside)
        if n % 20 == 0:
            print(f"  {n}/{len(todo)} new frames, {time.time() - t0:.0f}s")
    if masker:
        masker[0].close()

    frames = np.array([i for i in frames if i in rows])
    out = dict(frames=frames,
               z_surface=np.stack([rows[i][0] for i in frames]),
               on_mask=np.stack([rows[i][1] for i in frames]),
               in_frame=np.stack([rows[i][2] for i in frames]))
    out["z_fallback"] = np.full_like(out["z_surface"], np.nan)
    for k, i in enumerate(dp_frames):
        r = np.searchsorted(frames, i)
        if r < len(frames) and frames[r] == i:
            need = out["in_frame"][r] & ~np.isfinite(out["z_surface"][r])
            if need.any():
                out["z_fallback"][r] = fallback_depth(depth_pro_map(clip, dp, k, (W, H)),
                                                      cap["landmarks_2d"][i], need)
    out["focal_px"] = np.full(len(frames), focal)
    out["scale"] = scale
    out["scale_frames"] = dp_frames
    print(f"{clip.name}: {len(frames)} frames ({len(todo)} new) in {time.time() - t0:.0f}s, "
          f"{np.isfinite(out['z_fallback']).sum()} Depth Pro fallback readings")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--device")
    a = ap.parse_args(argv)
    clip = schema.Clip(a.clip)
    out = run(clip, a.stride, a.device)
    schema.save(clip.depth_joints, schema.load(clip.capture)["fps"], [f"mp{k}" for k in range(33)], **out)


if __name__ == "__main__":
    main()
