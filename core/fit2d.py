"""Stage 2c: fit to 2D. Reads clean + 1a + 1c + 1d + video, adds to the clean file:
    positions_fit2d [T,20,3], bone_lengths_fit2d [19], fit2d_camera_pos [T,3], fit2d_focal_px
and writes fit2d_depth_summary.json. Re-run after clean.py.

    python -m core.fit2d clip01 [--camera auto|static|moving]

MediaPipe's 2D keypoints are accurate; its 3D is not. This turns the 2D into
3D with a solver that uses bone lengths, Depth Pro depth and the 2b pose as
supporting evidence.

Unknowns, solved for all frames at once: pelvis position in camera space per
frame, one direction per bone per frame, one length per bone group per clip
(left = right). Joints are built from the pelvis outward (core.skeleton), so
shoulder/hip/spine pairs are tied by construction.

Camera: pinhole at the origin, Depth Pro's focal length, principal point at
the image centre: pixel = focal * (x, y) / z + centre.

Cost terms (soft-L1 each, so one bad reading can't drag the whole body):
    2D            projected joints on MediaPipe 2D        ~7 px; hidden joints count 1/3
    depth visible joint depth relative to the frame's     ~3 cm; relative, so Depth Pro's
                  other visible joints matches the map    absolute scale doesn't matter
    depth covered behind a hand / limb / the lap / hidden one-sided: at least that far
    depth anchor  fit-vs-map gap constant over the clip   stops drift toward/away
    prior         bone directions near 2b's               weak captured, strong inferred
    lean          torso toward/away tilt near 2b's        ~6 deg
    facing        hip/shoulder axes near 2b's             ~9 deg
    lengths       each group length near 2b's             ~12 % slack
    smoothness    low joint acceleration; pelvis camera distance even more so
    knee          knees don't bend forward                only when legs are captured
    legs apart    knees/ankles keep >= 5 cm apart along   only when both legs are captured
                  the hip axis (legs don't cross)
    knee sideways shin stays within 15 deg of the thigh's  legs captured
                  bending plane
    hip tilt      hip line within 15 deg of square to      one-sided
                  the spine
    feet under    upright: ankle within 12 cm of its hip,  fades out as the trunk tilts
                  sideways                                 past 20-45 deg
    twist         hips vs shoulders turn <= 30 deg        one-sided
    turn smooth   hip/shoulder axes turn smoothly         depth noise can't spin the body

Solving: (1) rigid placement of the 2b pose, one rotation + translation per
frame (also the "before" number in reports); (2) free fit of directions,
pelvis and lengths from the rotated 2b directions. L-BFGS, restarted until
the loss stops dropping.

Output placement: static camera (background < 0.5 px/frame) -> one offset for
the whole clip moves the solved body into the floor frame; moving camera ->
the solved pose rides 2b's hip track each frame. Clips with feet get one more
constant ankle offset. The camera position is saved so Blender can overlay
the footage.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import minimize

from core import proportions, schema
from core.filters import fill_gaps, hold_edges
from core.skeleton import (BONE_PAIRS, D, DIR_FROM, DIR_TO, GROUPS, J, LEFT_RIGHT, LEGS, N_JOINTS,
                           bone_lengths_from_groups, build_positions, extract_dirs,
                           group_lengths, map_mediapipe, normalize)

SIGMA_2D_PX = 7.0
HIDDEN_2D_WEIGHT = 1 / 3
SIGMA_DEPTH_M = 0.03
SIGMA_ANCHOR_M = 0.05
SIGMA_PRIOR_CAPTURED = 1 / 0.5  # radians-ish on unit vectors: weight 0.5
SIGMA_PRIOR_NO_DEPTH = 1 / 4.0  # without depth the prior is the only depth evidence
SIGMA_PRIOR_INFERRED = 0.05
SIGMA_LEAN = 0.1
SIGMA_FACING = 0.15              # hip/shoulder axes near 2b's (~9 deg): which way the body faces
LENGTH_SLACK = 0.12
SIGMA_ACC_M = 0.015              # per frame^2 at 30 fps
PELVIS_Z_ACC_FACTOR = 0.25
SIGMA_KNEE = 0.05
TWIST_MAX_DEG = 30.0            # hips vs shoulders about the spine
SIGMA_TWIST_DEG = 5.0
SIGMA_AXIS_ACC = 0.01            # hip/shoulder axis turn acceleration, per frame^2 at 30 fps
LEG_MIN_SEPARATION_M = 0.05      # knees/ankles stay this far apart along the hip axis
SIGMA_LEG_CROSS_M = 0.02
KNEE_SIDEWAYS_MAX_DEG = 15.0     # shin out of the thigh's bending plane (hip rotation slack)
HIP_TILT_MAX_DEG = 15.0          # hip line vs perpendicular to the spine
SIGMA_ANATOMY_DEG = 5.0
FOOT_UNDER_HIP_M = 0.12          # upright: ankle within this of its hip, sideways
SIGMA_FOOT_UNDER_M = 0.03
UPRIGHT_DEG = (20.0, 45.0)       # spine from vertical: full weight below, none above
ASIN_MAX = 1 - 1e-6              # asin has an infinite slope at 1
STATIC_PX_PER_FRAME = 0.5
COVER_MARGIN_M = 0.05
LIMB_RADIUS_M = 0.06

T_ = torch.float64


@dataclass
class FitInputs:
    uv: np.ndarray            # [T,20,2] px, NaN where missing
    w2d: np.ndarray           # [T,20] 1 visible, 1/3 occluded, 0 none
    focal: float
    center: np.ndarray        # [2]
    pos_2b: np.ndarray        # [T,20,3] world
    floor_R: np.ndarray       # camera -> world
    inferred: np.ndarray      # [T,20]
    tracked: np.ndarray       # [20]
    fps: float
    init_dist: np.ndarray     # [T] initial pelvis distance along the camera axis
    depth_frames: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    z_surface: np.ndarray = field(default_factory=lambda: np.zeros((0, N_JOINTS)))
    on_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, N_JOINTS), bool))
    in_frame: np.ndarray = field(default_factory=lambda: np.zeros((0, N_JOINTS), bool))
    occluded: np.ndarray | None = None
    patch_px: float = 12.0


# ---------------------------------------------------------------- torch pieces

def soft_l1(sq):
    return 2.0 * (torch.sqrt(1.0 + sq) - 1.0)


def rotvec_to_matrix(r):
    theta = torch.sqrt((r * r).sum(-1, keepdim=True) + 1e-18)
    k = r / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = torch.zeros_like(kx)
    K = torch.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], -1).reshape(*r.shape[:-1], 3, 3)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    eye = torch.eye(3, dtype=r.dtype).expand_as(K)
    return eye + s * K + (1 - c) * (K @ K)


def matrix_to_rotvec(R):
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_rotvec()


class Problem:
    """Shared data and cost terms for both phases."""

    def __init__(self, x: FitInputs):
        self.x = x
        self.T = len(x.uv)
        t = lambda a: torch.as_tensor(np.nan_to_num(a), dtype=T_)
        self.uv = t(x.uv)
        self.w2d = t(x.w2d * np.isfinite(x.uv).all(-1))
        self.f = float(x.focal)
        self.c = t(x.center)
        acc = SIGMA_ACC_M * (30.0 / x.fps) ** 2
        self.sigma_acc = acc
        self.df = torch.as_tensor(x.depth_frames, dtype=torch.long)
        self.zs = t(x.z_surface)
        self.vis_d = torch.zeros(len(x.depth_frames), N_JOINTS, dtype=T_)
        self.cov_d = torch.zeros_like(self.vis_d)

    def set_depth_masks(self, vis_d: np.ndarray, cov_d: np.ndarray):
        ok = vis_d.sum(1) >= 3  # need a few visible joints for a relative reading
        self.vis_d = torch.as_tensor(vis_d & ok[:, None], dtype=T_)
        self.cov_d = torch.as_tensor(cov_d & ok[:, None], dtype=T_)

    def project(self, P):
        z = P[..., 2].clamp(min=0.05)
        return self.f * P[..., :2] / z[..., None] + self.c

    def terms(self, P):
        """2D, depth (visible, covered, anchor) and smoothness on camera joints [T,20,3]."""
        out = {}
        r = ((self.project(P) - self.uv) / SIGMA_2D_PX).pow(2).sum(-1)
        out["2d"] = (self.w2d * soft_l1(r)).sum()
        if len(self.df) and self.vis_d.sum() > 0:
            zf = P[self.df, :, 2]
            nv = self.vis_d.sum(1, keepdim=True).clamp(min=1)
            mf = (zf * self.vis_d).sum(1, keepdim=True) / nv
            ms = (self.zs * self.vis_d).sum(1, keepdim=True) / nv
            rel_f, rel_s = zf - mf, self.zs - ms
            out["depth"] = (self.vis_d * soft_l1(((rel_f - rel_s) / SIGMA_DEPTH_M) ** 2)).sum()
            out["depth_cov"] = (self.cov_d * soft_l1((torch.relu(rel_s - rel_f) / SIGMA_DEPTH_M) ** 2)).sum()
            has = self.vis_d.sum(1) > 0
            gap = (mf - ms)[:, 0][has]
            out["anchor"] = soft_l1(((gap - gap.mean()) / SIGMA_ANCHOR_M) ** 2).sum()
        a = P[2:] - 2 * P[1:-1] + P[:-2]
        out["smooth"] = soft_l1((a / self.sigma_acc).pow(2).sum(-1)).sum()
        pz = a[:, J["pelvis"], 2] / (self.sigma_acc * PELVIS_Z_ACC_FACTOR)
        out["smooth"] = out["smooth"] + soft_l1(pz ** 2).sum()
        return out


def _lbfgs(fun, x0: np.ndarray, label: str, max_rounds: int = 8, tol: float = 1e-4):
    """scipy L-BFGS with torch gradients, restarted until the loss stops dropping."""
    x = x0.copy()
    last = None
    for k in range(max_rounds):
        res = minimize(fun, x, jac=True, method="L-BFGS-B", options=dict(maxiter=1500, maxcor=30))
        if not np.isfinite(res.fun) or not np.isfinite(res.x).all():
            print(f"    {label}: round {k + 1} went non-finite; keeping the last good result")
            break
        x = res.x
        if last is not None and (last - res.fun) <= tol * abs(last):
            last = res.fun
            break
        last = res.fun
    print(f"    {label}: loss {last:.1f} after {k + 1} round(s)" if last is not None else f"    {label}: no finite result")
    return x


def _wrap(loss_fn, shapes):
    sizes = [int(np.prod(s)) for s in shapes]

    def fun(x):
        parts, i = [], 0
        for s, n in zip(shapes, sizes):
            parts.append(torch.tensor(x[i:i + n].reshape(s), dtype=T_, requires_grad=True))
            i += n
        loss = loss_fn(*parts)
        loss.backward()
        g = np.concatenate([p.grad.numpy().ravel() for p in parts])
        return float(loss.detach()), g

    def unpack(x):
        out, i = [], 0
        for s, n in zip(shapes, sizes):
            out.append(x[i:i + n].reshape(s))
            i += n
        return out

    return fun, unpack


# ---------------------------------------------------------------- phases

def rigid_phase(pb: Problem, x: FitInputs):
    T = pb.T
    rel = x.pos_2b - x.pos_2b[:, J["pelvis"]][:, None]
    Rwc = x.floor_R.T
    r0 = np.tile(matrix_to_rotvec(Rwc), (T, 1))
    uvp = x.uv[:, J["pelvis"]] if x.tracked[J["pelvis"]] else x.uv[:, J["neck"]]
    ok = np.isfinite(uvp).all(1)
    uvp, _ = fill_gaps(uvp, ok, None)
    uvp = hold_edges(uvp)
    if not np.isfinite(uvp).all():
        uvp = np.tile(x.center, (T, 1))
    z = x.init_dist
    t0 = np.c_[(uvp - x.center) / x.focal * z[:, None], z]
    if not x.tracked[J["pelvis"]]:  # we seeded on the neck pixel
        t0 = t0 - (rel[:, J["neck"]] @ Rwc.T)
    relt = torch.as_tensor(rel, dtype=T_)

    def joints(r, t):
        return torch.einsum("tij,tkj->tki", rotvec_to_matrix(r), relt) + t[:, None]

    def loss(r, t):
        return sum(pb.terms(joints(r, t)).values())

    fun, unpack = _wrap(loss, [(T, 3), (T, 3)])
    xs = _lbfgs(fun, np.concatenate([r0.ravel(), t0.ravel()]), "rigid")
    r, t = unpack(xs)
    with torch.no_grad():
        Rm = rotvec_to_matrix(torch.as_tensor(r, dtype=T_)).numpy()
        P = joints(torch.as_tensor(r, dtype=T_), torch.as_tensor(t, dtype=T_)).numpy()
    return P, Rm


def free_phase(pb: Problem, x: FitInputs, P_rigid: np.ndarray, R_rigid: np.ndarray):
    T = pb.T
    dirs_2b_world = extract_dirs(x.pos_2b)
    dref = np.einsum("tij,tkj->tki", R_rigid, dirs_2b_world)
    glen_2b = group_lengths(x.pos_2b)
    inf_dir = x.inferred[:, DIR_FROM] | x.inferred[:, DIR_TO]
    sig_cap = SIGMA_PRIOR_CAPTURED if pb.vis_d.sum() > 0 else SIGMA_PRIOR_NO_DEPTH
    w_prior = torch.as_tensor(np.where(inf_dir, 1 / SIGMA_PRIOR_INFERRED, 1 / sig_cap), dtype=T_)
    dref_t = torch.as_tensor(dref, dtype=T_)
    lean_ref = torch.asin(dref_t[:, D["spine"], 2].clamp(-ASIN_MAX, ASIN_MAX))
    log_l2b = torch.as_tensor(np.log(glen_2b), dtype=T_)
    knee_sides = [s for s in ("l", "r") if x.tracked[LEGS[s][:3]].all()]
    sigma_axis = SIGMA_AXIS_ACC * (30.0 / x.fps) ** 2
    up_cam = torch.as_tensor(x.floor_R[2], dtype=T_)  # world up in camera coordinates

    def loss(pel, raw, logl):
        d = raw / raw.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        P = build_positions(pel, d, torch.exp(logl))
        terms = pb.terms(P)
        terms["prior"] = soft_l1(((d - dref_t).pow(2).sum(-1)) * w_prior ** 2).sum()
        lean = torch.asin(d[:, D["spine"], 2].clamp(-ASIN_MAX, ASIN_MAX))
        terms["lean"] = soft_l1(((lean - lean_ref) / SIGMA_LEAN) ** 2).sum()
        fa = [D["hip_axis"], D["shoulder_axis"]]
        terms["facing"] = soft_l1((d[:, fa] - dref_t[:, fa]).pow(2).sum(-1) / SIGMA_FACING ** 2).sum()
        dl = logl - log_l2b
        over = torch.relu(dl.abs() - np.log(1 + LENGTH_SLACK)) / 0.01
        terms["lengths"] = T * (soft_l1((dl / LENGTH_SLACK) ** 2) + over ** 2).sum() / len(GROUPS)
        h, sa, sp = d[:, D["hip_axis"]], d[:, D["shoulder_axis"]], d[:, D["spine"]]
        twist = torch.atan2((torch.cross(h, sa, dim=-1) * sp).sum(-1), (h * sa).sum(-1))
        r = torch.relu(twist.abs() - np.radians(TWIST_MAX_DEG)) / np.radians(SIGMA_TWIST_DEG)
        terms["twist"] = soft_l1(r ** 2).sum()
        axes = d[:, [D["hip_axis"], D["shoulder_axis"]]]
        acc = (axes[2:] - 2 * axes[1:-1] + axes[:-2]) / sigma_axis
        terms["turn_smooth"] = soft_l1(acc.pow(2).sum(-1)).sum()
        # basic anatomy, for what neither 2D nor depth settles in a side view
        tilt = torch.asin((h * sp).sum(-1).abs().clamp(max=ASIN_MAX))
        r = torch.relu(tilt - np.radians(HIP_TILT_MAX_DEG)) / np.radians(SIGMA_ANATOMY_DEG)
        terms["hip_tilt"] = soft_l1(r ** 2).sum()
        cos_up = (sp * up_cam).sum(-1)
        lo, hi = np.cos(np.radians(UPRIGHT_DEG[1])), np.cos(np.radians(UPRIGHT_DEG[0]))
        upright = ((cos_up - lo) / (hi - lo)).clamp(0, 1)
        for s in knee_sides:
            th, sh = d[:, D[f"{s}_thigh"]], d[:, D[f"{s}_shin"]]
            # bending plane = the one square to the hip line (hip line made square to the thigh);
            # stays defined standing, sitting (thigh forward) or lying
            n = h - (h * th).sum(-1, keepdim=True) * th
            n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-3)
            side = torch.asin((sh * n).sum(-1).abs().clamp(max=ASIN_MAX))
            r = torch.relu(side - np.radians(KNEE_SIDEWAYS_MAX_DEG)) / np.radians(SIGMA_ANATOMY_DEG)
            terms["knee_sideways"] = terms.get("knee_sideways", 0) + soft_l1(r ** 2).sum()
            sign = 1.0 if s == "l" else -1.0  # outward along the hip axis
            off = sign * ((P[:, J[f"{s}_ankle"]] - P[:, J[f"{s}_hip"]]) * h).sum(-1)
            r = torch.relu(off.abs() - FOOT_UNDER_HIP_M) / SIGMA_FOOT_UNDER_M
            terms["feet_under"] = terms.get("feet_under", 0) + (upright * soft_l1(r ** 2)).sum()
        if len(knee_sides) == 2:
            axis = d[:, D["hip_axis"]]  # points to the body's left
            for n in ("knee", "ankle"):
                sep = ((P[:, J[f"l_{n}"]] - P[:, J[f"r_{n}"]]) * axis).sum(-1)
                r = torch.relu(LEG_MIN_SEPARATION_M - sep) / SIGMA_LEG_CROSS_M
                terms["legs_apart"] = terms.get("legs_apart", 0) + soft_l1(r ** 2).sum()
        for s in knee_sides:
            th, sh = d[:, D[f"{s}_thigh"]], d[:, D[f"{s}_shin"]]
            bend = (d[:, D["hip_axis"]] * torch.cross(th, sh, dim=-1)).sum(-1)
            terms["knee"] = terms.get("knee", 0) + soft_l1((torch.relu(-bend) / SIGMA_KNEE) ** 2).sum()
        return sum(terms.values())

    x0 = np.concatenate([P_rigid[:, J["pelvis"]].ravel(), dref.ravel(), np.log(glen_2b)])
    fun, unpack = _wrap(loss, [(T, 3), (T, len(D), 3), (len(GROUPS),)])
    xs = _lbfgs(fun, x0, "free")
    pel, raw, logl = unpack(xs)
    d = normalize(raw)
    glen = np.exp(logl)
    return build_positions(pel, d, glen), glen


# ---------------------------------------------------------------- evidence prep

def covered_by_body(P: np.ndarray, frames: np.ndarray, focal, center, min_px) -> np.ndarray:
    """[N,20] joints whose pixel is behind another part of the body: a limb
    (bone as a LIMB_RADIUS_M tube, non-adjacent) or the torso (shoulder-hip quad),
    nearer the camera by more than COVER_MARGIN_M. In a side view this catches
    the far hip, far shoulder and far leg, which MediaPipe still calls visible."""
    import cv2

    N = len(frames)
    out = np.zeros((N, N_JOINTS), bool)
    quad = [J["l_shoulder"], J["r_shoulder"], J["r_hip"], J["l_hip"]]
    for k, t in enumerate(frames):
        Pt = P[t]
        uv = focal * Pt[:, :2] / Pt[:, 2:3] + center
        a, b = uv[BONE_PAIRS[:, 0]], uv[BONE_PAIRS[:, 1]]
        za, zb = Pt[BONE_PAIRS[:, 0], 2], Pt[BONE_PAIRS[:, 1], 2]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(1), 1e-9)
        poly = uv[quad].astype(np.float32).reshape(-1, 1, 2)
        z_torso = Pt[quad, 2].mean()
        r_torso = max(min_px, focal * LIMB_RADIUS_M / z_torso)
        for j in range(N_JOINTS):
            s = np.clip(((uv[j] - a) * ab).sum(1) / L2, 0, 1)
            dist = np.linalg.norm(a + s[:, None] * ab - uv[j], axis=1)
            zseg = za + s * (zb - za)
            r_px = np.maximum(min_px, focal * LIMB_RADIUS_M / zseg)
            adj = (BONE_PAIRS == j).any(1)
            limb = np.any(~adj & (dist < r_px) & (zseg < Pt[j, 2] - COVER_MARGIN_M))
            inside = cv2.pointPolygonTest(poly, (float(uv[j, 0]), float(uv[j, 1])), True) > -r_torso
            torso = inside and z_torso < Pt[j, 2] - COVER_MARGIN_M
            out[k, j] = limb or torso
    return out


def far_of_overlapping_pair(x: FitInputs, P_cam: np.ndarray) -> np.ndarray:
    """[N,20]: when a left/right pair overlaps on screen (side view), the depth map
    sees only the near one, so the far one's reading is a lower bound."""
    f = x.depth_frames
    out = np.zeros((len(f), N_JOINTS), bool)
    for l, r in LEFT_RIGHT:
        d2 = np.linalg.norm(x.uv[f, l] - x.uv[f, r], axis=-1)
        r_px = np.maximum(x.patch_px, x.focal * LIMB_RADIUS_M / P_cam[f][:, [l, r], 2].mean(1))
        overlap = d2 < r_px
        l_far = P_cam[f, l, 2] > P_cam[f, r, 2]
        out[:, l] |= overlap & l_far
        out[:, r] |= overlap & ~l_far
    return out


def prepare(clean: dict, depth: dict | None, dj: dict | None) -> FitInputs:
    W, H = int(clean["width"]), int(clean["height"])
    uv = clean["uv_2d"] * np.array([W, H])
    occ = clean["occluded"].astype(bool)
    inferred = clean["inferred"].astype(bool)
    w2d = np.where(occ, HIDDEN_2D_WEIGHT, 1.0) * np.isfinite(uv).all(-1) * ~inferred
    T = len(uv)

    focal = np.nan
    for src in (dj, depth):
        if src is not None and np.isfinite(src["focal_px"]).any():
            focal = float(np.nanmedian(src["focal_px"]))
            break
    if not np.isfinite(focal):
        focal = 1.2 * max(W, H)
        print("    no depth: assuming focal = 1.2 x image width")

    pos_2b = clean["positions_proportional"]
    x = FitInputs(uv=uv, w2d=w2d, focal=focal, center=np.array([(W - 1) / 2, (H - 1) / 2]),
                  pos_2b=pos_2b, floor_R=clean["floor_R"], inferred=inferred,
                  tracked=clean["tracked"].astype(bool), fps=float(clean["fps"]),
                  init_dist=np.zeros(T), occluded=occ,
                  patch_px=max(8.0, 0.012 * np.hypot(W, H)))

    if dj is not None:
        x.depth_frames = dj["frames"].astype(int)
        x.z_surface = map_mediapipe(dj["z_surface"][:, :, None])[..., 0]
        x.on_mask = map_mediapipe(dj["on_mask"][:, :, None].astype(float), "min")[..., 0] > 0.5
        x.in_frame = map_mediapipe(dj["in_frame"][:, :, None].astype(float), "min")[..., 0] > 0.5

    # initial distance: depth readings on the torso > depth.npz body distance > pinhole size
    torso = [J[n] for n in ("pelvis", "neck", "l_shoulder", "r_shoulder", "l_hip", "r_hip")]
    dist = np.full(T, np.nan)
    if len(x.depth_frames):
        zt = np.where(x.on_mask[:, torso], x.z_surface[:, torso], np.nan)
        with np.errstate(all="ignore"):
            dist[x.depth_frames] = np.nanmedian(zt, axis=1)
    if not np.isfinite(dist).any() and depth is not None and np.isfinite(depth["pelvis_distance_m"]).any():
        dist[:] = np.nanmedian(depth["pelvis_distance_m"])
    if not np.isfinite(dist).any():
        tr3 = np.linalg.norm(pos_2b[:, J["neck"]] - pos_2b[:, J["pelvis"]], axis=-1)
        tr2 = np.linalg.norm(uv[:, J["neck"]] - uv[:, J["pelvis"]], axis=-1)
        sh3 = np.linalg.norm(pos_2b[:, J["l_shoulder"]] - pos_2b[:, J["r_shoulder"]], axis=-1)
        sh2 = np.linalg.norm(uv[:, J["l_shoulder"]] - uv[:, J["r_shoulder"]], axis=-1)
        est = np.where(np.isfinite(tr2) & (tr2 > 5), focal * tr3 / tr2, focal * sh3 / sh2)
        dist[:] = np.nanmedian(est) if np.isfinite(est).any() else 3.0
    ok = np.isfinite(dist)
    dist, _ = fill_gaps(dist[:, None], ok, None)
    x.init_dist = hold_edges(dist)[:, 0]
    return x


def depth_masks(x: FitInputs, P_cam: np.ndarray):
    """Visible depth term vs one-sided (covered / hidden) per depth frame."""
    if not len(x.depth_frames):
        e = np.zeros((0, N_JOINTS), bool)
        return e, e
    f = x.depth_frames
    finite = np.isfinite(x.z_surface) & x.in_frame
    cov_seg = covered_by_body(P_cam, f, x.focal, x.center, 0.5 * x.patch_px) | far_of_overlapping_pair(x, P_cam)
    vis = (x.w2d[f] == 1.0) & x.on_mask & finite & ~cov_seg
    cov = finite & ~vis & (x.occluded[f] | ~x.on_mask | cov_seg | x.inferred[f])
    return vis, cov


# ---------------------------------------------------------------- reporting

def _stats(pb: Problem, x: FitInputs, P: np.ndarray, vis_d: np.ndarray) -> dict:
    with torch.no_grad():
        uvp = pb.project(torch.as_tensor(P, dtype=T_)).numpy()
    err = np.linalg.norm(uvp - x.uv, axis=-1)
    vis2d = x.w2d == 1.0
    out = {"px_2d_median_visible": float(np.nanmedian(err[vis2d])) if vis2d.any() else None,
           "px_2d_p90_visible": float(np.nanpercentile(err[vis2d], 90)) if vis2d.any() else None}
    if len(x.depth_frames) and vis_d.any():
        zf = P[x.depth_frames, :, 2]
        m = vis_d
        with np.errstate(all="ignore"):
            mf = np.nansum(np.where(m, zf, 0), 1, keepdims=True) / m.sum(1, keepdims=True)
            ms = np.nansum(np.where(m, x.z_surface, 0), 1, keepdims=True) / m.sum(1, keepdims=True)
        e = np.abs((zf - mf) - (x.z_surface - ms))[m]
        out["rel_depth_cm_median_visible"] = float(np.nanmedian(e) * 100)
    return out


# ---------------------------------------------------------------- placement

def background_motion(video_path, uv: np.ndarray, n_pairs: int = 30) -> float | None:
    """Median background motion in px/frame (person box masked out)."""
    import cv2

    from core import video

    try:
        vi = video.info(video_path)
    except FileNotFoundError:
        return None
    T = vi["n_frames"]
    starts = video.evenly_spaced(max(T - 1, 1), n_pairs)
    want = sorted(set(starts) | set(starts + 1))
    fr = {i: cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) for i, rgb in video.frames(video_path, want)}
    mags = []
    for s in starts:
        if s not in fr or s + 1 not in fr:
            continue
        g0, g1 = fr[s], fr[s + 1]
        mask = np.full(g0.shape, 255, np.uint8)
        pts = uv[min(s, len(uv) - 1)]
        pts = pts[np.isfinite(pts).all(1)]
        if len(pts):
            lo, hi = pts.min(0), pts.max(0)
            pad = 0.2 * (hi - lo) + 20
            x0, y0 = np.maximum(lo - pad, 0).astype(int)
            x1, y1 = (hi + pad).astype(int)
            mask[y0:y1, x0:x1] = 0
        p0 = cv2.goodFeaturesToTrack(g0, 200, 0.01, 8, mask=mask)
        if p0 is None or len(p0) < 8:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None)
        good = st[:, 0] == 1
        if good.sum() >= 8:
            mags.append(np.median(np.linalg.norm((p1 - p0)[good, 0], axis=1)))
    return float(np.median(mags)) if mags else None


def place(P_cam: np.ndarray, x: FitInputs, static: bool, has_feet: bool, glen: np.ndarray):
    R = x.floor_R
    pel_cam = P_cam[:, J["pelvis"]]
    pel_2b = x.pos_2b[:, J["pelvis"]]
    if static:
        offset = np.median(pel_2b - pel_cam @ R.T, axis=0)
        Wd = P_cam @ R.T + offset
        cam = np.tile(offset, (len(P_cam), 1))
    else:
        Wd = (P_cam - pel_cam[:, None]) @ R.T + pel_2b[:, None]
        cam = pel_2b - pel_cam @ R.T
    height = 2 * glen[GROUPS.index("spine")] / 0.288
    if has_feet:
        dz = proportions.ground_offset(Wd, height, ~x.inferred)
    else:  # never below the floor, inferred feet included
        dz = max(proportions.ground_offset(Wd, height), 0.0)
    Wd[..., 2] += dz
    cam[:, 2] += dz
    return Wd, cam


# ---------------------------------------------------------------- entry points

def solve(x: FitInputs, static: bool = True, has_feet: bool = False) -> dict:
    t0 = time.time()
    pb = Problem(x)
    # depth masks need a pose; take them from the 2b pose placed rigidly without depth first
    P_r, R_r = rigid_phase(pb, x)
    vis_d, cov_d = depth_masks(x, P_r)
    if not vis_d.any():
        print("    WARNING: no depth readings; toward/away comes from the 2b prior only (run core.joint_depth)")
    pb.set_depth_masks(vis_d, cov_d)
    if len(x.depth_frames):
        P_r, R_r = rigid_phase(pb, x)
    before = _stats(pb, x, P_r, vis_d)
    P_f, glen = free_phase(pb, x, P_r, R_r)
    if len(x.depth_frames):
        # the rigid 2b pose is too rough to judge who covers whom; re-check on the fit and solve again
        vis_d, cov_d = depth_masks(x, P_f)
        pb.set_depth_masks(vis_d, cov_d)
        P_f, glen = free_phase(pb, x, P_r, R_r)
    after = _stats(pb, x, P_f, vis_d)
    Wd, cam = place(P_f, x, static, has_feet, glen)
    summary = dict(before_rigid_2b=before, after_fit2d=after, seconds=round(time.time() - t0, 1),
                   focal_px=x.focal, camera="static" if static else "moving",
                   n_depth_frames=int(len(x.depth_frames)),
                   n_depth_visible=int(vis_d.sum()), n_depth_covered=int(cov_d.sum()),
                   length_change_pct={g: round(100 * (glen[i] / group_lengths(x.pos_2b)[i] - 1), 1)
                                      for i, g in enumerate(GROUPS)})
    return dict(positions_fit2d=Wd, positions_fit2d_cam=P_f, bone_lengths_fit2d=bone_lengths_from_groups(glen),
                fit2d_camera_pos=cam, fit2d_focal_px=x.focal, summary=summary)


def run(clip: schema.Clip, camera: str = "auto") -> dict:
    clean_path = clip.clean
    clean = schema.load(clean_path)
    depth = schema.load(clip.depth) if clip.depth.exists() else None
    dj = schema.load(clip.depth_joints) if clip.depth_joints.exists() else None
    x = prepare(clean, depth, dj)
    motion = None
    if camera == "auto":
        motion = background_motion(clip.video, x.uv)
        static = motion is None or motion < STATIC_PX_PER_FRAME
        if motion is None:
            print("    video unavailable for camera check; assuming static")
    else:
        static = camera == "static"
    print(f"{clip.name}: fit2d on {len(x.uv)} frames, {len(x.depth_frames)} depth frames, "
          f"camera {'static' if static else 'moving'}"
          + (f" (background {motion:.2f} px/frame)" if motion is not None else ""))
    out = solve(x, static, bool(clean["has_feet"]))
    summary = out.pop("summary")
    summary["background_px_per_frame"] = motion
    summary["clean_file"] = clean_path.name
    schema.update(clean_path, **out)
    (clip / "fit2d_depth_summary.json").write_text(json.dumps(summary, indent=2))
    b, a = summary["before_rigid_2b"], summary["after_fit2d"]
    print(f"  2D median px: {b['px_2d_median_visible']:.1f} -> {a['px_2d_median_visible']:.1f}"
          + (f"   rel depth cm: {b['rel_depth_cm_median_visible']:.1f} -> {a['rel_depth_cm_median_visible']:.1f}"
             if "rel_depth_cm_median_visible" in a else "")
          + f"   ({summary['seconds']} s)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--camera", default="auto", choices=["auto", "static", "moving"])
    a = ap.parse_args(argv)
    run(schema.Clip(a.clip), a.camera)


if __name__ == "__main__":
    main()
