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

Cost terms (soft-L1 unless noted, so one bad reading can't drag the whole body):
    2D            projected joints on MediaPipe 2D        1 % of the shorter side; weight = visibility
    depth visible joint depth relative to the frame's     ~3 cm; relative, so Depth Pro's
                  other visible joints matches the map    absolute scale doesn't matter
    depth visible readings are skin: a per-joint skin depth is added first
    depth covered behind a limb / the torso / the lap, near a  one-sided: at least that far
                  hand (or what it holds), nearer end of an
                  impossible bone, or hidden
    depth anchor  fit-vs-map gap constant over the clip   stops drift toward/away
    prior         bone directions near 2b's               quadratic; sigma 1.0 limbs, 2.0 torso,
                                                            strong inferred
    lean          torso toward/away tilt near 2b's        ~6 deg
    facing        hip/shoulder axes near 2b's             ~9 deg
    lengths       each group length near 2b's             ~12 % slack
    widths        shoulder / hip width = widest on-screen  ~4 %; only if the person faced
                  span at the fit's distance (p95)         the camera at some point
    smoothness    low joint acceleration; pelvis camera    quadratic, 0.012 m/frame^2
                  distance counts twice
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
    covered turn  shoulder turn rate changes <= 0.6 deg/   while a shoulder's depth is
                  frame per frame                          hidden: no mirror flip
    side view     hip adduction <= 15 deg, knee hinge      legs captured; pelvis / feet
                  <= 35 deg off the hip line, pelvis roll  ones only when upright
                  <= 12 deg, ankle midpoint <= 5 cm
                  sideways from the pelvis

Depth evidence: Sapiens2 readings every 4th frame (Depth Pro fallbacks at half
weight), linearly interpolated to the frames between, covered at either sample
= covered; the in-between rows share the samples' total weight.

Solving: (1) rigid placement of the 2b pose, one rotation + translation per
frame (also the "before" number in reports); (2) free fit of directions,
pelvis and lengths from the rotated 2b directions. L-BFGS, restarted until
the loss stops dropping.

Output placement: static camera (fastest of three horizontal background bands
< 0.5 px/frame) -> one offset for
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
from core.filters import fill_gaps, hold_edges, lowpass
from core.skeleton import (BONE_PAIRS, D, DIR_FROM, DIR_TO, GROUPS, J, JOINT_NAMES, LEFT_RIGHT, LEGS, N_JOINTS,
                           bone_lengths_from_groups, build_positions, extract_dirs,
                           group_lengths, map_mediapipe, normalize)

SIGMA_2D_FRAC = 0.01            # 2D noise: 1 % of the image's shorter side (11 px at 1080p, portrait too)
MIN_2D_WEIGHT = 0.2              # 2D weight = MediaPipe visibility, clipped to [0.2, 1]
HIDDEN_2D_WEIGHT = 1 / 3         # older clean files without visibility: occluded points
SIGMA_DEPTH_M = 0.03
SIGMA_ANCHOR_M = 0.05
SIGMA_PRIOR_LIMB = 1.0           # 2b direction prior on captured limb bones (quadratic; weight 1)
SIGMA_PRIOR_TORSO = 2.0          # spine, head, shoulder / hip axes: looser (weight 0.25)
SIGMA_PRIOR_NO_DEPTH = 0.25      # without depth the prior is the only depth evidence
SIGMA_PRIOR_INFERRED = 0.05
SIGMA_LEAN = 0.1
SIGMA_FACING = 0.15              # hip/shoulder axes near 2b's (~9 deg): which way the body faces
LENGTH_SLACK = 0.12
SIGMA_ACC_M = 0.012              # per frame^2 at 30 fps, quadratic: a big jump costs a lot; about a
                                 # walking wrist's / ankle's real acceleration, so real motion isn't smoothed away
PELVIS_Z_ACC_FACTOR = 1.0
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
BG_BANDS = 3                     # horizontal background bands for the camera check
COVER_MARGIN_M = 0.05
LIMB_RADIUS_M = 0.06
# A depth reading is skin, not the joint centre: the joint is this much farther.
# Relative depth cancels a common offset but not these differences (hips vs wrists).
SKIN_DEPTH_M = {"pelvis": 0.08, "spine": 0.10, "neck": 0.05, "head": 0.03,
                "l_hip": 0.08, "r_hip": 0.08, "l_knee": 0.05, "r_knee": 0.05,
                "l_ankle": 0.04, "r_ankle": 0.04, "l_heel": 0.03, "r_heel": 0.03,
                "l_toe": 0.02, "r_toe": 0.02, "l_shoulder": 0.05, "r_shoulder": 0.05,
                "l_elbow": 0.035, "r_elbow": 0.035, "l_wrist": 0.025, "r_wrist": 0.025}
SKIN_VEC = np.array([SKIN_DEPTH_M[n] for n in JOINT_NAMES])
LAP_TOWARD = 0.5                 # thigh's share pointing at the camera (30 deg) that puts the lap over the hip
HAND_RADIUS_M = 0.12             # joints this close to a hand centre: depth is a lower bound
IMPOSSIBLE_SLACK = 1.15          # bone length slack for the reachable depth gap
IMPOSSIBLE_MARGIN_M = 0.08
FALLBACK_WEIGHT = 0.5            # Depth Pro readings where Sapiens2 had none
MAX_INTERP_GAP = 8               # frames; depth isn't interpolated across longer gaps
COVERED_TURN_DEG = 0.6           # change of the shoulder turn per frame (deg/frame per frame at 30 fps) while a shoulder's depth is hidden
SIGMA_COVERED_TURN_DEG = 0.2
WIDTH_PCT = 95                   # widest on-screen shoulder / hip span
WIDTH_MIN_FRAC = 0.75            # of 2b's width: the person faced the camera at some point
WIDTH_SLACK = 0.04
WIDTH_SMOOTH_HZ = 1.5
HIP_ADDUCTION_MAX_DEG = 15.0     # side-view limits: what neither 2D nor depth settles
KNEE_HINGE_MAX_DEG = 35.0        # knee hinge axis vs the hip line
KNEE_HINGE_BEND_DEG = (10.0, 30.0)  # knee bend: limit off below, full weight above
PELVIS_ROLL_MAX_DEG = 12.0
FEET_MID_MAX_M = 0.05            # upright: ankle midpoint sideways from the pelvis
SIGMA_FEET_MID_M = 0.02

T_ = torch.float64


@dataclass
class FitInputs:
    uv: np.ndarray            # [T,20,2] px, NaN where missing
    w2d: np.ndarray           # [T,20] 2D weight: visibility in [0.2, 1], 0 none
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
    z_fallback: np.ndarray = field(default_factory=lambda: np.zeros((0, N_JOINTS)))
    occluded: np.ndarray | None = None
    patch_px: float = 12.0
    hand_uv: np.ndarray | None = None      # [T,2,2] px, left/right hand centre (wrist + fingers)
    frame_ok: np.ndarray | None = None     # [T] person detected (no depth evidence otherwise)
    vis2d: np.ndarray | None = None        # [T,20] seen in 2D (not occluded, not inferred); default w2d == 1
    sigma_2d: float = 7.0                  # px

    def __post_init__(self):
        if self.vis2d is None:
            self.vis2d = np.asarray(self.w2d) >= 1.0


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
        e = np.zeros((0, N_JOINTS))
        self.set_depth(dict(frames=np.zeros(0, int), z=e, w_vis=e, w_cov=e))

    def set_depth(self, ev: dict):
        """Depth rows: frames [N], readings z [N,20], weights of the relative
        (visible) term and the one-sided (covered) term."""
        w_vis, w_cov = ev["w_vis"], ev["w_cov"]
        ok = ((w_vis > 0).sum(1) >= 3)[:, None]  # need a few visible joints for a relative reading
        self.df = torch.as_tensor(ev["frames"], dtype=torch.long)
        self.zs = torch.as_tensor(np.nan_to_num(ev["z"]), dtype=T_)
        self.vis_m = torch.as_tensor((w_vis > 0) & ok, dtype=T_)
        self.vis_d = torch.as_tensor(w_vis * ok, dtype=T_)
        self.cov_d = torch.as_tensor(w_cov * ok, dtype=T_)

    def project(self, P):
        z = P[..., 2].clamp(min=0.05)
        return self.f * P[..., :2] / z[..., None] + self.c

    def terms(self, P):
        """2D, depth (visible, covered, anchor) and smoothness on camera joints [T,20,3]."""
        out = {}
        r = ((self.project(P) - self.uv) / self.x.sigma_2d).pow(2).sum(-1)
        out["2d"] = (self.w2d * soft_l1(r)).sum()
        if len(self.df) and self.vis_d.sum() > 0:
            zf = P[self.df, :, 2]
            nv = self.vis_m.sum(1, keepdim=True).clamp(min=1)
            mf = (zf * self.vis_m).sum(1, keepdim=True) / nv
            ms = (self.zs * self.vis_m).sum(1, keepdim=True) / nv
            rel_f, rel_s = zf - mf, self.zs - ms
            out["depth"] = (self.vis_d * soft_l1(((rel_f - rel_s) / SIGMA_DEPTH_M) ** 2)).sum()
            out["depth_cov"] = (self.cov_d * soft_l1((torch.relu(rel_s - rel_f) / SIGMA_DEPTH_M) ** 2)).sum()
            has = self.vis_d.sum(1) > 0
            gap = (mf - ms)[:, 0][has]
            out["anchor"] = soft_l1(((gap - gap.mean()) / SIGMA_ANCHOR_M) ** 2).sum()
        a = P[2:] - 2 * P[1:-1] + P[:-2]
        out["smooth"] = (a / self.sigma_acc).pow(2).sum()
        pz = a[:, J["pelvis"], 2] / (self.sigma_acc * PELVIS_Z_ACC_FACTOR)
        out["smooth"] = out["smooth"] + (pz ** 2).sum()
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


def free_phase(pb: Problem, x: FitInputs, P_rigid: np.ndarray, R_rigid: np.ndarray,
               widths: dict | None = None, shoulder_hidden: np.ndarray | None = None):
    T = pb.T
    dirs_2b_world = extract_dirs(x.pos_2b)
    dref = np.einsum("tij,tkj->tki", R_rigid, dirs_2b_world)
    glen_2b = group_lengths(x.pos_2b)
    inf_dir = x.inferred[:, DIR_FROM] | x.inferred[:, DIR_TO]
    torso = np.isin(np.arange(len(D)), [D[n] for n in ("spine", "head", "shoulder_axis", "hip_axis")])
    sig = np.where(torso, SIGMA_PRIOR_TORSO, SIGMA_PRIOR_LIMB)[None] if pb.vis_d.sum() > 0 else SIGMA_PRIOR_NO_DEPTH
    w_prior = torch.as_tensor(np.where(inf_dir, 1 / SIGMA_PRIOR_INFERRED, 1 / sig), dtype=T_)
    dref_t = torch.as_tensor(dref, dtype=T_)
    lean_ref = torch.asin(dref_t[:, D["spine"], 2].clamp(-ASIN_MAX, ASIN_MAX))
    log_l2b = torch.as_tensor(np.log(glen_2b), dtype=T_)
    from_image = {g: w for g, w in (widths or {}).items() if w.get("used")}
    on_ratio = torch.as_tensor([g not in from_image for g in GROUPS], dtype=T_)
    pairs = {"shoulder": ("l_shoulder", "r_shoulder"), "hip": ("l_hip", "r_hip")}
    hid = None
    if shoulder_hidden is not None and shoulder_hidden.any():
        hid = torch.as_tensor(shoulder_hidden[:-2] | shoulder_hidden[1:-1] | shoulder_hidden[2:], dtype=T_)
    turn_lim = np.radians(COVERED_TURN_DEG) * (30.0 / x.fps) ** 2
    G_INDEX = {g: i for i, g in enumerate(GROUPS)}
    knee_sides = [s for s in ("l", "r") if x.tracked[LEGS[s][:3]].all()]
    sigma_axis = SIGMA_AXIS_ACC * (30.0 / x.fps) ** 2
    up_cam = torch.as_tensor(x.floor_R[2], dtype=T_)  # world up in camera coordinates

    def loss(pel, raw, logl):
        d = raw / raw.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        P = build_positions(pel, d, torch.exp(logl))
        terms = pb.terms(P)
        terms["prior"] = ((d - dref_t).pow(2).sum(-1) * w_prior ** 2).sum()
        lean = torch.asin(d[:, D["spine"], 2].clamp(-ASIN_MAX, ASIN_MAX))
        terms["lean"] = soft_l1(((lean - lean_ref) / SIGMA_LEAN) ** 2).sum()
        fa = [D["hip_axis"], D["shoulder_axis"]]
        terms["facing"] = soft_l1((d[:, fa] - dref_t[:, fa]).pow(2).sum(-1) / SIGMA_FACING ** 2).sum()
        dl = logl - log_l2b
        over = torch.relu(dl.abs() - np.log(1 + LENGTH_SLACK)) / 0.01
        terms["lengths"] = T * (on_ratio * (soft_l1((dl / LENGTH_SLACK) ** 2) + over ** 2)).sum() / len(GROUPS)
        for g, w in from_image.items():
            # widest on-screen span, in metres at the fit's own distance
            l, r = J[pairs[g][0]], J[pairs[g][1]]
            z = P[w["frames"]][:, [l, r], 2].mean(1)
            dw = logl[G_INDEX[g]] - torch.log((torch.as_tensor(w["span_px"], dtype=T_) * z).mean() / (2 * pb.f))
            over = torch.relu(dw.abs() - np.log(1 + WIDTH_SLACK)) / 0.01
            terms["lengths"] = terms["lengths"] + T * (soft_l1((dw / WIDTH_SLACK) ** 2) + over ** 2) / len(GROUPS)
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
        if hid is not None:
            # no mirror flip while a shoulder's depth is hidden: 2D gives how much the
            # shoulder line turns, not which way. The turn rate may change only slowly:
            # a spin carries on, a sudden reversal into the mirror pose doesn't
            p = sa - (sa * up_cam).sum(-1, keepdim=True) * up_cam
            n = p.norm(dim=-1)
            p = p / n[:, None].clamp(min=1e-6)
            cr = (torch.cross(p[:-1], p[1:], dim=-1) * up_cam).sum(-1)
            turn = torch.atan2(cr, (p[:-1] * p[1:]).sum(-1).clamp(min=-ASIN_MAX))
            defined = ((n[:-2] > 0.3) & (n[1:-1] > 0.3) & (n[2:] > 0.3)).to(T_)  # yaw undefined with the axis vertical
            r = torch.relu((turn[1:] - turn[:-1]).abs() - turn_lim) / np.radians(SIGMA_COVERED_TURN_DEG)
            terms["covered_turn"] = (hid * defined * soft_l1(r ** 2)).sum()
        if knee_sides:
            # side view: the body's left-right axis points at the camera; 2D sees none of it
            # and depth only the near side. These limits sit above what front views show.
            roll = torch.asin((h * up_cam).sum(-1).abs().clamp(max=ASIN_MAX))
            r = torch.relu(roll - np.radians(PELVIS_ROLL_MAX_DEG)) / np.radians(SIGMA_ANATOMY_DEG)
            terms["pelvis_roll"] = (upright * soft_l1(r ** 2)).sum()
        for s in knee_sides:
            sign = 1.0 if s == "l" else -1.0  # outward along the hip axis
            th, sh = d[:, D[f"{s}_thigh"]], d[:, D[f"{s}_shin"]]
            add = torch.asin((-sign * (th * h).sum(-1)).clamp(-ASIN_MAX, ASIN_MAX))
            r = torch.relu(add - np.radians(HIP_ADDUCTION_MAX_DEG)) / np.radians(SIGMA_ANATOMY_DEG)
            terms["hip_adduction"] = terms.get("hip_adduction", 0) + soft_l1(r ** 2).sum()
            hinge = torch.cross(th, sh, dim=-1)
            bend = hinge.norm(dim=-1)
            hinge = hinge / bend[:, None].clamp(min=1e-6)
            off = torch.acos((hinge * h).sum(-1).clamp(-ASIN_MAX, ASIN_MAX))
            lo, hi = np.sin(np.radians(KNEE_HINGE_BEND_DEG))
            bent = ((bend - lo) / (hi - lo)).clamp(0, 1)  # the hinge is undefined when straight
            r = torch.relu(off - np.radians(KNEE_HINGE_MAX_DEG)) / np.radians(SIGMA_ANATOMY_DEG)
            terms["knee_hinge"] = terms.get("knee_hinge", 0) + (bent * soft_l1(r ** 2)).sum()
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
            mid = 0.5 * (P[:, J["l_ankle"]] + P[:, J["r_ankle"]]) - P[:, J["pelvis"]]
            r = torch.relu((mid * h).sum(-1).abs() - FEET_MID_MAX_M) / SIGMA_FEET_MID_M
            terms["feet_mid"] = (upright * soft_l1(r ** 2)).sum()
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


HAND_LANDMARKS = ((15, 17, 19, 21), (16, 18, 20, 22))  # wrist, pinky, index, thumb


def prepare(clean: dict, depth: dict | None, dj: dict | None, cap: dict | None = None) -> FitInputs:
    W, H = int(clean["width"]), int(clean["height"])
    uv = clean["uv_2d"] * np.array([W, H])
    occ = clean["occluded"].astype(bool)
    inferred = clean["inferred"].astype(bool)
    seen = np.isfinite(uv).all(-1) & ~inferred
    if "visibility" in clean:
        w2d = np.clip(clean["visibility"], MIN_2D_WEIGHT, 1.0) * seen
    else:
        w2d = np.where(occ, HIDDEN_2D_WEIGHT, 1.0) * seen
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
                  patch_px=max(8.0, 0.012 * np.hypot(W, H)),
                  frame_ok=~clean["undetected"].astype(bool) if "undetected" in clean else np.ones(T, bool),
                  vis2d=seen & ~occ, sigma_2d=SIGMA_2D_FRAC * min(W, H))
    if cap is not None:
        with np.errstate(all="ignore"):
            hands = np.stack([np.nanmean(cap["landmarks_2d"][:, list(ix)], axis=1) for ix in HAND_LANDMARKS], 1)
        x.hand_uv = np.where(x.frame_ok[:, None, None], hands, np.nan)

    if dj is not None:
        x.depth_frames = dj["frames"].astype(int)
        x.z_surface = map_mediapipe(dj["z_surface"][:, :, None])[..., 0] + SKIN_VEC
        x.on_mask = map_mediapipe(dj["on_mask"][:, :, None].astype(float), "min")[..., 0] > 0.5
        x.in_frame = map_mediapipe(dj["in_frame"][:, :, None].astype(float), "min")[..., 0] > 0.5
        x.z_fallback = (map_mediapipe(dj["z_fallback"][:, :, None])[..., 0] + SKIN_VEC if "z_fallback" in dj
                        else np.full_like(x.z_surface, np.nan))

    # initial distance: depth readings on the torso > depth.npz body distance > pinhole size
    torso = [J[n] for n in ("pelvis", "neck", "l_shoulder", "r_shoulder", "l_hip", "r_hip")]
    dist = np.full(T, np.nan)
    if len(x.depth_frames):
        zt = np.where(x.on_mask[:, torso], x.z_surface[:, torso], np.nan)
        has = np.isfinite(zt).any(1)
        dist[x.depth_frames[has]] = np.nanmedian(zt[has], axis=1)
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


def lap_cover(x: FitInputs, P_cam: np.ndarray) -> np.ndarray:
    """[N,20]: a thigh pointing at the camera (seated, facing it) puts the lap over
    its hip pixel, so the hip reading is the lap: at least this far. The cover
    check by other limbs can't see this, since the thigh is attached to the hip."""
    f = x.depth_frames
    out = np.zeros((len(f), N_JOINTS), bool)
    for s in "lr":
        th = P_cam[f, J[f"{s}_knee"]] - P_cam[f, J[f"{s}_hip"]]
        lap = -th[:, 2] / np.maximum(np.linalg.norm(th, axis=1), 1e-9) > LAP_TOWARD
        out[:, J[f"{s}_hip"]] |= lap
        out[:, J["pelvis"]] |= lap
    return out


def near_hand(x: FitInputs, P_cam: np.ndarray) -> np.ndarray:
    """[N,20] joints within HAND_RADIUS_M of a hand centre on screen: the reading
    may be the hand or whatever it holds (cup, bag, phone), which the person mask
    may not cover."""
    f = x.depth_frames
    if x.hand_uv is None:
        return np.zeros((len(f), N_JOINTS), bool)
    d = np.linalg.norm(x.uv[f][:, :, None] - x.hand_uv[f][:, None], axis=-1)   # [N,20,2]
    with np.errstate(all="ignore"):
        d = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=-1)
    return d < x.focal * HAND_RADIUS_M / np.maximum(P_cam[f, :, 2], 0.3)


def impossible_depth(x: FitInputs, z: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """[N,20]: a bone of length L whose ends are some span apart on screen can only
    reach sqrt((1.15 L)^2 - span^2) + 8 cm toward or away from the camera. When its
    two readings differ by more, one is wrong; the nearer end becomes a lower bound."""
    f = x.depth_frames
    out = np.zeros((len(f), N_JOINTS), bool)
    L = bone_lengths_from_groups(group_lengths(x.pos_2b))
    for (a, b), l in zip(BONE_PAIRS, L):
        ok = finite[:, a] & finite[:, b] & np.isfinite(x.uv[f][:, [a, b]]).all((1, 2))
        if not ok.any():
            continue
        za, zb = z[:, a], z[:, b]
        span = np.linalg.norm(x.uv[f, a] - x.uv[f, b], axis=-1) * 0.5 * (za + zb) / x.focal
        with np.errstate(all="ignore"):
            reach = np.sqrt(np.clip((IMPOSSIBLE_SLACK * l) ** 2 - span ** 2, 0, None)) + IMPOSSIBLE_MARGIN_M
            bad = ok & (np.abs(za - zb) > reach)
        out[:, a] |= bad & (za < zb)
        out[:, b] |= bad & (zb <= za)
    return out


def interpolate_rows(frames: np.ndarray, z: np.ndarray, w_vis: np.ndarray, w_cov: np.ndarray):
    """Depth rows for the frames between samples, linearly interpolated. A reading
    is visible only if both samples are; if either sample was covered, the frame
    is covered too (a limb that covered a joint at one sample may still cover it
    a few frames later)."""
    rows = [(frames, z, w_vis, w_cov)]
    have = (w_vis > 0) | (w_cov > 0)
    for k in range(len(frames) - 1):
        a, b = frames[k], frames[k + 1]
        if b - a < 2 or b - a > MAX_INTERP_GAP:
            continue
        both = have[k] & have[k + 1]
        vis = (w_vis[k] > 0) & (w_vis[k + 1] > 0)
        w = np.minimum(np.maximum(w_vis[k], w_cov[k]), np.maximum(w_vis[k + 1], w_cov[k + 1]))
        s = ((np.arange(a + 1, b) - a) / (b - a))[:, None]
        n = len(s)
        rows.append((np.arange(a + 1, b), (1 - s) * z[k] + s * z[k + 1],
                     np.tile(np.where(vis, w, 0.0), (n, 1)), np.tile(np.where(both & ~vis, w, 0.0), (n, 1))))
    f, zz, wv, wc = (np.concatenate(c) for c in zip(*rows))
    order = np.argsort(f, kind="stable")
    k = len(frames) / len(f)   # same total depth weight as the samples alone: in-between rows add no new evidence
    return f[order], zz[order], k * wv[order], k * wc[order]


def depth_evidence(x: FitInputs, P_cam: np.ndarray, interpolate: bool = True) -> dict:
    """Depth rows for the solver: which readings count as visible (relative depth)
    and which only as "at least this far" (covered by a limb, the torso, the near
    side of an overlapping pair, a hand, or an impossible bone)."""
    T = len(x.uv)
    f = x.depth_frames
    e = np.zeros((0, N_JOINTS))
    out = dict(frames=np.zeros(0, int), z=e, w_vis=e, w_cov=e, shoulder_hidden=np.zeros(T, bool), counts={})
    if not len(f):
        return out
    zfb = x.z_fallback if x.z_fallback.shape == x.z_surface.shape else np.full_like(x.z_surface, np.nan)
    fallback = ~np.isfinite(x.z_surface) & np.isfinite(zfb)
    z = np.where(fallback, zfb, x.z_surface)
    frame_ok = x.frame_ok[f] if x.frame_ok is not None else np.ones(len(f), bool)
    finite = np.isfinite(z) & x.in_frame & frame_ok[:, None]
    cov_seg = (covered_by_body(P_cam, f, x.focal, x.center, 0.5 * x.patch_px) | far_of_overlapping_pair(x, P_cam)
               | lap_cover(x, P_cam))
    hand = near_hand(x, P_cam)
    imp = impossible_depth(x, z, finite & ~cov_seg)
    on = x.on_mask | fallback   # a fallback patch missed the person mask by definition
    vis = x.vis2d[f] & on & finite & ~cov_seg & ~hand & ~imp
    cov = finite & ~vis & (x.occluded[f] | ~on | cov_seg | x.inferred[f] | hand | imp)
    w = np.where(fallback, FALLBACK_WEIGHT, 1.0)
    w_vis, w_cov = np.where(vis, w, 0.0), np.where(cov, w, 0.0)
    out["counts"] = dict(n_depth_frames=int(len(f)), n_depth_visible=int(vis.sum()),
                         n_depth_covered=int(cov.sum()), n_depth_hand=int((hand & finite).sum()),
                         n_depth_impossible=int((imp & finite).sum()),
                         n_depth_fallback=int((fallback & finite).sum()))
    if interpolate:
        f, z, w_vis, w_cov = interpolate_rows(f, np.nan_to_num(z), w_vis, w_cov)
    out.update(frames=f, z=z, w_vis=w_vis, w_cov=w_cov)
    seen = np.zeros((T, N_JOINTS), bool)
    seen[f] = w_vis > 0
    out["shoulder_hidden"] = ~(seen[:, J["l_shoulder"]] & seen[:, J["r_shoulder"]])
    return out


def image_widths(x: FitInputs, P_cam: np.ndarray) -> dict:
    """Shoulder and hip width from the image. A wrong width makes the solver fake a
    body turn to get the projected width right. The widest on-screen span (95th
    percentile, metres at the pair's depth) is the true width if the person faced
    the camera at some point, i.e. it reaches 0.75 of 2b's ratio width."""
    out = {}
    glen = group_lengths(x.pos_2b)
    for g, (l, r) in (("shoulder", ("l_shoulder", "r_shoulder")), ("hip", ("l_hip", "r_hip"))):
        l, r = J[l], J[r]
        ok = x.vis2d[:, l] & x.vis2d[:, r]
        expected = 2 * glen[GROUPS.index(g)]
        if ok.sum() < 10:
            out[g] = dict(used=False, ratio_width_m=float(expected))
            continue
        # low-pass first: the 95th percentile of a raw span is mostly 2D jitter
        px = lowpass(np.where(ok, np.linalg.norm(x.uv[:, l] - x.uv[:, r], axis=-1), np.nan), x.fps, WIDTH_SMOOTH_HZ)
        span = px * P_cam[:, [l, r], 2].mean(1) / x.focal
        good = ok & np.isfinite(span)
        if good.sum() < 10:
            out[g] = dict(used=False, ratio_width_m=float(expected))
            continue
        wide = float(np.percentile(span[good], WIDTH_PCT))
        # the frames around the percentile; the fit re-measures them at its own distance
        lo, hi = np.percentile(span[good], [WIDTH_PCT - 5, min(WIDTH_PCT + 4, 100)])
        frames = np.nonzero(good & (span >= lo) & (span <= hi))[0]
        out[g] = dict(used=bool(wide >= WIDTH_MIN_FRAC * expected), image_width_m=round(wide, 3),
                      ratio_width_m=round(float(expected), 3), frames=frames, span_px=px[frames])
    return out


# ---------------------------------------------------------------- reporting

def _stats(pb: Problem, x: FitInputs, P: np.ndarray) -> dict:
    with torch.no_grad():
        uvp = pb.project(torch.as_tensor(P, dtype=T_)).numpy()
    err = np.linalg.norm(uvp - x.uv, axis=-1)
    vis2d = x.vis2d
    out = {"px_2d_median_visible": float(np.nanmedian(err[vis2d])) if vis2d.any() else None,
           "px_2d_p90_visible": float(np.nanpercentile(err[vis2d], 90)) if vis2d.any() else None}
    m = pb.vis_m.numpy() > 0
    if m.any():
        zf, zs = P[pb.df.numpy(), :, 2], pb.zs.numpy()
        with np.errstate(all="ignore"):
            mf = np.where(m, zf, 0).sum(1, keepdims=True) / m.sum(1, keepdims=True)
            ms = np.where(m, zs, 0).sum(1, keepdims=True) / m.sum(1, keepdims=True)
        e = np.abs((zf - mf) - (zs - ms))[m]
        out["rel_depth_cm_median_visible"] = float(np.nanmedian(e) * 100)
    return out


# ---------------------------------------------------------------- placement

def background_motion(video_path, uv: np.ndarray, n_pairs: int = 30, n_bands: int = BG_BANDS):
    """Background motion in px/frame (person box masked out): the fastest of
    n_bands horizontal bands, each the median over frame pairs of its features'
    median motion. A dolly shot moves the nearby ground much faster than the
    distant buildings, which often have more features, so one median over the
    whole background would call it static. Returns (px/frame, per-band list) or
    (None, []) when the video is unavailable."""
    import cv2

    from core import video

    try:
        vi = video.info(video_path)
    except FileNotFoundError:
        return None, []
    T = vi["n_frames"]
    starts = video.evenly_spaced(max(T - 1, 1), n_pairs)
    want = sorted(set(starts) | set(starts + 1))
    fr = {i: cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) for i, rgb in video.frames(video_path, want)}
    mags = [[] for _ in range(n_bands)]
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
        edges = np.linspace(0, g0.shape[0], n_bands + 1).astype(int)
        for b in range(n_bands):
            band = np.zeros_like(mask)
            band[edges[b]:edges[b + 1]] = mask[edges[b]:edges[b + 1]]
            p0 = cv2.goodFeaturesToTrack(g0, 100, 0.01, 8, mask=band)
            if p0 is None or len(p0) < 8:
                continue
            p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None)
            good = st[:, 0] == 1
            if good.sum() >= 8:
                mags[b].append(np.median(np.linalg.norm((p1 - p0)[good, 0], axis=1)))
    bands = [float(np.median(m)) if len(m) >= 3 else None for m in mags]
    found = [b for b in bands if b is not None]
    return (max(found) if found else None), bands


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
    # depth evidence needs a pose; take it from the 2b pose placed rigidly without depth first
    P_r, R_r = rigid_phase(pb, x)
    ev = depth_evidence(x, P_r)
    if not (ev["w_vis"] > 0).any():
        print("    WARNING: no depth readings; toward/away comes from the 2b prior only (run core.joint_depth)")
    pb.set_depth(ev)
    if len(x.depth_frames):
        P_r, R_r = rigid_phase(pb, x)
    before = _stats(pb, x, P_r)
    widths = image_widths(x, P_r)
    hidden = ev["shoulder_hidden"] if len(x.depth_frames) else None
    P_f, glen = free_phase(pb, x, P_r, R_r, widths, hidden)
    if len(x.depth_frames):
        # the rigid 2b pose is too rough to judge who covers whom; re-check on the fit and solve again
        ev = depth_evidence(x, P_f)
        pb.set_depth(ev)
        P_f, glen = free_phase(pb, x, P_r, R_r, widths, ev["shoulder_hidden"])
    after = _stats(pb, x, P_f)
    Wd, cam = place(P_f, x, static, has_feet, glen)
    summary = dict(before_rigid_2b=before, after_fit2d=after, seconds=round(time.time() - t0, 1),
                   focal_px=x.focal, camera="static" if static else "moving",
                   n_depth_rows=int(len(ev["frames"])), **ev["counts"],
                   shoulder_hidden_frames=int(ev["shoulder_hidden"].sum()) if hidden is not None else 0,
                   widths={g: {k: v for k, v in w.items() if k not in ("frames", "span_px")}
                           for g, w in widths.items()},
                   length_change_pct={g: round(100 * (glen[i] / group_lengths(x.pos_2b)[i] - 1), 1)
                                      for i, g in enumerate(GROUPS)})
    return dict(positions_fit2d=Wd, positions_fit2d_cam=P_f, bone_lengths_fit2d=bone_lengths_from_groups(glen),
                fit2d_camera_pos=cam, fit2d_focal_px=x.focal, summary=summary)


def run(clip: schema.Clip, camera: str = "auto") -> dict:
    clean_path = clip.clean
    clean = schema.load(clean_path)
    depth = schema.load(clip.depth) if clip.depth.exists() else None
    dj = schema.load(clip.depth_joints) if clip.depth_joints.exists() else None
    cap = schema.load(clip.capture) if clip.capture.exists() else None
    x = prepare(clean, depth, dj, cap)
    motion, bands = None, []
    if camera == "auto":
        motion, bands = background_motion(clip.video, x.uv)
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
    summary["background_bands_px_per_frame"] = bands
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
