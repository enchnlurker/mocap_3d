"""Stage 2: cleanup (+ 2b proportions). Reads 1a + 1c.

    python -m core.clean clip01 [--no-depth] [--free-floor]

Writes clean_depth.npz (when depth.npz was used) or clean.npz:
  1. Map 33 landmarks to 20 joints.
  2. Fix left/right swaps using 2D continuity between frames.
  3. Drop bad points. Low-visibility points are dropped unless occluded:
     present, inside the frame (3% margin), and the joint is kept in >= 50% of
     frames. Occluded points keep MediaPipe's guess, flagged `occluded`.
     Spikes are dropped either way.
  4. Fill and smooth: short gaps get shape-preserving interpolation, then a
     lag-free low-pass, heavier on depth.
  5. Level the floor: floor direction from depth > feet (RANSAC) > none; Z = 0.
  6. Set the height: ONE offset per clip puts the planted ankle at ankle height;
     clips without feet use the depth-measured hip height instead.
  7. Add hip motion: bob (Z) + sway (X) from the 2D hip track, band-passed
     0.4-6 Hz, pixels -> metres at the body's distance. No forward travel.
  8. Fill hidden limbs with rules (core.infer_hidden), flagged `inferred`.
  9. Detect foot contacts; save per-joint 2D points (uv_2d) for 2c.
 2b. Proportions (core.proportions), on by default.
"""

from __future__ import annotations

import argparse

import numpy as np

from core import proportions, schema
from core.filters import bandpass, fill_gaps, hold_edges, lowpass, mode_filter, spikes
from core.infer_hidden import infer_hidden
from core.skeleton import (ANKLE_HEIGHT_RATIO, BONE_NAMES, BONE_PAIRS, CONTACT_JOINTS, FEET, J,
                           JOINT_NAMES, bone_lengths_from_groups, group_lengths, map_mediapipe)
from core.depth import plane_basis

VIS_THRESH = 0.5
PRESENCE_THRESH = 0.5
EDGE_MARGIN = 0.03
MIN_KEPT_FRAC = 0.5
MAX_GAP_S = 0.5
SMOOTH_HZ = 6.0
SMOOTH_DEPTH_HZ = 3.0
HIP_BAND_HZ = (0.4, 6.0)
CONTACT_HEIGHT_M = 0.04
CONTACT_SPEED_MS = 0.4

SWAP_CHAINS = [
    ([J["l_knee"], J["l_ankle"], J["l_heel"], J["l_toe"]], [J["r_knee"], J["r_ankle"], J["r_heel"], J["r_toe"]]),
    ([J["l_elbow"], J["l_wrist"]], [J["r_elbow"], J["r_wrist"]]),
]


SWAP_SWITCH_COST = 0.3   # per label flip, in body-size units
SWAP_STAY_COST = 0.02    # per swapped frame: ties go to MediaPipe's own labels


def fix_swaps(uv, lm3, arrays, scale_px, scale_m):
    """Swap limb chains whose left/right labels jumped between frames.

    Per chain, picks the labelling over the whole clip (Viterbi over
    "as given" / "swapped") that moves least between frames, in 2D and in
    MediaPipe's 3D (which separates near and far limbs that overlap in a
    side view). Flipping costs SWAP_SWITCH_COST, so one ambiguous frame can't
    start a long run of wrong labels.

    uv [T,20,2], lm3 [T,20,3]; both and `arrays` are swapped in place.
    Returns swapped [T,20].
    """
    swapped = np.zeros(uv.shape[:2], bool)
    for lc, rc in SWAP_CHAINS:
        both = lc + rc
        frames = np.nonzero(np.isfinite(uv[:, both]).all((1, 2)) & np.isfinite(lm3[:, both]).all((1, 2)))[0]
        if len(frames) < 2:
            continue

        def move(a, b, flip):
            """Mean motion of the chain from frame a to b, b's labels flipped or not."""
            src = rc + lc if flip else both
            d2 = np.linalg.norm(uv[b, src] - uv[a, both], axis=-1).mean() / scale_px
            d3 = np.linalg.norm(lm3[b, src] - lm3[a, both], axis=-1).mean() / scale_m
            return d2 + d3

        n = len(frames)
        cost = np.array([0.0, SWAP_STAY_COST])
        back = np.zeros((n, 2), int)
        for k in range(1, n):
            a, b = frames[k - 1], frames[k]
            same, cross = move(a, b, False), move(a, b, True)
            new = np.empty(2)
            for s in (0, 1):
                # from state p at a to state s at b: relative flip iff p != s
                opts = [cost[p] + (same if p == s else cross + SWAP_SWITCH_COST) for p in (0, 1)]
                back[k, s] = int(np.argmin(opts))
                new[s] = min(opts) + SWAP_STAY_COST * s
            cost = new
        state = np.zeros(n, int)
        state[-1] = int(np.argmin(cost))
        for k in range(n - 1, 0, -1):
            state[k - 1] = back[k, state[k]]
        for t in frames[state == 1]:
            for arr in [uv, lm3, *arrays]:
                arr[t, both] = arr[t, rc + lc]
            swapped[t, both] = True
    return swapped


def _pair_scale(a: np.ndarray, keep: np.ndarray) -> tuple[float, str]:
    """Median trunk (pelvis-neck) or shoulder width of a [T,20,D] array over kept frames."""
    for p, q in (("pelvis", "neck"), ("l_shoulder", "r_shoulder")):
        m = keep[:, J[p]] & keep[:, J[q]]
        if m.sum() >= 5:
            return float(np.median(np.linalg.norm(a[m, J[p]] - a[m, J[q]], axis=-1))), f"{p}-{q}"
    return float("nan"), ""


def floor_from_feet(lm3: np.ndarray, keep: np.ndarray, free: bool = False, rng=None):
    """RANSAC on the lowest foot point per frame (camera-aligned coords, y down).

    Pitch-only by default (y = a z + b); free=True also fits sideways tilt.
    Returns an up normal in camera coords, or None if pitch is unobservable.
    """
    rng = rng or np.random.default_rng(0)
    pts = []
    for t in range(len(lm3)):
        f = [j for j in FEET if keep[t, j]]
        if f:
            pts.append(lm3[t, f][np.argmax(lm3[t, f, 1])])
    pts = np.asarray(pts)
    if len(pts) < 10 or np.ptp(pts[:, 2]) < 0.1:
        return None
    X = np.c_[pts[:, 2], pts[:, 0], np.ones(len(pts))] if free else np.c_[pts[:, 2], np.ones(len(pts))]
    y = pts[:, 1]
    best, best_n = None, 0
    k = X.shape[1]
    for _ in range(300):
        s = rng.choice(len(pts), k, replace=False)
        try:
            coef = np.linalg.solve(X[s], y[s])
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(X @ coef - y) < 0.02
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    if best is None or best_n < 0.3 * len(pts):
        return None
    coef = np.linalg.lstsq(X[best], y[best], rcond=None)[0]
    a, c = coef[0], (coef[1] if free else 0.0)
    n = np.array([c, -1.0, a])  # gradient of a z + c x - y, pointing up (-y)
    return n / np.linalg.norm(n)


def clean_arrays(cap: dict, depth: dict | None = None, *, free_floor: bool = False) -> dict:
    fps = float(cap["fps"])
    W, H = float(cap["width"]), float(cap["height"])
    T = len(cap["landmarks"])
    max_gap = int(round(MAX_GAP_S * fps))

    # 1. map
    lm3 = map_mediapipe(cap["landmarks"]).astype(float)
    lm2 = map_mediapipe(cap["landmarks_2d"]).astype(float)
    vis = map_mediapipe(cap["confidence"], "min").astype(float)
    pres = map_mediapipe(cap["presence"], "min").astype(float)
    raw3 = lm3.copy()
    detected = np.isfinite(lm2).all(-1) & np.isfinite(lm3).all(-1)

    # 2. swaps
    scale_px, _ = _pair_scale(lm2, detected)
    if not np.isfinite(scale_px):
        scale_px = 0.25 * max(W, H)
    scale_m, _ = _pair_scale(lm3, detected)
    if not np.isfinite(scale_m):
        scale_m = 0.5
    swapped = fix_swaps(lm2, lm3, [vis, pres], scale_px, scale_m)

    # 3. drop
    m = EDGE_MARGIN
    in_frame = detected & (lm2[..., 0] >= m * W) & (lm2[..., 0] <= (1 - m) * W) \
        & (lm2[..., 1] >= m * H) & (lm2[..., 1] <= (1 - m) * H)
    visible = detected & (vis >= VIS_THRESH)
    occluded = detected & ~visible & (pres >= PRESENCE_THRESH) & in_frame
    weak = (visible | occluded).mean(0) < MIN_KEPT_FRAC
    occluded[:, weak] = False  # no revival from edge frames
    keep = visible | occluded
    for j in range(len(JOINT_NAMES)):
        keep[spikes(lm2[:, j], keep[:, j], 0.15 * scale_px), j] = False
    occluded &= keep
    dropped = detected & ~keep
    lm3[~keep] = np.nan
    lm2[~keep] = np.nan
    tracked = keep.any(0)

    # 4. fill short gaps + zero-phase smoothing (heavier on depth = camera z)
    filled = np.zeros_like(keep)
    for j in np.nonzero(tracked)[0]:
        lm3[:, j], filled[:, j] = fill_gaps(lm3[:, j], keep[:, j], max_gap)
    lm3[..., :2] = lowpass(lm3[..., :2], fps, SMOOTH_HZ)
    lm3[..., 2:] = lowpass(lm3[..., 2:], fps, SMOOTH_DEPTH_HZ)
    have = keep | filled

    # feet count only when actually seen: a guess under a blanket or a table isn't a planted foot
    feet_frac = (keep & ~occluded)[:, FEET].any(1).mean()
    has_feet = bool(feet_frac >= MIN_KEPT_FRAC)

    # 5. floor direction: depth > feet > none
    n_cam, floor_source = None, "none"
    if depth is not None and bool(depth.get("floor_found", False)):
        n_cam, floor_source = np.asarray(depth["floor_normal_clip"], float), "depth"
    elif has_feet:
        n_cam = floor_from_feet(lm3, have, free_floor)
        floor_source = "feet" if n_cam is not None else "none"
    if n_cam is None:
        n_cam = np.array([0.0, -1.0, 0.0])
    R = plane_basis(n_cam)  # camera -> world
    pos = lm3 @ R.T
    pos_raw = raw3 @ R.T

    # 7. hip motion from the 2D hip track (pelvis, or neck if hips never seen)
    root_joint = "pelvis" if keep[:, J["pelvis"]].mean() >= MIN_KEPT_FRAC else "neck"
    focal = float("nan")
    if depth is not None:
        dist = np.nanmedian(depth["pelvis_distance_m"]) if np.isfinite(depth["pelvis_distance_m"]).any() else np.nan
        focal = float(np.nanmedian(depth["focal_px"]))
        m_per_px = dist / focal
    else:
        m_per_px = np.nan
    if not np.isfinite(m_per_px):
        s3, _ = _pair_scale(lm3, keep)
        s2, _ = _pair_scale(lm2, keep)
        m_per_px = s3 / s2 if np.isfinite(s3) and s2 > 0 else 0.0
    track = lm2[:, J[root_joint]].copy()
    root_motion = np.zeros((T, 3))
    ok = np.isfinite(track).all(1)
    if ok.sum() > 3 * fps:
        track, _ = fill_gaps(track, ok, None)
        track = hold_edges(track)
        band = bandpass(track, fps, *HIP_BAND_HZ)
        root_motion[:, 0] = band[:, 0] * m_per_px
        root_motion[:, 2] = -band[:, 1] * m_per_px
    pos += root_motion[:, None]
    pos_raw += root_motion[:, None]

    # 6. height: feet > depth hip height > unknown
    height = proportions.height_from_trunk(pos, have)
    if has_feet:
        offset, vertical = proportions.ground_offset(pos, height, have), "feet"
    else:
        offset, vertical = None, "unknown"
        if depth is not None and bool(depth.get("floor_found", False)) and root_joint == "pelvis":
            hh = _depth_hip_heights(depth, lm2_px=map_mediapipe(cap["landmarks_2d"]), W=W, H=H)
            if np.isfinite(hh).any():
                p = pos[:, J["pelvis"], 2]
                offset, vertical = float(np.nanmedian(hh) - np.nanmedian(p)), "depth_hip"
    if offset is not None:
        pos[..., 2] += offset
        pos_raw[..., 2] += offset

    # 8. hidden limbs
    pos, inferred, activity = infer_hidden(pos, tracked, height, vertical != "unknown", fps)
    if vertical != "feet":
        # unknown: put the inferred feet on the floor. depth_hip: the body may not sink
        # into the floor (the hip reading can hit a lap, a table or a newspaper)
        lift = proportions.ground_offset(pos, height)
        if vertical == "depth_hip":
            lift = max(lift, 0.0)
            if lift > 0.01:
                vertical = "depth_hip+lifted"
        pos[..., 2] += lift
        pos_raw[..., 2] += lift
        offset = (offset or 0.0) + lift
    captured = have & ~inferred

    # 9. contacts
    contacts = np.zeros((T, 4), bool)
    if has_feet:
        for c, j in enumerate(CONTACT_JOINTS):
            z = pos[:, j, 2]
            ok = captured[:, j]
            if ok.sum() < 5:
                continue
            ground = np.percentile(z[ok], 5)
            v = np.zeros(T)
            v[1:] = np.linalg.norm(np.diff(pos[:, j, :2], axis=0), axis=1) * fps
            contacts[:, c] = mode_filter(ok & (z - ground < CONTACT_HEIGHT_M) & (v < CONTACT_SPEED_MS), 5)

    glen = group_lengths(pos, captured)
    glen = np.where(np.isfinite(glen), glen, group_lengths(pos))

    # 2b
    prop, prop_bones, _ = proportions.proportional(pos, height, has_feet, captured)

    return dict(
        positions=pos, positions_raw=pos_raw, root_pos=pos[:, J["pelvis"]], root_motion=root_motion,
        contacts=contacts, contact_names=np.array(["l_heel", "l_toe", "r_heel", "r_toe"]),
        bone_names=np.array(BONE_NAMES), bone_pairs=BONE_PAIRS, bone_lengths=bone_lengths_from_groups(glen),
        floor_R=R, floor_offset=float(offset or 0.0), floor_source=floor_source, floor_normal_cam=n_cam,
        vertical_source=vertical, height_m=height,
        dropped=dropped, filled=filled, swapped=swapped, occluded=occluded, presence=pres,
        inferred=inferred, tracked=tracked, activity=activity,
        uv_2d=lm2 / np.array([W, H]), width=int(W), height=int(H),
        m_per_px=m_per_px, focal_px=focal, root_joint=root_joint, has_feet=has_feet,
        positions_proportional=prop, bone_lengths_proportional=prop_bones,
        proportional_height_m=height, source="mediapipe",
    )


def _depth_hip_heights(depth: dict, lm2_px: np.ndarray, W: float, H: float) -> np.ndarray:
    """Pelvis height above the depth floor at each depth frame."""
    n = np.asarray(depth["floor_normal_clip"], float)
    d = float(depth["floor_d_clip"])
    cx, cy = (W - 1) / 2, (H - 1) / 2
    out = []
    for k, t in enumerate(depth["frames"]):
        z, f = depth["pelvis_distance_m"][k], depth["focal_px"][k]
        u, v = lm2_px[t, J["pelvis"]]
        p = np.array([(u - cx) / f * z, (v - cy) / f * z, z])
        out.append(float(n @ p + d))
    return np.asarray(out)


def run(clip: schema.Clip, use_depth: bool = True, free_floor: bool = False, plots: bool = True) -> dict:
    cap = schema.load(clip.capture)
    depth = schema.load(clip.depth) if use_depth and clip.depth.exists() else None
    out = clean_arrays(cap, depth, free_floor=free_floor)
    path = clip / ("clean_depth.npz" if depth is not None else "clean.npz")
    schema.save(path, cap["fps"], JOINT_NAMES, **out)
    _report(clip, out, path)
    if plots:
        from core import plots as P

        P.clean_before_after(out, cap["fps"], clip / "clean_before_after.png")
    return out


def _report(clip, out, path):
    T = len(out["positions"])
    cap = out["tracked"].sum()
    print(f"{clip.name}: {T} frames -> {path.name}")
    print(f"  tracked {cap}/20 joints, feet={'yes' if out['has_feet'] else 'no'}, "
          f"floor={out['floor_source']}, height={out['vertical_source']} ({out['height_m']:.2f} m body)")
    print(f"  dropped {out['dropped'].mean():.1%}  filled {out['filled'].mean():.1%}  "
          f"occluded {out['occluded'].mean():.1%}  swapped frames {out['swapped'].any(1).sum()}  "
          f"inferred {out['inferred'].mean():.1%}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--no-depth", action="store_true", help="ignore depth.npz even if present")
    ap.add_argument("--free-floor", action="store_true", help="feet floor fit with sideways tilt")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    run(schema.Clip(a.clip), not a.no_depth, a.free_floor, not a.no_plots)


if __name__ == "__main__":
    main()
