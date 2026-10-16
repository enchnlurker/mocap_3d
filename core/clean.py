"""Stage 2: cleanup (+ 2b proportions). Reads 1a + 1c.

    python -m core.clean clip01 [--no-depth] [--free-floor] [--lean-fix] [--config c.json]

Every threshold lives in CleanConfig. configs/<clip>/clean.json (or --config)
overrides any of its fields for one clip.

Writes clean_depth.npz (when depth.npz was used) or clean.npz, plus
clean_summary.json (counts, depth reference, lean stats, the config used) and
clean_before_after.png:
  1. Map 33 landmarks to 20 joints. Drop shrunk-body frames: a person walking
     very close to the camera, with only hips and legs in shot, still comes
     back as a whole, confident person squeezed into the visible part. Such a
     frame's on-screen size per metre (from the least foreshortened bone, as
     projection can only shorten a bone) falls far below the surrounding second's.
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

Optional (lean_fix, off by default): rotate the upper body about the pelvis so
its lean toward the camera matches Depth Pro's torso surface. At ~3 m the
reading is inside Depth Pro's error, so it is only reported unless asked for.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from core import proportions, schema
from core.filters import bandpass, fill_gaps, hold_edges, lowpass, mode_filter, spikes
from core.infer_hidden import infer_hidden
from core.skeleton import (BONE_NAMES, BONE_PAIRS, CONTACT_JOINTS, FEET, J, JOINT_NAMES,
                           bone_lengths_from_groups, group_lengths, map_mediapipe)
from core.depth import plane_basis


@dataclass
class CleanConfig:
    vis_thresh: float = 0.5
    presence_thresh: float = 0.5
    edge_margin: float = 0.03          # fraction of the image kept as "inside the frame"
    min_kept_frac: float = 0.5         # joint kept in fewer frames loses its occluded samples; feet seen in fewer -> no feet
    max_gap_s: float = 0.5
    smooth_hz: float = 6.0
    smooth_depth_hz: float = 3.0
    hip_band_hz: tuple[float, float] = (0.4, 6.0)
    contact_height_m: float = 0.04
    contact_speed_ms: float = 0.4
    swap_switch_cost: float = 0.3      # per label flip, in body-size units
    swap_stay_cost: float = 0.02       # per swapped frame: ties go to MediaPipe's own labels
    shrink_ratio: float = 1.6          # frame size per metre below 1/ratio of the surrounding second -> undetected
    shrink_window_s: float = 1.0
    shrink_min_bone_m: float = 0.1     # bones shorter than this in MediaPipe 3D don't measure size
    free_floor: bool = False           # feet floor fit with sideways tilt
    lean_fix: bool = False             # rotate the upper body to Depth Pro's torso lean

    @classmethod
    def load(cls, path: str | Path | None, **overrides) -> "CleanConfig":
        """Defaults, then the JSON file (if it exists), then overrides that aren't None."""
        cfg = cls()
        if path is not None and Path(path).exists():
            data = json.loads(Path(path).read_text())
            names = {f.name for f in dataclasses.fields(cls)}
            unknown = set(data) - names
            if unknown:
                raise ValueError(f"{path}: unknown CleanConfig fields {sorted(unknown)}")
            if "hip_band_hz" in data:
                data["hip_band_hz"] = tuple(data["hip_band_hz"])
            cfg = dataclasses.replace(cfg, **data)
        return dataclasses.replace(cfg, **{k: v for k, v in overrides.items() if v is not None})


def clip_config_path(clip: schema.Clip) -> Path:
    return schema.ROOT / "configs" / clip.name / "clean.json"

SWAP_CHAINS = [
    ([J["l_knee"], J["l_ankle"], J["l_heel"], J["l_toe"]], [J["r_knee"], J["r_ankle"], J["r_heel"], J["r_toe"]]),
    ([J["l_elbow"], J["l_wrist"]], [J["r_elbow"], J["r_wrist"]]),
]


def fix_swaps(uv, lm3, arrays, scale_px, scale_m, switch_cost=0.3, stay_cost=0.02):
    """Swap limb chains whose left/right labels jumped between frames.

    Per chain, picks the labelling over the whole clip (Viterbi over
    "as given" / "swapped") that moves least between frames, in 2D and in
    MediaPipe's 3D (which separates near and far limbs that overlap in a
    side view). Flipping costs switch_cost, so one ambiguous frame can't
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
        cost = np.array([0.0, stay_cost])
        back = np.zeros((n, 2), int)
        for k in range(1, n):
            a, b = frames[k - 1], frames[k]
            same, cross = move(a, b, False), move(a, b, True)
            new = np.empty(2)
            for s in (0, 1):
                # from state p at a to state s at b: relative flip iff p != s
                opts = [cost[p] + (same if p == s else cross + switch_cost) for p in (0, 1)]
                back[k, s] = int(np.argmin(opts))
                new[s] = min(opts) + stay_cost * s
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


def body_size(lm2: np.ndarray, lm3: np.ndarray, min_bone_m: float = 0.1) -> np.ndarray:
    """[T] on-screen size per metre: the largest px / m over bones. Projection can
    only shorten a bone, so the least foreshortened one gives the true scale."""
    a, b = BONE_PAIRS[:, 0], BONE_PAIRS[:, 1]
    l2 = np.linalg.norm(lm2[:, b] - lm2[:, a], axis=-1)
    l3 = np.linalg.norm(lm3[:, b] - lm3[:, a], axis=-1)
    with np.errstate(all="ignore"):
        r = np.where(l3 > min_bone_m, l2 / l3, np.nan)
    out = np.full(len(r), np.nan)
    ok = np.isfinite(r).any(1)
    out[ok] = np.nanmax(r[ok], axis=1)
    return out


def shrunk_frames(size: np.ndarray, fps: float, ratio: float, window_s: float) -> np.ndarray:
    """[T] frames whose body size is below 1/ratio of the surrounding window's median."""
    ok = np.isfinite(size)
    if ok.sum() < 3:
        return np.zeros(len(size), bool)
    filled = np.interp(np.arange(len(size)), np.nonzero(ok)[0], size[ok])
    ref = median_filter(filled, size=max(3, int(round(2 * window_s * fps)) | 1), mode="nearest")
    return ok & (size < ref / ratio)


def torso_lean(depth: dict, lm3_cam: np.ndarray) -> dict:
    """Torso lean toward the camera (deg, + = top nearer) per 1c frame, from
    Depth Pro's torso surface and from MediaPipe's trunk; camera-aligned axes."""
    dp, mp = [], []
    for k, t in enumerate(depth["frames"]):
        pts = depth["torso_points_cam"][k]
        pts = pts[np.isfinite(pts).all(1)]
        u = lm3_cam[t, J["neck"]] - lm3_cam[t, J["pelvis"]]
        if len(pts) < 10 or not np.isfinite(u).all():
            continue
        A = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
        b = np.linalg.lstsq(A, pts[:, 2], rcond=None)[0][1]   # dz / dy on the surface
        dp.append(np.degrees(np.arctan(b)))
        mp.append(np.degrees(np.arctan2(-u[2], -u[1])))
    dp, mp = np.asarray(dp), np.asarray(mp)
    if not len(dp):
        return dict(n_frames=0)
    diff = dp - mp
    return dict(n_frames=int(len(dp)), depth_deg_median=float(np.median(dp)),
                mediapipe_deg_median=float(np.median(mp)), diff_deg_median=float(np.median(diff)),
                diff_deg_spread=float(np.percentile(diff, 75) - np.percentile(diff, 25)))


UPPER_BODY = [J[n] for n in ("neck", "head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
                             "l_wrist", "r_wrist")]


def rotate_upper_body(lm3: np.ndarray, deg: float) -> np.ndarray:
    """Lean the upper body toward the camera by deg about the pelvis (camera x axis)."""
    a = np.radians(deg)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    out = lm3.copy()
    pel = lm3[:, J["pelvis"]][:, None]
    out[:, UPPER_BODY] = (lm3[:, UPPER_BODY] - pel) @ R.T + pel
    out[:, J["spine"]] = 0.5 * (out[:, J["pelvis"]] + out[:, J["neck"]])
    return out


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


def clean_arrays(cap: dict, depth: dict | None = None, cfg: CleanConfig | None = None) -> dict:
    cfg = cfg or CleanConfig()
    fps = float(cap["fps"])
    W, H = float(cap["width"]), float(cap["height"])
    T = len(cap["landmarks"])
    max_gap = int(round(cfg.max_gap_s * fps))

    # 1. map; shrunk-body frames count as undetected
    lm3 = map_mediapipe(cap["landmarks"]).astype(float)
    lm2 = map_mediapipe(cap["landmarks_2d"]).astype(float)
    vis = map_mediapipe(cap["confidence"], "min").astype(float)
    pres = map_mediapipe(cap["presence"], "min").astype(float)
    size = body_size(lm2, lm3, cfg.shrink_min_bone_m)
    shrunk = shrunk_frames(size, fps, cfg.shrink_ratio, cfg.shrink_window_s)
    lm3[shrunk] = np.nan
    lm2[shrunk] = np.nan
    vis[shrunk] = 0.0
    pres[shrunk] = 0.0
    lean = torso_lean(depth, lm3) if depth is not None and "torso_points_cam" in depth else dict(n_frames=0)
    lean["applied_deg"] = 0.0
    if cfg.lean_fix and lean["n_frames"]:
        lm3 = rotate_upper_body(lm3, lean["diff_deg_median"])
        lean["applied_deg"] = lean["diff_deg_median"]
    raw3 = lm3.copy()
    detected = np.isfinite(lm2).all(-1) & np.isfinite(lm3).all(-1)
    undetected = ~detected.any(1)

    # 2. swaps
    scale_px, _ = _pair_scale(lm2, detected)
    if not np.isfinite(scale_px):
        scale_px = 0.25 * max(W, H)
    scale_m, _ = _pair_scale(lm3, detected)
    if not np.isfinite(scale_m):
        scale_m = 0.5
    swapped = fix_swaps(lm2, lm3, [vis, pres], scale_px, scale_m, cfg.swap_switch_cost, cfg.swap_stay_cost)

    # 3. drop
    m = cfg.edge_margin
    in_frame = detected & (lm2[..., 0] >= m * W) & (lm2[..., 0] <= (1 - m) * W) \
        & (lm2[..., 1] >= m * H) & (lm2[..., 1] <= (1 - m) * H)
    visible = detected & (vis >= cfg.vis_thresh)
    occluded = detected & ~visible & (pres >= cfg.presence_thresh) & in_frame
    weak = (visible | occluded).mean(0) < cfg.min_kept_frac
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
    lm3[..., :2] = lowpass(lm3[..., :2], fps, cfg.smooth_hz)
    lm3[..., 2:] = lowpass(lm3[..., 2:], fps, cfg.smooth_depth_hz)
    have = keep | filled

    # feet count only when actually seen: a guess under a blanket or a table isn't a planted foot
    feet_frac = (keep & ~occluded)[:, FEET].any(1).mean()
    has_feet = bool(feet_frac >= cfg.min_kept_frac)

    # 5. floor direction: depth > feet > none
    n_cam, floor_source = None, "none"
    if depth is not None and bool(depth.get("floor_found", False)):
        n_cam, floor_source = np.asarray(depth["floor_normal_clip"], float), "depth"
    elif has_feet:
        n_cam = floor_from_feet(lm3, have, cfg.free_floor)
        floor_source = "feet" if n_cam is not None else "none"
    if n_cam is None:
        n_cam = np.array([0.0, -1.0, 0.0])
    R = plane_basis(n_cam)  # camera -> world
    pos = lm3 @ R.T
    pos_raw = raw3 @ R.T

    # 7. hip motion from the 2D hip track (pelvis, or neck if hips never seen)
    root_joint = "pelvis" if keep[:, J["pelvis"]].mean() >= cfg.min_kept_frac else "neck"
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
        band = bandpass(track, fps, *cfg.hip_band_hz)
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
            contacts[:, c] = mode_filter(ok & (z - ground < cfg.contact_height_m) & (v < cfg.contact_speed_ms), 5)

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
        undetected=undetected, shrunk=shrunk, body_size_px_per_m=size, lean=lean,
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


def summary(out: dict, depth: dict | None, cfg: CleanConfig, path: Path) -> dict:
    """What clean_summary.json records: counts, depth reference, lean stats, config."""
    ref = dict(used=depth is not None)
    if depth is not None:
        ok = np.asarray(depth["floor_valid"], bool)
        ref.update(floor_found=bool(depth["floor_found"]), floor_frames=f"{int(ok.sum())}/{len(ok)}",
                   camera_pitch_deg=float(np.nanmedian(depth["camera_pitch_deg"])) if ok.any() else None,
                   camera_height_m=float(depth["floor_d_clip"]) if ok.any() else None,
                   body_distance_m=float(np.nanmedian(depth["pelvis_distance_m"]))
                   if np.isfinite(depth["pelvis_distance_m"]).any() else None,
                   focal_px=float(np.nanmedian(depth["focal_px"])))
    return dict(
        file=path.name, frames=int(len(out["positions"])), tracked_joints=int(out["tracked"].sum()),
        has_feet=bool(out["has_feet"]), floor_source=str(out["floor_source"]),
        vertical_source=str(out["vertical_source"]), height_m=round(float(out["height_m"]), 3),
        undetected_frames=int(out["undetected"].sum()), shrunk_frames=int(out["shrunk"].sum()),
        dropped=round(float(out["dropped"].mean()), 4), filled=round(float(out["filled"].mean()), 4),
        occluded=round(float(out["occluded"].mean()), 4), inferred=round(float(out["inferred"].mean()), 4),
        swapped_frames=int(out["swapped"].any(1).sum()),
        depth_reference=ref, lean=out["lean"], config=dataclasses.asdict(cfg))


def run(clip: schema.Clip, use_depth: bool = True, cfg: CleanConfig | None = None, plots: bool = True) -> dict:
    cfg = cfg or CleanConfig.load(clip_config_path(clip))
    cap = schema.load(clip.capture)
    depth = schema.load(clip.depth) if use_depth and clip.depth.exists() else None
    out = clean_arrays(cap, depth, cfg)
    path = clip / ("clean_depth.npz" if depth is not None else "clean.npz")
    lean = out.pop("lean")
    schema.save(path, cap["fps"], JOINT_NAMES, **out)
    out["lean"] = lean
    summ = summary(out, depth, cfg, path)
    (clip / "clean_summary.json").write_text(json.dumps(summ, indent=2))
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
          f"inferred {out['inferred'].mean():.1%}  undetected frames {out['undetected'].sum()} "
          f"(shrunk {out['shrunk'].sum()})")
    lean = out["lean"]
    if lean.get("n_frames"):
        print(f"  torso lean vs depth: {lean['diff_deg_median']:+.1f} deg (applied {lean['applied_deg']:+.1f})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--no-depth", action="store_true", help="ignore depth.npz even if present")
    ap.add_argument("--free-floor", action="store_true", default=None, help="feet floor fit with sideways tilt")
    ap.add_argument("--lean-fix", action="store_true", default=None, help="match the torso lean to Depth Pro")
    ap.add_argument("--config", help="CleanConfig JSON (default configs/<clip>/clean.json)")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    clip = schema.Clip(a.clip)
    cfg = CleanConfig.load(a.config or clip_config_path(clip), free_floor=a.free_floor, lean_fix=a.lean_fix)
    run(clip, not a.no_depth, cfg, not a.no_plots)


if __name__ == "__main__":
    main()
