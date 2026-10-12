"""Fill never-seen joints and long gaps with a flagged prior (inferred [T,J]).

Never as capture: metrics use captured joints only.
  - Tracked joints with long gaps: interpolated across (edges held).
  - Legs never seen: body-relative hip flexion/abduction + knee flexion from
    activity rules (lie / sit / stand from geometry), attached to the captured
    pelvis frame. Inferred legs are rigidly swung above the support plane.
  - Hidden wrist continues the upper arm; hidden elbow hangs along the trunk.
Inferred bone lengths = anthropometric ratio x height from trunk.
"""

from __future__ import annotations

import numpy as np

from core.filters import fill_gaps, hold_edges, mode_filter
from core.skeleton import J, LEGS, RATIOS, normalize

# activity -> (hip flexion, hip abduction, knee flexion) in degrees
LEG_POSE = {"stand": (0.0, 4.0, 0.0), "sit": (90.0, 8.0, 90.0), "lie": (0.0, 4.0, 0.0)}
LIE_TILT_DEG = 60
SIT_PELVIS_RATIO = 0.40  # pelvis below 0.40 H -> sitting


def body_frame(pos: np.ndarray):
    """Per-frame (up, left, forward) unit vectors of the trunk."""
    up = normalize(pos[:, J["neck"]] - pos[:, J["pelvis"]])
    lat = pos[:, J["l_hip"]] - pos[:, J["r_hip"]]
    sh = pos[:, J["l_shoulder"]] - pos[:, J["r_shoulder"]]
    lat = np.where(np.isfinite(lat).all(1, keepdims=True), lat, sh)
    lat = lat - (lat * up).sum(1, keepdims=True) * up
    left = normalize(lat)
    fwd = np.cross(left, up)
    return up, left, fwd


def classify_activity(pos: np.ndarray, height: float, height_known: bool, fps: float) -> np.ndarray:
    up, _, _ = body_frame(pos)
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
    act = np.full(len(pos), "sit", dtype="<U5")
    if height_known:
        act[pos[:, J["pelvis"], 2] >= SIT_PELVIS_RATIO * height] = "stand"
    act[tilt > LIE_TILT_DEG] = "lie"
    size = max(1, int(fps)) | 1
    return mode_filter(act, size)


def _leg_dirs(act, up, left, fwd, side: float):
    """Thigh, shin, heel and foot directions for one side (+1 left, -1 right)."""
    fl, ab, kf = (np.radians(np.array([LEG_POSE[a][k] for a in act]))[:, None] for k in range(3))
    down = -up
    thigh = np.cos(fl) * down + np.sin(fl) * fwd
    thigh = normalize(np.cos(ab) * thigh + np.sin(ab) * side * left)
    shin = normalize(np.cos(fl - kf) * down + np.sin(fl - kf) * fwd + np.sin(ab) * side * left)
    # foot plane follows the shin: toe points forward, heel back, both "down" the shin
    sole = normalize(fwd - (fwd * shin).sum(1, keepdims=True) * shin)
    foot = normalize(0.95 * sole + 0.3 * shin)
    heel = normalize(-0.6 * sole + 0.8 * shin)
    return thigh, shin, heel, foot


def _swing_above_floor(pos, leg, hip, left, inferred_leg):
    """Rotate inferred leg joints about the hip (around the body's left axis)
    by the smallest angle that keeps them above Z = 0."""
    for t in np.nonzero(inferred_leg)[0]:
        pts = pos[t, leg[1:]] - pos[t, hip]
        if pos[t, leg[1:], 2].min() >= 0:
            continue
        axis = left[t]
        best = None
        for deg in sorted(np.arange(-150, 151, 2), key=abs):
            a = np.radians(deg)
            # Rodrigues rotation of pts about axis
            rot = (pts * np.cos(a) + np.cross(axis, pts) * np.sin(a)
                   + np.outer(pts @ axis, axis) * (1 - np.cos(a)))
            if (rot[:, 2] + pos[t, hip, 2]).min() >= 0:
                best = rot
                break
        if best is not None:
            pos[t, leg[1:]] = pos[t, hip] + best


def infer_hidden(pos: np.ndarray, tracked: np.ndarray, height: float, height_known: bool,
                 fps: float):
    """pos [T,20,3] (NaN where not captured), tracked [20] -> filled pos, inferred [T,20], activity."""
    pos = pos.copy()
    T = len(pos)
    inferred = np.zeros((T, pos.shape[1]), bool)

    # tracked joints: long gaps and clip edges
    for j in np.nonzero(tracked)[0]:
        miss = ~np.isfinite(pos[:, j]).all(1)
        if miss.any():
            pos[:, j], _ = fill_gaps(pos[:, j], ~miss, None)
            pos[:, j] = hold_edges(pos[:, j])
            inferred[:, j] |= miss

    L = {g: r * height for g, r in RATIOS.items()}

    def need(name):
        return not tracked[J[name]]

    def put(name, value):
        pos[:, J[name]] = value
        inferred[:, J[name]] = True

    # trunk: fall back on vertical when the pelvis or neck was never seen
    vert = np.array([0.0, 0.0, 1.0])
    if need("pelvis") and need("neck"):
        raise ValueError("neither pelvis nor neck was ever seen; nothing to attach a body to")
    if need("pelvis"):
        put("pelvis", pos[:, J["neck"]] - 2 * L["spine"] * vert)
    if need("neck"):
        put("neck", pos[:, J["pelvis"]] + 2 * L["spine"] * vert)
    if need("spine"):
        put("spine", 0.5 * (pos[:, J["pelvis"]] + pos[:, J["neck"]]))
    up, left, fwd = body_frame(pos)
    if not np.isfinite(left).all():  # no hips and no shoulders: assume facing the camera
        left = np.where(np.isfinite(left), left, np.array([-1.0, 0.0, 0.0]))
        fwd = np.cross(left, up)
    if need("head"):
        put("head", pos[:, J["neck"]] + L["head"] * up)
    for s, sign in (("l", 1), ("r", -1)):
        if need(f"{s}_shoulder"):
            put(f"{s}_shoulder", pos[:, J["neck"]] + sign * L["shoulder"] * left)
        if need(f"{s}_elbow"):
            put(f"{s}_elbow", pos[:, J[f"{s}_shoulder"]] - L["upper_arm"] * up)
        if need(f"{s}_wrist"):
            ua = normalize(pos[:, J[f"{s}_elbow"]] - pos[:, J[f"{s}_shoulder"]])
            put(f"{s}_wrist", pos[:, J[f"{s}_elbow"]] + L["forearm"] * ua)

    activity = classify_activity(pos, height, height_known, fps)
    for s, sign in (("l", 1.0), ("r", -1.0)):
        if need(f"{s}_hip"):
            put(f"{s}_hip", pos[:, J["pelvis"]] + sign * L["hip"] * left)
        thigh, shin, heel, foot = _leg_dirs(activity, up, left, fwd, sign)
        if need(f"{s}_knee"):
            put(f"{s}_knee", pos[:, J[f"{s}_hip"]] + L["thigh"] * thigh)
        if need(f"{s}_ankle"):
            put(f"{s}_ankle", pos[:, J[f"{s}_knee"]] + L["shin"] * shin)
        if need(f"{s}_heel"):
            put(f"{s}_heel", pos[:, J[f"{s}_ankle"]] + L["heel"] * heel)
        if need(f"{s}_toe"):
            put(f"{s}_toe", pos[:, J[f"{s}_ankle"]] + L["foot"] * foot)
        leg = LEGS[s]
        rule_leg = ~tracked[leg[1:]].any()
        if rule_leg and height_known:
            _swing_above_floor(pos, leg, leg[0], left, np.ones(T, bool))
    return pos, inferred, activity
