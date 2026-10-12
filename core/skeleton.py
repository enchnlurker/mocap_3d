"""Canonical 20-joint skeleton (the merge contract, DESIGN 3.1).

Both capture sources fill these joints. Everything after the merge point reads
only this layout. Units are metres, world frame is Z-up (X right, Y forward).

The skeleton is also a parametrisation: a pelvis position, 16 unit directions
and 10 bone-group lengths rebuild all 20 joints (`build_positions`). 2b and 2c
both use it, so shoulder/hip/spine pairs are tied by construction: both
shoulders hang off the neck in opposite directions, both hips off the pelvis,
and the two spine halves are in line, so neck and pelvis stay true midpoints.
"""

from __future__ import annotations

import numpy as np

JOINT_NAMES = [
    "pelvis", "spine", "neck", "head",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
    "l_heel", "r_heel", "l_toe", "r_toe",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
]
J = {name: i for i, name in enumerate(JOINT_NAMES)}
N_JOINTS = len(JOINT_NAMES)

# Bone groups share one length per clip (left = right).
GROUPS = ["spine", "head", "shoulder", "hip", "thigh", "shin",
          "heel", "foot", "upper_arm", "forearm"]
G = {name: i for i, name in enumerate(GROUPS)}

# One direction per frame each. "*_axis" point from the right joint to the left.
DIRS = ["spine", "head", "shoulder_axis", "hip_axis",
        "l_thigh", "r_thigh", "l_shin", "r_shin", "l_heel", "r_heel",
        "l_foot", "r_foot", "l_upper_arm", "r_upper_arm", "l_forearm", "r_forearm"]
D = {name: i for i, name in enumerate(DIRS)}

# (parent, child, group, direction, sign); ordered so parents come first.
BONES = [
    ("pelvis", "spine", "spine", "spine", 1),
    ("spine", "neck", "spine", "spine", 1),
    ("neck", "head", "head", "head", 1),
    ("pelvis", "l_hip", "hip", "hip_axis", 1),
    ("pelvis", "r_hip", "hip", "hip_axis", -1),
    ("l_hip", "l_knee", "thigh", "l_thigh", 1),
    ("r_hip", "r_knee", "thigh", "r_thigh", 1),
    ("l_knee", "l_ankle", "shin", "l_shin", 1),
    ("r_knee", "r_ankle", "shin", "r_shin", 1),
    ("l_ankle", "l_heel", "heel", "l_heel", 1),
    ("r_ankle", "r_heel", "heel", "r_heel", 1),
    ("l_ankle", "l_toe", "foot", "l_foot", 1),
    ("r_ankle", "r_toe", "foot", "r_foot", 1),
    ("neck", "l_shoulder", "shoulder", "shoulder_axis", 1),
    ("neck", "r_shoulder", "shoulder", "shoulder_axis", -1),
    ("l_shoulder", "l_elbow", "upper_arm", "l_upper_arm", 1),
    ("r_shoulder", "r_elbow", "upper_arm", "r_upper_arm", 1),
    ("l_elbow", "l_wrist", "forearm", "l_forearm", 1),
    ("r_elbow", "r_wrist", "forearm", "r_forearm", 1),
]
BONE_NAMES = [f"{p}-{c}" for p, c, *_ in BONES]
BONE_PAIRS = np.array([[J[p], J[c]] for p, c, *_ in BONES])
BONE_GROUP = np.array([G[g] for _, _, g, _, _ in BONES])

# Direction i is measured from joint DIR_FROM[i] to joint DIR_TO[i].
_DIR_ENDS = {
    "spine": ("pelvis", "neck"), "head": ("neck", "head"),
    "shoulder_axis": ("r_shoulder", "l_shoulder"), "hip_axis": ("r_hip", "l_hip"),
}
for _p, _c, _g, _d, _s in BONES:
    _DIR_ENDS.setdefault(_d, (_p, _c))
DIR_FROM = np.array([J[_DIR_ENDS[d][0]] for d in DIRS])
DIR_TO = np.array([J[_DIR_ENDS[d][1]] for d in DIRS])

# Generic adult segment lengths as a fraction of standing height H
# (Drillis & Contini). Shoulder/hip are half-widths, spine is half the trunk.
# Widths are MediaPipe landmark spacing, which sits near the joints, not body breadth:
# hips 0.40 x trunk, shoulders 0.64 x trunk (measured on a frontal walking clip).
# head (neck -> ear midpoint) and foot (ankle -> foot_index) also follow MediaPipe's landmarks.
TRUNK_RATIO = 0.288          # hip-joint height 0.530 H to shoulder height 0.818 H
ANKLE_HEIGHT_RATIO = 0.039
RATIOS = {
    "spine": TRUNK_RATIO / 2, "head": 0.115, "shoulder": 0.64 * TRUNK_RATIO / 2, "hip": 0.40 * TRUNK_RATIO / 2,
    "thigh": 0.245, "shin": 0.246, "heel": 0.045, "foot": 0.115,
    "upper_arm": 0.186, "forearm": 0.146,
}
RATIO_VEC = np.array([RATIOS[g] for g in GROUPS])
DEFAULT_HEIGHT_M = 1.70

LEFT_RIGHT = [(J[n], J["r" + n[1:]]) for n in JOINT_NAMES if n.startswith("l_")]
LEGS = {"l": [J[n] for n in ("l_hip", "l_knee", "l_ankle", "l_heel", "l_toe")],
        "r": [J[n] for n in ("r_hip", "r_knee", "r_ankle", "r_heel", "r_toe")]}
FEET = [J[n] for n in ("l_ankle", "r_ankle", "l_heel", "r_heel", "l_toe", "r_toe")]
CONTACT_JOINTS = [J[n] for n in ("l_heel", "l_toe", "r_heel", "r_toe")]
CONTACT_NAMES = ["l_heel", "l_toe", "r_heel", "r_toe"]

# MediaPipe Pose: 33 landmarks -> canonical joints (mean of the listed indices).
# spine = mean of both hips and both shoulders = halfway between pelvis and neck.
MEDIAPIPE_MAP = {
    "pelvis": (23, 24), "spine": (23, 24, 11, 12), "neck": (11, 12), "head": (7, 8),
    "l_hip": (23,), "r_hip": (24,), "l_knee": (25,), "r_knee": (26,),
    "l_ankle": (27,), "r_ankle": (28,), "l_heel": (29,), "r_heel": (30,),
    "l_toe": (31,), "r_toe": (32,),
    "l_shoulder": (11,), "r_shoulder": (12,), "l_elbow": (13,), "r_elbow": (14,),
    "l_wrist": (15,), "r_wrist": (16,),
}


def map_mediapipe(arr: np.ndarray, reduce: str = "mean") -> np.ndarray:
    """[..., 33, ...] landmarks -> [..., 20, ...] canonical joints along axis 1.

    reduce="mean" for positions, "min" for visibility/presence/masks (a
    midpoint is only as trustworthy as its weakest landmark).
    """
    fn = np.mean if reduce == "mean" else np.min
    return np.stack([fn(arr[:, list(MEDIAPIPE_MAP[n])], axis=1) for n in JOINT_NAMES], axis=1)


def _xp(x):
    if isinstance(x, np.ndarray):
        return np
    import torch
    return torch


def build_positions(pelvis, dirs, group_lengths):
    """Forward kinematics: pelvis [...,3], unit dirs [...,16,3], lengths [10] -> [...,20,3].

    Works for numpy arrays and torch tensors (autograd-friendly).
    """
    xp = _xp(pelvis)
    joints = [None] * N_JOINTS
    joints[J["pelvis"]] = pelvis
    for p, c, g, d, s in BONES:
        joints[J[c]] = joints[J[p]] + s * group_lengths[G[g]] * dirs[..., D[d], :]
    return xp.stack(joints, -2)


def normalize(v, axis=-1, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def extract_dirs(positions: np.ndarray) -> np.ndarray:
    """[...,20,3] -> unit directions [...,16,3] (NaN where an end is NaN)."""
    return normalize(positions[..., DIR_TO, :] - positions[..., DIR_FROM, :])


def group_lengths(positions: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Symmetric median length per bone group [10] (NaN if never measurable).

    valid [T,20] restricts to frames where both ends of a bone are valid.
    """
    seg = np.linalg.norm(positions[:, BONE_PAIRS[:, 1]] - positions[:, BONE_PAIRS[:, 0]], axis=-1)
    if valid is not None:
        seg = np.where(valid[:, BONE_PAIRS[:, 0]] & valid[:, BONE_PAIRS[:, 1]], seg, np.nan)
    out = np.full(len(GROUPS), np.nan)
    for gi in range(len(GROUPS)):
        vals = seg[:, BONE_GROUP == gi].ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[gi] = np.median(vals)
    return out


def bone_lengths_from_groups(glen: np.ndarray) -> np.ndarray:
    """[10] group lengths -> [19] per-bone lengths."""
    return np.asarray(glen)[BONE_GROUP]


def trunk_length(positions: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Median pelvis -> neck distance (NaN if never seen)."""
    t = np.linalg.norm(positions[:, J["neck"]] - positions[:, J["pelvis"]], axis=-1)
    if valid is not None:
        t = np.where(valid[:, J["neck"]] & valid[:, J["pelvis"]], t, np.nan)
    t = t[np.isfinite(t)]
    return float(np.median(t)) if len(t) else float("nan")
