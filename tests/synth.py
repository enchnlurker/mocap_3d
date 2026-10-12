"""Synthetic clip: a known body walking in place, seen by a known pitched camera."""

from __future__ import annotations

import numpy as np

from core.skeleton import D, DIRS, GROUPS, J, MEDIAPIPE_MAP, RATIO_VEC, build_positions, normalize

W, H = 1296, 972
FOCAL = 1000.0
FPS = 30.0


def camera(pitch_deg=12.0, pos=(0.0, -3.5, 1.3)):
    p = np.radians(pitch_deg)
    z = np.array([0.0, np.cos(p), -np.sin(p)])
    y = np.array([0.0, -np.sin(p), -np.cos(p)])
    x = np.cross(y, z)
    return np.stack([x, y, z]), np.asarray(pos)  # R_cw (world -> camera), centre


def _rot(v, axis, ang):
    axis = normalize(axis)
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return v * c + np.cross(axis, v) * s + axis * (v * axis).sum(-1, keepdims=True) * (1 - c)


def motion(T=150, height=1.70, seed=0):
    """World positions [T,20,3] of a person facing the camera (-Y), walking in place."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / FPS
    ph = 2 * np.pi * 0.9 * t
    up, left, fwd = np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, -1.0, 0])
    one = np.ones((T, 1))
    dirs = np.zeros((T, len(DIRS), 3))
    dirs[:, D["spine"]] = normalize(up + 0.08 * fwd + 0.02 * np.sin(ph)[:, None] * left)
    dirs[:, D["head"]] = normalize(up + 0.15 * fwd)
    dirs[:, D["shoulder_axis"]] = normalize(left + 0.1 * np.sin(ph)[:, None] * fwd)
    dirs[:, D["hip_axis"]] = normalize(left - 0.1 * np.sin(ph)[:, None] * fwd)
    for s, sg in (("l", 1), ("r", -1)):
        swing = sg * 0.45 * np.sin(ph)                      # hip flexion
        knee = 0.6 * np.clip(-sg * np.sin(ph + 0.6), 0, None)  # bend on the back swing
        thigh = _rot(-up * one, left, swing)
        shin = _rot(thigh, left, -knee)
        dirs[:, D[f"{s}_thigh"]] = thigh
        dirs[:, D[f"{s}_shin"]] = shin
        dirs[:, D[f"{s}_foot"]] = normalize(_rot((0.95 * fwd - 0.3 * up) * one, left, swing - knee))
        dirs[:, D[f"{s}_heel"]] = normalize(_rot((-0.6 * fwd - 0.8 * up) * one, left, swing - knee))
        arm = -sg * 0.5 * np.sin(ph)
        ua = _rot(normalize(-up + 0.15 * sg * left) * one, left, arm)
        dirs[:, D[f"{s}_upper_arm"]] = ua
        dirs[:, D[f"{s}_forearm"]] = _rot(ua, left, np.full(T, 0.5))
    glen = RATIO_VEC * height * (1 + 0.04 * rng.standard_normal(len(GROUPS)))
    pelvis = np.c_[0.03 * np.sin(ph / 2), np.zeros(T), 0.92 + 0.02 * np.cos(2 * ph)]
    pos = build_positions(pelvis, dirs, glen)
    pos[..., 2] -= pos[..., 2].min() - 0.0  # lowest point on the floor
    return pos, glen


def to_camera(pos, R_cw, C):
    return (pos - C) @ R_cw.T


def project(P):
    return FOCAL * P[..., :2] / P[..., 2:3] + np.array([(W - 1) / 2, (H - 1) / 2])


def to_mediapipe(P_cam, rng, depth_squash=0.5):
    """Fake 33-landmark capture: hips-centred camera-aligned 3D with MediaPipe-like
    depth errors, plus exact 2D (noise added by the caller)."""
    T = len(P_cam)
    lm3 = np.zeros((T, 33, 3))
    lm2 = np.zeros((T, 33, 2))
    uv = project(P_cam)
    hips = 0.5 * (P_cam[:, J["l_hip"]] + P_cam[:, J["r_hip"]])
    rel = P_cam - hips[:, None]
    rel[..., 2] *= depth_squash
    for name, idx in MEDIAPIPE_MAP.items():
        if len(idx) == 1:
            lm3[:, idx[0]] = rel[:, J[name]]
            lm2[:, idx[0]] = uv[:, J[name]]
    lat = P_cam[:, J["l_shoulder"]] - P_cam[:, J["r_shoulder"]]
    for k, sgn in ((7, 0.25), (8, -0.25)):  # ears either side of the head
        e = P_cam[:, J["head"]] + sgn * lat * 0.5
        er = e - hips
        er[:, 2] *= depth_squash
        lm3[:, k], lm2[:, k] = er, project(e)
    for k in range(33):
        if not lm3[:, k].any():
            lm3[:, k], lm2[:, k] = lm3[:, 7], lm2[:, 7]
    return lm3, lm2
