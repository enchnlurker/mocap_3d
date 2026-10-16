"""Stage 2b: proportions. Direction-preserving rebuild with realistic lengths.

Body height H = trunk length / 0.288. Every bone is set to a standard fraction
of H, left equal to right, and the skeleton is rebuilt outward from the pelvis
keeping every bone's direction. Lengths become realistic but MediaPipe's
directions (including its depth errors) stay. 2b is the start point and the
reference pose for 2c. Clips with feet get a floor re-snap.

Saved as separate keys (positions untouched):
    positions_proportional [T,20,3], bone_lengths_proportional [19], proportional_height_m
"""

from __future__ import annotations

import numpy as np

from core.skeleton import (ANKLE_HEIGHT_RATIO, DEFAULT_HEIGHT_M, D, J, RATIO_VEC, TRUNK_RATIO,
                           bone_lengths_from_groups, build_positions, extract_dirs, normalize, trunk_length)


HEIGHT_RANGE_M = (1.50, 1.95)    # adult heights; a trunk measured outside gives a clamped height
HEAD_TIP_MAX_DEG = 20.0          # head (neck -> ears) vs the spine: MediaPipe's depth tips it too far


def height_from_trunk(positions: np.ndarray, valid: np.ndarray | None = None) -> float:
    t = trunk_length(positions, valid)
    h = t / TRUNK_RATIO if np.isfinite(t) and t > 0 else DEFAULT_HEIGHT_M
    return float(np.clip(h, *HEIGHT_RANGE_M))


def limit_head_tip(dirs: np.ndarray, max_deg: float = HEAD_TIP_MAX_DEG) -> np.ndarray:
    """Rotate the head direction toward the spine where it tips more than max_deg."""
    dirs = dirs.copy()
    h, s = dirs[:, D["head"]], dirs[:, D["spine"]]
    ang = np.arccos(np.clip((h * s).sum(-1), -1, 1))
    over = np.isfinite(ang) & (ang > np.radians(max_deg))
    if over.any():
        perp = normalize(h[over] - (h[over] * s[over]).sum(-1, keepdims=True) * s[over])
        a = np.radians(max_deg)
        dirs[over, D["head"]] = np.cos(a) * s[over] + np.sin(a) * perp
    return dirs


def ground_offset(positions: np.ndarray, height: float, valid: np.ndarray | None = None) -> float:
    """ONE vertical offset per clip putting the median lower ankle at ankle height."""
    la, ra = positions[:, J["l_ankle"], 2], positions[:, J["r_ankle"], 2]
    if valid is not None:
        la = np.where(valid[:, J["l_ankle"]], la, np.nan)
        ra = np.where(valid[:, J["r_ankle"]], ra, np.nan)
    lower = np.fmin(la, ra)
    lower = lower[np.isfinite(lower)]
    if not len(lower):
        return 0.0
    return float(ANKLE_HEIGHT_RATIO * height - np.median(lower))


def proportional(positions: np.ndarray, height: float, has_feet: bool, valid: np.ndarray | None = None):
    """-> positions_proportional, bone_lengths_proportional [19], group lengths [10]."""
    glen = RATIO_VEC * height
    dirs = limit_head_tip(extract_dirs(positions))
    pos = build_positions(positions[:, J["pelvis"]], dirs, glen)
    if has_feet:
        pos[..., 2] += ground_offset(pos, height, valid)
    return pos, bone_lengths_from_groups(glen), glen
