import numpy as np
import pytest

from core import fit2d
from core.clean import clean_arrays
from core.depth import find_floor, plane_basis
from core.joint_depth import clip_scale, surface_depth
from core.skeleton import (GROUPS, J, N_JOINTS, RATIO_VEC, build_positions, extract_dirs,
                           group_lengths)
from tests import synth


def mpjpe(a, b):
    """Mean per-joint error after aligning pelvises, metres."""
    a = a - a[:, J["pelvis"]][:, None]
    b = b - b[:, J["pelvis"]][:, None]
    return float(np.linalg.norm(a - b, axis=-1).mean())


def test_fk_roundtrip():
    pos, glen = synth.motion(T=20)
    rebuilt = build_positions(pos[:, J["pelvis"]], extract_dirs(pos), group_lengths(pos))
    assert np.allclose(rebuilt, pos, atol=1e-9)
    assert np.allclose(group_lengths(pos), glen)


def test_plane_basis_level_camera():
    R = plane_basis(np.array([0.0, -1.0, 0.0]))
    assert np.allclose(R @ [1, 0, 0], [1, 0, 0])   # right stays right
    assert np.allclose(R @ [0, 0, 1], [0, 1, 0])   # camera forward -> +Y
    assert np.allclose(R @ [0, -1, 0], [0, 0, 1])  # camera up -> +Z


def test_find_floor_picks_lowest_level_surface():
    R_cw, C = synth.camera(pitch_deg=15, pos=(0, -3, 1.4))
    h, w, f = 120, 160, 150.0
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    rays_cam = np.stack([(u - (w - 1) / 2) / f, (v - (h - 1) / 2) / f, np.ones_like(u, float)], -1)
    rays_w = rays_cam @ R_cw
    # floor z=0 everywhere, plus a "table top" at z=0.7 occupying part of the view
    tz = np.where(rays_w[..., 2] < 0, -C[2] / rays_w[..., 2], 50.0)
    hit = C + tz[..., None] * rays_w
    table = (hit[..., 0] > 0.3) & (hit[..., 1] > 1.0) & (hit[..., 1] < 2.5)
    tt = np.where(rays_w[..., 2] < 0, (0.7 - C[2]) / rays_w[..., 2], 50.0)
    t = np.where(table, tt, tz)
    pts = (t[..., None] * rays_w) @ R_cw.T
    fl = find_floor(pts, np.zeros((h, w), bool))
    assert fl is not None
    assert fl["d"] == pytest.approx(1.4, abs=0.03)
    assert fl["pitch_deg"] == pytest.approx(15, abs=1.0)


def _synthetic_capture(T=150, seed=0, depth_squash=0.5):
    rng = np.random.default_rng(seed)
    pos, glen = synth.motion(T=T, seed=seed)
    R_cw, C = synth.camera()
    P = synth.to_camera(pos, R_cw, C)
    lm3, lm2 = synth.to_mediapipe(P, rng, depth_squash)
    lm2 = lm2 + rng.normal(0, 2.0, lm2.shape)
    cap = dict(landmarks=lm3, landmarks_2d=lm2, confidence=np.full((T, 33), 0.95),
               presence=np.full((T, 33), 0.99), width=synth.W, height=synth.H, fps=synth.FPS)
    return cap, pos, glen, P, R_cw


def test_clean_fixes_swap_and_spike_and_finds_floor():
    # unsquashed MediaPipe depth here: squashed depth biases the feet floor pitch
    # (the reason depth beats feet for floor direction)
    cap, pos, glen, P, R_cw = _synthetic_capture(depth_squash=1.0)
    lm2, lm3 = cap["landmarks_2d"], cap["landmarks"]
    for a in (lm2, lm3):  # swap left/right ankles+heels+toes+knees at frame 60
        a[60, [25, 27, 29, 31, 26, 28, 30, 32]] = a[60, [26, 28, 30, 32, 25, 27, 29, 31]].copy()
    lm2[90, 15] += 300  # spike on the left wrist
    out = clean_arrays(cap)
    assert out["swapped"][60, J["l_ankle"]]
    assert not out["swapped"][[59, 61]].any()
    assert out["dropped"][90, J["l_wrist"]]
    assert np.isfinite(out["positions"]).all()
    assert out["has_feet"] and out["floor_source"] == "feet"
    assert out["tracked"].all() and not out["inferred"].any()
    # floor from feet RANSAC should recover the 12 degree camera pitch roughly
    n = out["floor_normal_cam"]
    assert -np.degrees(np.arcsin(n[2])) == pytest.approx(12, abs=4)
    # planted ankles at ankle height
    lower = np.minimum(out["positions"][:, J["l_ankle"], 2], out["positions"][:, J["r_ankle"], 2])
    assert np.median(lower) == pytest.approx(0.039 * out["height_m"], abs=0.01)
    # 2b: standard proportions, directions kept
    g = group_lengths(out["positions_proportional"])
    assert np.allclose(g, RATIO_VEC * out["height_m"], rtol=1e-6)
    d0, d1 = extract_dirs(out["positions"]), extract_dirs(out["positions_proportional"])
    assert np.nanmax(np.linalg.norm(d0 - d1, axis=-1)) < 1e-6


def test_clean_infers_never_seen_legs():
    cap, *_ = _synthetic_capture(T=90)
    legs = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    cap["confidence"][:, legs] = 0.05
    cap["presence"][:, legs] = 0.05  # off the bottom of the frame
    cap["confidence"][:, [23, 24]] = 0.9  # hips still seen
    cap["presence"][:, [23, 24]] = 0.99
    out = clean_arrays(cap)
    assert not out["has_feet"]
    for n in ("l_knee", "r_ankle", "l_toe"):
        assert out["inferred"][:, J[n]].all()
    assert not out["inferred"][:, J["l_wrist"]].any()
    assert np.isfinite(out["positions"]).all()
    assert out["positions"][..., 2].min() >= -0.04  # one offset per clip; legs ride the hip bob


def _fit_inputs(with_depth=True, T=120, seed=1):
    rng = np.random.default_rng(seed)
    pos, glen = synth.motion(T=T, seed=seed)
    R_cw, C = synth.camera()
    P = synth.to_camera(pos, R_cw, C)
    uv = synth.project(P) + rng.normal(0, 2.0, (T, N_JOINTS, 2))
    # a MediaPipe-like 2b: right directions in the image plane, squashed depth, wrong scale
    d_cam = extract_dirs(P)
    d_cam[..., 2] *= 0.4
    d_cam += rng.normal(0, 0.05, d_cam.shape)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    H_est = 1.70 * 1.06
    glen_2b = RATIO_VEC * H_est
    pos_2b = build_positions(np.zeros((T, 3)), d_cam @ R_cw, glen_2b)
    pos_2b += pos[:, J["pelvis"]][:, None]
    x = fit2d.FitInputs(uv=uv, w2d=np.ones((T, N_JOINTS)), focal=synth.FOCAL,
                        center=np.array([(synth.W - 1) / 2, (synth.H - 1) / 2]), pos_2b=pos_2b,
                        floor_R=R_cw.T, inferred=np.zeros((T, N_JOINTS), bool),
                        tracked=np.ones(N_JOINTS, bool), fps=synth.FPS,
                        init_dist=np.full(T, 3.0), occluded=np.zeros((T, N_JOINTS), bool))
    if with_depth:
        f = np.arange(0, T, 2)
        # Depth Pro-like: skin is ~5 cm in front of the joint, plus an absolute bias and noise
        x.depth_frames = f
        x.z_surface = P[f, :, 2] - 0.05 + 0.4 + rng.normal(0, 0.015, (len(f), N_JOINTS))
        x.on_mask = np.ones((len(f), N_JOINTS), bool)
        x.in_frame = np.ones((len(f), N_JOINTS), bool)
    return x, P, pos


@pytest.mark.parametrize("with_depth", [True, False])
def test_fit2d_recovers_pose(with_depth):
    x, P_true, pos_true = _fit_inputs(with_depth)
    out = fit2d.solve(x, static=True, has_feet=True)
    s = out["summary"]
    assert s["after_fit2d"]["px_2d_median_visible"] < 3.5
    assert s["after_fit2d"]["px_2d_median_visible"] < s["before_rigid_2b"]["px_2d_median_visible"]
    err_2b = mpjpe(x.pos_2b, pos_true)
    err = mpjpe(out["positions_fit2d_cam"], P_true)
    print(f"MPJPE 2b {err_2b * 100:.1f} cm -> fit2d {err * 100:.1f} cm")
    if with_depth:
        assert err < 0.04 and err < 0.4 * err_2b
    else:  # 2D alone cannot fix MediaPipe's depth; it must not make it worse
        assert err < err_2b
    if with_depth:
        assert s["after_fit2d"]["rel_depth_cm_median_visible"] < 3.0
    # placement: planted ankles at ankle height of the fitted body
    W = out["positions_fit2d"]
    height = 2 * group_lengths(W)[GROUPS.index("spine")] / 0.288
    lower = np.fmin(W[:, J["l_ankle"], 2], W[:, J["r_ankle"], 2])
    assert np.median(lower) == pytest.approx(0.039 * height, abs=0.005)


def test_clean_no_false_swap_when_arms_overlap_in_2d():
    # side-view case: near and far arm overlap in the image for a few frames;
    # MediaPipe's labels stay right, so nothing may be swapped (and no run may start)
    cap, *_ = _synthetic_capture(depth_squash=1.0)
    lm2 = cap["landmarks_2d"]
    for t in range(70, 74):
        mid = 0.5 * (lm2[t, [13, 15]] + lm2[t, [14, 16]])
        lm2[t, [13, 15]] = mid + 1.0
        lm2[t, [14, 16]] = mid - 1.0
    out = clean_arrays(cap)
    assert not out["swapped"][:, [J["l_elbow"], J["l_wrist"], J["r_elbow"], J["r_wrist"]]].any()


def test_surface_depth_on_person_only():
    depth = np.full((100, 100), 2.0, np.float32)
    depth[:, 50:] = 1.0                          # a cup in front, right half
    person = np.zeros((100, 100), bool)
    person[:, :40] = True
    uv = np.array([[20.0, 50], [80, 50], [-5, 50]])
    z, on, inside = surface_depth(depth, person, uv)
    assert z[0] == pytest.approx(2.0) and on[0]
    assert np.isnan(z[1]) and not on[1]          # patch misses the person: no reading
    assert np.isnan(z[2]) and not inside[2]


def test_clip_scale_recovers_factor():
    rng = np.random.default_rng(0)
    s2 = rng.uniform(0.8, 1.2, (8, 30, 40)).astype(np.float32)
    masks = np.zeros_like(s2, bool)
    masks[:, 10:20, 15:25] = True
    ref = np.where(masks, 3.7 * s2, 9.0)         # background disagrees; only the body counts
    masks[0] = False                             # a frame without the person is skipped
    assert clip_scale(ref, s2, masks) == pytest.approx(3.7, rel=1e-4)


def test_shrunk_body_frames_are_dropped():
    # a pass-by close to the camera: MediaPipe squeezes the whole person into the visible part
    cap, *_ = _synthetic_capture(T=120)
    lm2 = cap["landmarks_2d"]
    hips = 0.5 * (lm2[:, 23] + lm2[:, 24])
    lm2[50:58] = hips[50:58, None] + 0.4 * (lm2[50:58] - hips[50:58, None])
    out = clean_arrays(cap)
    assert out["shrunk"][50:58].all() and out["shrunk"].sum() == 8
    assert out["undetected"][50:58].all()
    # a real turn only narrows the body ~1.4x at most: that must survive
    cap, *_ = _synthetic_capture(T=120)
    lm2 = cap["landmarks_2d"]
    hips = 0.5 * (lm2[:, 23] + lm2[:, 24])
    lm2[50:58, :, 0] = hips[50:58, None, 0] + (lm2[50:58, :, 0] - hips[50:58, None, 0]) / 1.4
    assert not clean_arrays(cap)["shrunk"].any()


def test_clean_config_json_override(tmp_path):
    from core.clean import CleanConfig

    p = tmp_path / "clean.json"
    p.write_text('{"vis_thresh": 0.3, "hip_band_hz": [0.5, 5.0]}')
    cfg = CleanConfig.load(p, lean_fix=True, free_floor=None)
    assert cfg.vis_thresh == 0.3 and cfg.hip_band_hz == (0.5, 5.0) and cfg.lean_fix and not cfg.free_floor
    p.write_text('{"vis_tresh": 0.3}')
    with pytest.raises(ValueError, match="unknown"):
        CleanConfig.load(p)
    assert CleanConfig.load(tmp_path / "missing.json") == CleanConfig()


def test_background_motion_takes_fastest_band(tmp_path):
    import cv2

    # distant buildings (top two thirds) still, nearby ground (bottom third) sliding 6 px/frame
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.uniform(0, 255, (240, 1200)).astype(np.uint8), (5, 5), 0)
    path = tmp_path / "dolly.mp4"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for t in range(40):
        img = tex[:, :320].copy()
        img[160:] = tex[160:, 6 * t:6 * t + 320]
        wr.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    wr.release()
    uv = np.full((40, N_JOINTS, 2), [160.0, 120.0]) + np.linspace(-5, 5, N_JOINTS)[None, :, None]
    motion, bands = fit2d.background_motion(path, uv, n_pairs=12)
    assert bands[0] < 0.5 and bands[1] < 0.5
    assert motion == pytest.approx(6.0, abs=0.7)


def _depth_case():
    x, P, pos = _fit_inputs(True, T=40)
    x.z_fallback = np.full_like(x.z_surface, np.nan)
    return x, P


def test_impossible_depth_marks_nearer_end():
    x, P = _depth_case()
    z = x.z_surface.copy()
    k, w, e = 3, J["l_wrist"], J["l_elbow"]
    z[k, w] = z[k, e] - 0.6          # forearm ~25 cm long cannot reach 60 cm toward the camera
    imp = fit2d.impossible_depth(x, z, np.isfinite(z))
    assert imp[k, w] and not imp[k, e]
    assert imp[k].sum() == 1
    assert imp.mean() < 0.01      # noise alone rarely trips it


def test_near_hand_and_interpolation_carry_cover():
    x, P = _depth_case()
    x.hand_uv = np.stack([x.uv[:, J["l_wrist"]], x.uv[:, J["r_wrist"]]], 1)
    near = fit2d.near_hand(x, P)
    assert near[:, J["l_wrist"]].all() and not near[:, J["head"]].any()
    f = np.array([0, 4, 8])
    z = np.ones((3, N_JOINTS))
    w_vis = np.ones((3, N_JOINTS))
    w_cov = np.zeros((3, N_JOINTS))
    w_vis[1, 5], w_cov[1, 5] = 0.0, 1.0   # joint 5 covered at the middle sample
    ff, zz, wv, wc = fit2d.interpolate_rows(f, z, w_vis, w_cov)
    assert list(ff) == list(range(9))
    assert (wc[1:8, 5] > 0).all() and (wv[1:8, 5] == 0).all()     # covered carries both ways
    assert (wv[:, 6] > 0).all()
    assert wv.sum() + wc.sum() == pytest.approx(w_vis.sum() + w_cov.sum())  # no extra evidence


def test_image_widths_facing_vs_side():
    x, P = _depth_case()
    w = fit2d.image_widths(x, P)
    assert w["shoulder"]["used"] and w["hip"]["used"]
    true = np.linalg.norm(P[:, J["l_shoulder"]] - P[:, J["r_shoulder"]], axis=-1).max()
    assert w["shoulder"]["image_width_m"] == pytest.approx(true, rel=0.08)
    # side view: the shoulders overlap on screen, so the ratio width stays
    x.uv[:, J["r_shoulder"]] = x.uv[:, J["l_shoulder"]] + 3.0
    assert not fit2d.image_widths(x, P)["shoulder"]["used"]
