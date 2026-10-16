import json

import numpy as np
import pytest

from core import bodygen, env_extract, playback, retarget
from core.skeleton import D, DIRS, GROUPS, J, JOINT_NAMES, RATIO_VEC, build_positions, normalize
from tests import synth

mujoco = pytest.importorskip("mujoco")


def _body(tmp_path, glen):
    spec = bodygen.spec_from_lengths("synth", dict(zip(GROUPS, glen)))
    xml = tmp_path / "body.xml"
    xml.write_text(bodygen.mjcf(spec))
    return xml, spec


def test_bodygen_sites_masses_and_ranges(tmp_path):
    xml, spec = _body(tmp_path, RATIO_VEC * 1.70)
    m = mujoco.MjModel.from_xml_path(str(xml))
    for n in JOINT_NAMES:
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) >= 0
    assert sum(m.body_mass) == pytest.approx(22 * 1.70 ** 2, rel=0.01)   # BMI 22
    assert spec["height_m"] == pytest.approx(1.70, abs=0.01)
    knee = m.joint("l_knee")
    assert np.degrees(m.jnt_range[knee.id]) == pytest.approx([0, 150])
    assert all(m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE for i in range(1, m.njnt))
    # rest pose: sites sit where the skeleton puts them (standing, facing +Y, right = +X)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    site = lambda n: d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)]
    assert site("l_hip")[0] < site("r_hip")[0]
    assert site("l_toe")[1] > site("l_heel")[1]
    assert min(site(f"l_sole{c}")[2] for c in range(4)) == pytest.approx(0.0, abs=1e-6)


def test_retarget_tracks_synthetic_walk(tmp_path):
    pos, glen = synth.motion(T=60)
    pos = pos @ np.diag([-1.0, -1.0, 1.0])   # face +Y like the rest pose; any heading works
    xml, _ = _body(tmp_path, glen)
    out = retarget.retarget(xml, pos, np.zeros(pos.shape[:2], bool), synth.FPS, verbose=False)
    res = out["ik_residual"]
    assert np.median(res) < 0.01
    assert np.percentile(res[:, [J[n] for n in ("pelvis", "l_ankle", "r_wrist", "head")]], 95) < 0.03
    assert not out["joint_pops"].any()
    shapes = playback.body_shapes(mujoco.MjModel.from_xml_path(str(xml)), out["qpos"])
    assert shapes["geom_xpos"].shape[:2] == (60, len(shapes["geom_names"]))
    feet = env_extract.part_bounds({**shapes, "fps": synth.FPS})["feet"][0]
    assert feet[:, 2].min() > -0.005        # sole corners never under the floor


def _sitting(T=30, H=1.70):
    up, left, fwd = np.array([0, 0, 1.0]), np.array([-1.0, 0, 0]), np.array([0, 1.0, 0])
    dirs = np.zeros((T, len(DIRS), 3))
    dirs[:, D["spine"]] = up
    dirs[:, D["head"]] = up
    dirs[:, D["shoulder_axis"]] = left
    dirs[:, D["hip_axis"]] = left
    for s in "lr":
        dirs[:, D[f"{s}_thigh"]] = fwd
        dirs[:, D[f"{s}_shin"]] = -up
        dirs[:, D[f"{s}_foot"]] = normalize(fwd - 0.34 * up)
        dirs[:, D[f"{s}_heel"]] = normalize(-0.5 * fwd - up)
        dirs[:, D[f"{s}_upper_arm"]] = -up
        dirs[:, D[f"{s}_forearm"]] = fwd
    glen = RATIO_VEC * H
    a = 0.039 * H
    pel = np.tile([0.0, 0.0, a + glen[GROUPS.index("shin")]], (T, 1))
    return build_positions(pel, dirs, glen), glen


def test_code_props_finds_a_seat(tmp_path):
    pos, glen = _sitting()
    xml, spec = _body(tmp_path, glen)
    out = retarget.retarget(xml, pos, np.zeros(pos.shape[:2], bool), 30.0, verbose=False)
    shapes = {**playback.body_shapes(mujoco.MjModel.from_xml_path(str(xml)), out["qpos"]), "fps": 30.0}
    props = env_extract.code_props(None, shapes, {"activity": np.full(len(pos), "sit")}, spec)
    assert [p["name"] for p in props] == ["seat0"]
    seat = props[0]
    top = seat["center"][2] + seat["half_size"][2]
    hips = env_extract.part_bounds(shapes)["hips"][0][:, 2]
    assert top == pytest.approx(np.median(hips), abs=0.01)
    assert 0.3 < top < 0.5
    pen = env_extract.penetration(env_extract.scene_xml(xml, props), out["qpos"])
    assert max(pen.values()) < env_extract.SINK_TOL_M     # the shins hang in front of it


CUBE_FACES = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
              [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]


def test_sandbox_runs_valid_code_and_blocks_the_rest(tmp_path):
    pos, glen = synth.motion(T=10)
    xml, _ = _body(tmp_path, glen)
    out = retarget.retarget(xml, pos, np.zeros(pos.shape[:2], bool), synth.FPS, verbose=False)
    shapes = playback.body_shapes(mujoco.MjModel.from_xml_path(str(xml)), out["qpos"])
    src = f"""
def build():
    v = [[x, y, z] for x in (0.0, 0.08) for y in (0.0, 0.08) for z in (0.0, 0.1)]
    v = (np.array(v) + np.array([0.3, 0.0, 0.8])).tolist()
    return [dict(name="cup", kind="held", attach="r_forearm", frame=2, vertices=v, faces={CUBE_FACES})]
"""
    props = env_extract.check_props(env_extract.run_sandboxed(src), shapes)
    assert props[0]["attach"] == "r_forearm" and len(props[0]["vertices"]) == 8
    json.dumps(props)
    for bad in ("import os\ndef build(): return []", "def build(): return open('x').read()",
                "def build(): return ().__class__", "def build():\n    while True: pass"):
        with pytest.raises((ValueError, RuntimeError)):
            env_extract.run_sandboxed(bad)
    with pytest.raises(ValueError, match="not a body part"):
        env_extract.check_props([dict(kind="held", attach="world", vertices=[[0, 0, 0]] * 4,
                                      faces=[[0, 1, 2]])], shapes)
