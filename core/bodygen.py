"""Stage 3A body generator: 2c bone lengths -> a MuJoCo body.

    python -m core.bodygen walk01 [--name walk01] [--force-spec]

bodies/<name>/spec.json is the editable source of truth; body.xml (MJCF) is
generated from it. The spec is written from the clip's 2c lengths only when it
doesn't exist yet (or with --force-spec), so hand edits survive re-runs.

The body:
  - tree: one body per bone, hinges only (no ball joints), so every axis gets
    its own human range (knee 0-150 deg, hip flexion -30 to 125 deg, ...).
    Spine: abdomen and chest, three hinges each. Head: three. Hip and shoulder:
    flexion, abduction, rotation. Knee and elbow: one. Ankle: flexion and roll.
  - masses: de Leva segment fractions x a BMI-22 body weight (22 x H^2 kg).
  - shapes: mannequin-like; ellipsoid pelvis, trunk and head, capsule limbs,
    box feet with a site on each sole corner.
  - sites: every canonical joint (core.skeleton) has a site with the same name,
    so 2c targets map one-to-one onto the body.

Rest pose: standing, arms hanging, facing +Y, Z up, so the body's right is +X.
Positive hinge angles are flexion, abduction (away from the midline) and
inward rotation. Lengths, ranges and shapes in metres / degrees.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core import schema
from core.skeleton import ANKLE_HEIGHT_RATIO, BONE_GROUP, GROUPS, JOINT_NAMES, RATIOS

BODIES = schema.ROOT / "bodies"
BMI = 22.0

# de Leva (1996), male segment mass fractions; the hand rides on the forearm.
MASS_FRACTIONS = {
    "pelvis": 0.1117, "abdomen": 0.1633, "chest": 0.1596, "head": 0.0694,
    "upper_arm": 0.0271, "forearm": 0.0162 + 0.0061,
    "thigh": 0.1416, "shin": 0.0433, "foot": 0.0137,
}

# hinge name -> (axis in the rest frame, range in degrees). l_/r_ abduction and
# rotation axes are mirrored (see _side_axis).
JOINTS = {
    "abdomen_flex": ([-1, 0, 0], [-25, 45]), "abdomen_side": ([0, 1, 0], [-25, 25]),
    "abdomen_twist": ([0, 0, 1], [-30, 30]),
    "chest_flex": ([-1, 0, 0], [-25, 45]), "chest_side": ([0, 1, 0], [-25, 25]),
    "chest_twist": ([0, 0, 1], [-30, 30]),
    "neck_flex": ([-1, 0, 0], [-60, 70]), "neck_side": ([0, 1, 0], [-45, 45]),
    "neck_twist": ([0, 0, 1], [-80, 80]),
    "hip_flex": ([1, 0, 0], [-30, 125]), "hip_abd": ([0, 1, 0], [-30, 45]),
    "hip_rot": ([0, 0, -1], [-45, 45]),
    "knee": ([-1, 0, 0], [0, 150]),
    "ankle_flex": ([1, 0, 0], [-50, 30]), "ankle_roll": ([0, 1, 0], [-30, 30]),
    "shoulder_flex": ([1, 0, 0], [-60, 180]), "shoulder_abd": ([0, 1, 0], [-30, 180]),
    "shoulder_rot": ([0, 0, -1], [-90, 90]),
    "elbow": ([1, 0, 0], [0, 150]),
}
# Twist and roll hinges: the targets barely see them (weak neutral pull, speed limit in retarget).
TWIST_JOINTS = ["abdomen_twist", "chest_twist", "neck_twist", "hip_rot", "shoulder_rot", "ankle_roll"]


def _side_axis(name: str, axis, side: str):
    """Mirror abduction / rotation / roll axes for the right side (+X is the body's right)."""
    a = np.asarray(axis, float)
    if side == "r" and name.endswith(("_abd", "_rot", "_roll")):
        a = -a
    return a


# ---------------------------------------------------------------- spec

def spec_from_lengths(name: str, lengths: dict, source: str = "") -> dict:
    """Default spec for group lengths in metres (core.skeleton.GROUPS)."""
    H = float(np.median([lengths[g] / RATIOS[g] for g in GROUPS]))   # robust to one odd bone
    ankle_h = min(ANKLE_HEIGHT_RATIO * H, 0.85 * lengths["heel"])
    return {
        "name": name,
        "source": source,
        "height_m": round(H, 4),
        "bmi": BMI,
        "mass_kg": round(BMI * H ** 2, 2),
        "lengths_m": {g: round(float(lengths[g]), 4) for g in GROUPS},
        "ankle_height_m": round(ankle_h, 4),
        "radii_m": default_radii(lengths, H),
        "mass_fractions": MASS_FRACTIONS,
        "joints": {k: {"axis": a, "range_deg": r} for k, (a, r) in JOINTS.items()},
    }


def default_radii(L: dict, H: float) -> dict:
    """Mannequin shapes (metres) for group lengths L and height H: a rib cage wider
    than the waist, a shoulder girdle, a neck under the head, deltoids, tapered
    thighs and calves, hands. Ellipsoids give [x, y, z] half axes (x = right,
    y = forward, z = up in the rest pose); capsules a radius."""
    r = lambda *v: [round(float(x), 4) for x in v] if len(v) > 1 else round(float(v[0]), 4)
    sp, th, sh = L["spine"], L["thigh"], L["shin"]
    return {
        "pelvis": r(max(L["hip"] + 0.035 * H, 0.090 * H), 0.062 * H, 0.070 * H),
        "abdomen": r(0.080 * H, 0.058 * H, 0.65 * sp),
        "chest": r(max(0.95 * L["shoulder"], 0.092 * H), 0.064 * H, 0.65 * sp),
        "shoulder_girdle": r(0.036 * H),
        "neck": r(0.032 * H),
        "head": r(0.050 * H, 0.058 * H, 0.066 * H),
        "deltoid": r(0.034 * H, 0.036 * H, 0.050 * H),
        "upper_arm": r(0.024 * H), "forearm": r(0.019 * H),
        "hand": r(0.013 * H, 0.027 * H, 0.050 * H),
        "thigh": r(0.034 * H), "thigh_muscle": r(0.046 * H, 0.048 * H, 0.42 * th),
        "shin": r(0.024 * H), "calf": r(0.030 * H, 0.034 * H, 0.32 * sh),
        "foot_half_width": r(0.026 * H),
    }


def lengths_from_clip(clip: schema.Clip) -> dict:
    d = schema.load(clip.clean)
    if "bone_lengths_fit2d" not in d:
        raise SystemExit(f"{clip.name}: no 2c lengths; run core.fit2d first")
    bl = np.asarray(d["bone_lengths_fit2d"], float)
    return {g: float(bl[np.nonzero(BONE_GROUP == i)[0][0]]) for i, g in enumerate(GROUPS)}


# ---------------------------------------------------------------- MJCF

def _v(a) -> str:
    return " ".join(f"{float(x):.5g}" for x in np.atleast_1d(a))


def mjcf(spec: dict) -> str:
    L = spec["lengths_m"]
    R = {**default_radii(L, spec["height_m"]), **spec["radii_m"]}   # older specs lack the newer shapes
    M, H = spec["mass_kg"], spec["height_m"]
    mf = spec["mass_fractions"]
    jt = spec["joints"]
    a = spec["ankle_height_m"]
    sp, hw, sw = L["spine"], L["hip"], L["shoulder"]
    heel_y = -np.sqrt(max(L["heel"] ** 2 - a ** 2, 1e-6))
    toe_y = np.sqrt(max(L["foot"] ** 2 - a ** 2, 1e-6))
    hip_height = a + L["shin"] + L["thigh"]
    out = []
    w = out.append

    def hinge(name, base, side=None, ind=0):
        j = jt[base]
        axis = _side_axis(base, j["axis"], side) if side else np.asarray(j["axis"], float)
        lo, hi = j["range_deg"]
        w(" " * ind + f'<joint name="{name}" type="hinge" axis="{_v(axis)}" range="{lo} {hi}"/>')

    def site(name, pos, ind):
        w(" " * ind + f'<site name="{name}" pos="{_v(pos)}" size="0.015"/>')

    def mass(part):
        return M * mf[part]

    w(f'<mujoco model="{spec["name"]}">')
    w('  <compiler angle="degree" autolimits="true"/>')
    w('  <option timestep="0.002" gravity="0 0 -9.81"/>')
    w('  <default>')
    w('    <geom contype="1" conaffinity="0" condim="3" friction="0.9 0.005 0.0001" rgba="0.80 0.78 0.74 1"/>')
    w('    <joint armature="0.01" damping="1"/>')
    w('    <site rgba="0.9 0.2 0.2 0.6" group="3"/>')
    w('  </default>')
    w('  <visual><global offwidth="1920" offheight="1080"/></visual>')
    w('  <asset>')
    w('    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.82 0.82 0.82" rgb2="0.74 0.74 0.74"/>')
    w('    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>')
    w('  </asset>')
    w('  <worldbody>')
    w('    <light name="sun" pos="0 0 4" dir="0.3 0.3 -1" directional="true"/>')
    w('    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="1"/>')
    w(f'    <body name="pelvis" pos="0 0 {hip_height:.5g}">')
    w('      <freejoint name="root"/>')
    w(f'      <geom name="pelvis" type="ellipsoid" pos="0 -0.008 {0.015 * H:.5g}" size="{_v(R["pelvis"])}" mass="{mass("pelvis"):.4g}"/>')
    site("pelvis", [0, 0, 0], 6)
    site("l_hip", [-hw, 0, 0], 6)
    site("r_hip", [hw, 0, 0], 6)
    # spine: abdomen (pelvis -> spine site), chest (spine -> neck, shoulders)
    w('      <body name="abdomen" pos="0 0 0">')
    for k in ("abdomen_flex", "abdomen_side", "abdomen_twist"):
        hinge(k, k, ind=8)
    w(f'        <geom name="abdomen" type="ellipsoid" pos="0 0 {0.55 * sp:.5g}" size="{_v(R["abdomen"])}" mass="{mass("abdomen"):.4g}"/>')
    site("spine", [0, 0, sp], 8)
    w(f'        <body name="chest" pos="0 0 {sp:.5g}">')
    for k in ("chest_flex", "chest_side", "chest_twist"):
        hinge(k, k, ind=10)
    w(f'          <geom name="chest" type="ellipsoid" pos="0 0.005 {0.42 * sp:.5g}" size="{_v(R["chest"])}" mass="{mass("chest"):.4g}"/>')
    sg = 0.9 * sw
    w(f'          <geom name="shoulder_girdle" type="capsule" fromto="{-sg:.5g} -0.008 {0.92 * sp:.5g} {sg:.5g} -0.008 {0.92 * sp:.5g}" size="{R["shoulder_girdle"]}" mass="0"/>')
    site("chest", [0, 0, 0], 10)
    site("neck", [0, 0, sp], 10)
    site("l_shoulder", [-sw, 0, sp], 10)
    site("r_shoulder", [sw, 0, sp], 10)
    w(f'          <body name="head" pos="0 0 {sp:.5g}">')
    for k in ("neck_flex", "neck_side", "neck_twist"):
        hinge(k, k, ind=12)
    hz = max(L["head"], R["head"][2] + 0.03 * H)          # the head's centre, with a neck under it
    w(f'            <geom name="neck" type="capsule" fromto="0 0 {-0.03 * H:.5g} 0 0.008 {hz - 0.7 * R["head"][2]:.5g}" size="{R["neck"]}" mass="0"/>')
    w(f'            <geom name="head" type="ellipsoid" pos="0 {0.15 * R["head"][1]:.5g} {hz:.5g}" size="{_v(R["head"])}" mass="{mass("head"):.4g}"/>')
    site("head", [0, 0, L["head"]], 12)
    w('          </body>')
    for s, x in (("l", -sw), ("r", sw)):
        ua, fa = L["upper_arm"], L["forearm"]
        w(f'          <body name="{s}_upper_arm" pos="{x:.5g} 0 {sp:.5g}">')
        for k in ("shoulder_flex", "shoulder_abd", "shoulder_rot"):
            hinge(f"{s}_{k}", k, s, 12)
        w(f'            <geom name="{s}_upper_arm" type="capsule" fromto="0 0 0 0 0 {-ua:.5g}" size="{R["upper_arm"]}" mass="{mass("upper_arm"):.4g}"/>')
        w(f'            <geom name="{s}_deltoid" type="ellipsoid" pos="{0.15 * R["deltoid"][0] * (-1 if s == "l" else 1):.5g} 0 {-0.6 * R["deltoid"][2]:.5g}" size="{_v(R["deltoid"])}" mass="0"/>')
        site(f"{s}_elbow", [0, 0, -ua], 12)
        w(f'            <body name="{s}_forearm" pos="0 0 {-ua:.5g}">')
        hinge(f"{s}_elbow", "elbow", s, 14)
        w(f'              <geom name="{s}_forearm" type="capsule" fromto="0 0 0 0 0 {-fa:.5g}" size="{R["forearm"]}" mass="{mass("forearm"):.4g}"/>')
        w(f'              <geom name="{s}_hand" type="ellipsoid" pos="0 0 {-fa - 0.9 * R["hand"][2]:.5g}" size="{_v(R["hand"])}" mass="0"/>')
        site(f"{s}_wrist", [0, 0, -fa], 14)
        w('            </body>')
        w('          </body>')
    w('        </body>')
    w('      </body>')
    for s, x in (("l", -hw), ("r", hw)):
        th, sh, fw = L["thigh"], L["shin"], R["foot_half_width"]
        w(f'      <body name="{s}_thigh" pos="{x:.5g} 0 0">')
        for k in ("hip_flex", "hip_abd", "hip_rot"):
            hinge(f"{s}_{k}", k, s, 8)
        w(f'        <geom name="{s}_thigh" type="capsule" fromto="0 0 0 0 0 {-th:.5g}" size="{R["thigh"]}" mass="{mass("thigh"):.4g}"/>')
        w(f'        <geom name="{s}_thigh_muscle" type="ellipsoid" pos="0 0.004 {-0.40 * th:.5g}" size="{_v(R["thigh_muscle"])}" mass="0"/>')
        site(f"{s}_knee", [0, 0, -th], 8)
        w(f'        <body name="{s}_shin" pos="0 0 {-th:.5g}">')
        hinge(f"{s}_knee", "knee", s, 10)
        w(f'          <geom name="{s}_shin" type="capsule" fromto="0 0 0 0 0 {-sh:.5g}" size="{R["shin"]}" mass="{mass("shin"):.4g}"/>')
        w(f'          <geom name="{s}_calf" type="ellipsoid" pos="0 {-0.25 * R["calf"][1]:.5g} {-0.33 * sh:.5g}" size="{_v(R["calf"])}" mass="0"/>')
        site(f"{s}_ankle", [0, 0, -sh], 10)
        w(f'          <body name="{s}_foot" pos="0 0 {-sh:.5g}">')
        hinge(f"{s}_ankle_flex", "ankle_flex", s, 12)
        hinge(f"{s}_ankle_roll", "ankle_roll", s, 12)
        cy, hy = 0.5 * (toe_y + heel_y), 0.5 * (toe_y - heel_y)
        w(f'            <geom name="{s}_foot" type="box" pos="0 {cy:.5g} {-a / 2:.5g}" size="{fw:.5g} {hy:.5g} {a / 2:.5g}" mass="{mass("foot"):.4g}"/>')
        site(f"{s}_heel", [0, heel_y, -a], 12)
        site(f"{s}_toe", [0, toe_y, -a], 12)
        for c, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            site(f"{s}_sole{c}", [sx * fw, cy + sy * hy, -a], 12)
        w('          </body>')
        w('        </body>')
        w('      </body>')
    w('    </body>')
    w('  </worldbody>')
    w('</mujoco>')
    return "\n".join(out) + "\n"


def build(name: str, clip: schema.Clip | None = None, force_spec: bool = False) -> Path:
    """Write bodies/<name>/spec.json (if missing) and body.xml; returns body.xml."""
    d = BODIES / name
    d.mkdir(parents=True, exist_ok=True)
    spec_path = d / "spec.json"
    if force_spec or not spec_path.exists():
        if clip is None:
            raise SystemExit(f"{spec_path} missing; pass a clip to create it")
        spec = spec_from_lengths(name, lengths_from_clip(clip), source=str(clip.clean.relative_to(schema.ROOT))
                                 if clip.clean.is_relative_to(schema.ROOT) else str(clip.clean))
        spec_path.write_text(json.dumps(spec, indent=2))
    spec = json.loads(spec_path.read_text())
    xml = d / "body.xml"
    xml.write_text(mjcf(spec))
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(xml))
    missing = [n for n in JOINT_NAMES if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) < 0]
    if missing:
        raise RuntimeError(f"body has no site for {missing}")
    total = sum(m.body_mass)
    print(f"{name}: {spec['height_m']:.2f} m, {total:.1f} kg, {m.njnt - 1} hinges, {m.nsite} sites -> {xml}")
    return xml


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", help="clip whose 2c lengths seed the spec")
    ap.add_argument("--name", help="body name (default: the clip name)")
    ap.add_argument("--force-spec", action="store_true", help="rewrite spec.json from the clip")
    a = ap.parse_args(argv)
    clip = schema.Clip(a.clip)
    build(a.name or clip.name, clip, a.force_spec)


if __name__ == "__main__":
    main()
