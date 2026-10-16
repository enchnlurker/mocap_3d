"""Stage 3A playback: set the retargeted joint angles each frame (no physics).

    python -m core.playback walk01 [--body walk01] [--viewer] [--no-video] [--scene]

Renders data/<clip>/playback_<body>.mp4 with the 2c targets drawn as spheres
(green captured, orange inferred), or opens the MuJoCo viewer (--viewer; on
macOS run it under `mjpython`). --scene shows the 3A2 props too (scene_<body>.xml,
written by core.env_extract). Always exports the per-frame body shapes for
Blender (core.blender_view --retarget) and 3A2 (core.env_extract):

data/<clip>/bodyshapes_<body>.npz
    geom_names [G], geom_body [G], geom_type [G] (sphere/capsule/ellipsoid/box/...),
    geom_size [G,3] (MuJoCo sizes), geom_xpos [T,G,3], geom_xmat [T,G,3,3],
    body_names [B], body_xpos [T,B,3], body_xmat [T,B,3,3]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from core import schema
from core.skeleton import JOINT_NAMES

GREEN = (0.15, 0.8, 0.3, 0.9)
ORANGE = (1.0, 0.55, 0.1, 0.9)
TARGET_RADIUS = 0.025
GEOM_TYPES = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}


def load(clip: schema.Clip, body: str):
    import mujoco

    r = schema.load(clip / f"retarget_{body}.npz")
    model = mujoco.MjModel.from_xml_path(str(r["xml"]))
    return model, r


def body_shapes(model, qpos: np.ndarray) -> dict:
    """Per-frame world pose of every body geom (the floor excluded) and every body."""
    import mujoco

    data = mujoco.MjData(model)
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0]
    T = len(qpos)
    gx, gm = np.zeros((T, len(geoms), 3)), np.zeros((T, len(geoms), 3, 3))
    bx, bm = np.zeros((T, model.nbody, 3)), np.zeros((T, model.nbody, 3, 3))
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        gx[t], gm[t] = data.geom_xpos[geoms], data.geom_xmat[geoms].reshape(-1, 3, 3)
        bx[t], bm[t] = data.xpos, data.xmat.reshape(-1, 3, 3)
    return dict(geom_names=np.array([model.geom(g).name for g in geoms]),
                geom_body=np.array([model.body(model.geom_bodyid[g]).name for g in geoms]),
                geom_type=np.array([GEOM_TYPES[int(model.geom_type[g])] for g in geoms]),
                geom_size=model.geom_size[geoms].copy(), geom_xpos=gx, geom_xmat=gm,
                body_names=np.array([model.body(b).name for b in range(model.nbody)]),
                body_xpos=bx, body_xmat=bm)


def _add_sphere(scene, pos, rgba):
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.full(3, TARGET_RADIUS), np.asarray(pos, float), np.eye(3).ravel(),
                        np.asarray(rgba, np.float32))
    scene.ngeom += 1


def render(model, qpos, targets, inferred, fps, path, width=1280, height=720):
    import cv2
    import mujoco

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 3.5, -12.0, 120.0
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    pel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    look = None
    for t in range(len(qpos)):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        p = data.xpos[pel].copy()
        look = p if look is None else 0.9 * look + 0.1 * p   # follow the body without shaking
        cam.lookat[:] = (look[0], look[1], 0.9)
        renderer.update_scene(data, cam)
        for j in range(len(JOINT_NAMES)):
            if np.isfinite(targets[t, j]).all():
                _add_sphere(renderer.scene, targets[t, j], ORANGE if inferred[t, j] else GREEN)
        img = renderer.render()
        cv2.putText(img, str(t), (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
        wr.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    wr.release()
    renderer.close()


def view(model, qpos, targets, inferred, fps):
    """Loop the clip in the MuJoCo viewer (macOS: run under mjpython)."""
    import mujoco
    import mujoco.viewer

    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as v:
        t = 0
        while v.is_running():
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            with v.lock():
                v.user_scn.ngeom = 0
                for j in range(len(JOINT_NAMES)):
                    if np.isfinite(targets[t, j]).all():
                        _add_sphere(v.user_scn, targets[t, j], ORANGE if inferred[t, j] else GREEN)
            v.sync()
            time.sleep(1.0 / fps)
            t = (t + 1) % len(qpos)


def run(clip: schema.Clip, body: str | None = None, video: bool = True, viewer: bool = False,
        scene: bool = False):
    import mujoco

    body = body or clip.name
    model, r = load(clip, body)
    d = schema.load(clip.clean)
    targets, inferred, fps = np.asarray(d["positions_fit2d"], float), d["inferred"].astype(bool), d["fps"]
    shapes = body_shapes(model, r["qpos"])
    out = clip / f"bodyshapes_{body}.npz"
    schema.save(out, fps, JOINT_NAMES, body=body, **shapes)
    print(f"{clip.name}: body shapes -> {out}")
    if scene:
        xml = clip / f"scene_{body}.xml"
        if not xml.exists():
            raise SystemExit(f"{xml} missing; run core.env_extract first")
        model = mujoco.MjModel.from_xml_path(str(xml))
    if viewer:
        view(model, r["qpos"], targets, inferred, fps)
    elif video:
        path = clip / f"playback_{body}{'_scene' if scene else ''}.mp4"
        render(model, r["qpos"], targets, inferred, fps, path)
        print(f"  -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--body", help="default: the clip name")
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer instead of writing an mp4")
    ap.add_argument("--no-video", action="store_true", help="only export body shapes")
    ap.add_argument("--scene", action="store_true", help="show the 3A2 props (scene_<body>.xml)")
    a = ap.parse_args(argv)
    run(schema.Clip(a.clip), a.body, not a.no_video, a.viewer, a.scene)


if __name__ == "__main__":
    main()
