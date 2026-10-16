"""Open a clip's 3D joints in Blender: animated skeletons and the floor, plus on
request the solved camera over the footage, the retargeted body, its props, a
second clean file side by side, or a depth point cloud.

    blender --python core/blender_view.py -- walk01 [--layers fit2d 2b clean] [--save out.blend]

or from the repo venv:  python -m core.blender_view walk01   (finds Blender and launches it)

Layers (World frame, Z up, metres):
    fit2d  positions_fit2d          green
    2b     positions_proportional   orange
    clean  positions                blue   (hidden by default)
    raw    positions_raw            grey   (hidden by default)

Options:
    --camera            the solved camera with the source video as its background
                        (look through it, Numpad 0, to see the skeleton on the footage)
    --retarget [BODY]   the MuJoCo body posed by 3A (bodyshapes_<body>.npz, from core.playback)
    --props [BODY]      3A2 props (props_<body>.json, from core.env_extract); held objects ride their body part
    --compare FILE      a second clean file (path, or a clip name) next to the first
    --ply FILE          a point cloud from core.depth_surface
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAYERS = {
    "fit2d": ("positions_fit2d", (0.10, 0.75, 0.25, 1.0)),
    "2b": ("positions_proportional", (0.95, 0.50, 0.05, 1.0)),
    "clean": ("positions", (0.15, 0.45, 0.95, 1.0)),
    "raw": ("positions_raw", (0.55, 0.55, 0.55, 1.0)),
}
DEFAULT_VISIBLE = ("fit2d", "2b")
BONE_RADIUS = 0.018
JOINT_SIZE = 0.03
BODY_RGBA = (0.72, 0.74, 0.78, 1.0)
PROP_RGBA = {"support": (0.55, 0.42, 0.30, 1.0), "held": (0.30, 0.50, 0.80, 1.0)}


def _args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--layers", nargs="+", default=list(LAYERS), choices=list(LAYERS))
    ap.add_argument("--camera", action="store_true", help="solved camera with the footage as background")
    ap.add_argument("--retarget", nargs="?", const="", metavar="BODY", help="posed MuJoCo body (default: clip name)")
    ap.add_argument("--props", nargs="?", const="", metavar="BODY", help="3A2 props (default: clip name)")
    ap.add_argument("--compare", metavar="FILE", help="second clean file or clip name, shown alongside")
    ap.add_argument("--compare-offset", type=float, default=1.5, help="X offset of --compare (m)")
    ap.add_argument("--ply", help="point cloud to import (core.depth_surface)")
    ap.add_argument("--save", help="also save a .blend file")
    ap.add_argument("--blender", help="path to the Blender executable (launcher mode)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- inside Blender

def _material(name, rgba):
    import bpy

    m = bpy.data.materials.new(name)
    m.diffuse_color = rgba
    bsdf = m.node_tree.nodes.get("Principled BSDF") if m.node_tree else None
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
    return m


def _collection(name):
    import bpy

    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)
    return coll


def _skeletons(d, layers, prefix="", offset=(0.0, 0.0, 0.0), clip_name=""):
    import bpy
    import numpy as np

    from core.skeleton import BONE_PAIRS, JOINT_NAMES

    T = len(d["positions"])
    for layer in layers:
        key, rgba = LAYERS[layer]
        if key not in d:
            print(f"[blender_view] {clip_name}: no {key}, skipping layer {prefix}{layer}")
            continue
        P = np.asarray(d[key], float) + np.asarray(offset)
        name = f"{prefix}{layer}"
        coll = _collection(name)
        if prefix:
            rgba = tuple(0.5 + 0.5 * c for c in rgba[:3]) + (1.0,)
        mat = _material(name, rgba)

        # one empty per joint, keyframed every frame
        empties = []
        for j, jn in enumerate(JOINT_NAMES):
            e = bpy.data.objects.new(f"{name}.{jn}", None)
            e.empty_display_type = "SPHERE"
            e.empty_display_size = JOINT_SIZE
            coll.objects.link(e)
            for t in range(T):
                if np.isfinite(P[t, j]).all():
                    e.location = P[t, j]
                    e.keyframe_insert("location", frame=t + 1)
            empties.append(e)

        # bones: a mesh whose vertices are hooked to the empties, thickened by a Skin modifier
        me = bpy.data.meshes.new(f"{name}.bones")
        me.from_pydata([(0.0, 0.0, 0.0)] * len(JOINT_NAMES), [tuple(map(int, p)) for p in BONE_PAIRS], [])
        ob = bpy.data.objects.new(f"{name}.skeleton", me)
        coll.objects.link(ob)
        ob.data.materials.append(mat)
        ob.color = rgba
        for j, e in enumerate(empties):
            h = ob.modifiers.new(f"hook.{JOINT_NAMES[j]}", "HOOK")
            h.object = e
            h.vertex_indices_set([j])
        sk = ob.modifiers.new("skin", "SKIN")
        sk.use_smooth_shade = True
        for v in me.skin_vertices[0].data:
            v.radius = (BONE_RADIUS, BONE_RADIUS)
        me.skin_vertices[0].data[0].use_root = True

        hidden = layer not in DEFAULT_VISIBLE
        for o in coll.objects:
            o.hide_set(hidden)
            o.hide_render = hidden


def _camera(d, clip):
    """fit2d camera (OpenCV axes x right, y down, z forward) -> Blender camera (-z forward, +y up)."""
    import bpy
    import numpy as np
    from mathutils import Matrix

    if "fit2d_camera_pos" not in d:
        print(f"[blender_view] {clip.name}: no fit2d camera; run core.fit2d")
        return
    scene = bpy.context.scene
    T = len(d["positions"])
    R = np.asarray(d["floor_R"], float) @ np.diag([1.0, -1.0, -1.0])
    cam_data = bpy.data.cameras.new("fit2d_camera")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.lens = float(d["fit2d_focal_px"]) * 36.0 / float(d["width"])
    cam_data.clip_start = 0.05
    cam = bpy.data.objects.new("fit2d_camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    C = np.asarray(d["fit2d_camera_pos"], float)
    static = np.allclose(C, C[0])
    for t in range(1 if static else T):
        M = Matrix([list(R[0]) + [C[t, 0]], list(R[1]) + [C[t, 1]], list(R[2]) + [C[t, 2]], [0, 0, 0, 1]])
        cam.matrix_world = M
        if not static:
            cam.keyframe_insert("location", frame=t + 1)
            cam.keyframe_insert("rotation_euler", frame=t + 1)
    if clip.video.exists():
        mc = bpy.data.movieclips.load(str(clip.video))
        cam_data.show_background_images = True
        bg = cam_data.background_images.new()
        bg.source = "MOVIE_CLIP"
        bg.clip = mc
        bg.alpha = 1.0
        bg.frame_method = "FIT"
        bg.display_depth = "BACK"
        bg.clip_user.proxy_render_size = "FULL"


def _geom_mesh(name, kind, size):
    """Mesh for a MuJoCo geom in its own frame (sizes baked into the vertices)."""
    import bmesh
    import bpy

    bm = bmesh.new()
    if kind == "box":
        bmesh.ops.create_cube(bm, size=2.0)
        for v in bm.verts:
            v.co.x, v.co.y, v.co.z = v.co.x * size[0], v.co.y * size[1], v.co.z * size[2]
    else:
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=1.0)
        for v in bm.verts:
            if kind == "ellipsoid":
                v.co.x, v.co.y, v.co.z = v.co.x * size[0], v.co.y * size[1], v.co.z * size[2]
            elif kind == "capsule":  # a sphere split at the equator and stretched
                r, h = size[0], size[1]
                v.co.x, v.co.y = v.co.x * r, v.co.y * r
                v.co.z = v.co.z * r + (h if v.co.z > 1e-6 else -h if v.co.z < -1e-6 else 0.0)
            else:
                v.co *= size[0]
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for poly in me.polygons:
        poly.use_smooth = True
    return me


def _keyframe_pose(ob, xpos, xmat):
    from mathutils import Matrix

    ob.rotation_mode = "QUATERNION"
    for t in range(len(xpos)):
        ob.location = tuple(xpos[t])
        ob.rotation_quaternion = Matrix([list(r) for r in xmat[t]]).to_quaternion()
        ob.keyframe_insert("location", frame=t + 1)
        ob.keyframe_insert("rotation_quaternion", frame=t + 1)


def _retarget(clip, body):
    import bpy

    from core import schema

    path = clip / f"bodyshapes_{body}.npz"
    if not path.exists():
        print(f"[blender_view] {path} missing; run core.retarget and core.playback")
        return
    s = schema.load(path)
    coll = _collection(f"body.{body}")
    mat = _material(f"body.{body}", BODY_RGBA)
    for g, (name, kind) in enumerate(zip(s["geom_names"], s["geom_type"])):
        ob = bpy.data.objects.new(f"{body}.{name}", _geom_mesh(f"{body}.{name}", str(kind), s["geom_size"][g]))
        ob.data.materials.append(mat)
        ob.color = BODY_RGBA
        coll.objects.link(ob)
        _keyframe_pose(ob, s["geom_xpos"][:, g], s["geom_xmat"][:, g])


def _props(clip, body):
    import json

    import bpy
    import numpy as np

    from core import schema

    path = clip / f"props_{body}.json"
    if not path.exists():
        print(f"[blender_view] {path} missing; run core.env_extract")
        return
    props = json.loads(path.read_text())["props"]
    shapes_path = clip / f"bodyshapes_{body}.npz"
    shapes = schema.load(shapes_path) if shapes_path.exists() else None
    coll = _collection(f"props.{body}")
    for p in props:
        rgba = PROP_RGBA[p["kind"]]
        if p["shape"] == "box":
            me = _geom_mesh(p["name"], "box", p["half_size"])
        else:
            me = bpy.data.meshes.new(p["name"])
            me.from_pydata([tuple(v) for v in p["vertices"]], [], [tuple(f) for f in p["faces"]])
        ob = bpy.data.objects.new(f"prop.{p['name']}", me)
        ob.data.materials.append(_material(f"prop.{p['name']}", rgba))
        ob.color = rgba
        coll.objects.link(ob)
        if p["shape"] == "box":
            ob.location = p["center"]
            ob.rotation_euler = (0.0, 0.0, np.radians(p.get("yaw_deg", 0.0)))
        if p.get("attach") and shapes is not None:
            b = list(shapes["body_names"]).index(p["attach"])
            _keyframe_pose(ob, shapes["body_xpos"][:, b], shapes["body_xmat"][:, b])


def build(a):
    import bpy

    sys.path.insert(0, str(ROOT))
    from core import schema

    clip = schema.Clip(a.clip)
    d = schema.load(clip.clean)
    fps = d["fps"]
    T = len(d["positions"])

    # fresh scene
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    scene = bpy.context.scene
    scene.name = clip.name
    scene.render.fps = int(round(fps))
    scene.render.fps_base = scene.render.fps / fps
    scene.frame_start, scene.frame_end = 1, T
    scene.render.resolution_x, scene.render.resolution_y = int(d["width"]), int(d["height"])

    _skeletons(d, a.layers, clip_name=clip.name)
    if a.compare:
        cpath = Path(a.compare)
        other = schema.load(cpath if cpath.suffix == ".npz" else schema.Clip(a.compare).clean)
        _skeletons(other, a.layers, prefix="compare.", offset=(a.compare_offset, 0.0, 0.0), clip_name=a.compare)

    # floor
    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "floor"
    floor.data.materials.append(_material("floor", (0.8, 0.8, 0.8, 1.0)))

    if a.camera:
        _camera(d, clip)
    if a.retarget is not None:
        _retarget(clip, a.retarget or clip.name)
    if a.props is not None:
        _props(clip, a.props or clip.name)
    if a.ply:
        bpy.ops.wm.ply_import(filepath=str(Path(a.ply).resolve()))

    # light so solid/material views read well
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.rotation_euler = (0.6, 0.2, 0.8)
    scene.collection.objects.link(sun)

    scene.frame_set(1)
    _frame_view()
    if a.save:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(a.save).resolve()))
    print(f"[blender_view] {clip.name}: {T} frames at {fps:.2f} fps, layers {a.layers}")


def _frame_view():
    """Point the 3D viewports at the skeleton with solid shading and object colours."""
    import bpy

    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.type = "SOLID"
                    space.shading.color_type = "OBJECT"
                    space.overlay.show_relationship_lines = False  # hook lines to every joint
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region:
                with bpy.context.temp_override(window=window, area=area, region=region):
                    bpy.ops.view3d.view_all()


# ---------------------------------------------------------------- launcher

def find_blender(explicit: str | None = None) -> str:
    for c in (explicit, shutil.which("blender"), "/Applications/Blender.app/Contents/MacOS/Blender"):
        if c and Path(c).exists():
            return c
    raise SystemExit("Blender not found: install it (brew install --cask blender) or pass --blender")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    a = _args(argv)
    cmd = [find_blender(a.blender), "--python", str(Path(__file__).resolve()), "--", *[x for x in argv]]
    subprocess.Popen(cmd)


if __name__ == "__main__":
    try:
        import bpy  # noqa: F401
    except ImportError:
        main()
    else:
        args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
        build(_args(args))
