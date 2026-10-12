"""Open a clip's 3D joints in Blender: animated skeletons, floor, and the solved
camera with the source video as its background (look through the camera,
Numpad 0, to see the skeleton overlaid on the footage).

    blender --python core/blender_view.py -- walk01 [--layers fit2d 2b clean] [--save out.blend]

or from the repo venv:  python -m core.blender_view walk01   (finds Blender and launches it)

Layers (World frame, Z up, metres):
    fit2d  positions_fit2d          green
    2b     positions_proportional   orange
    clean  positions                blue   (hidden by default)
    raw    positions_raw            grey   (hidden by default)
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


def _args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--layers", nargs="+", default=list(LAYERS), choices=list(LAYERS))
    ap.add_argument("--save", help="also save a .blend file")
    ap.add_argument("--blender", help="path to the Blender executable (launcher mode)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- inside Blender

def build(a):
    import bpy
    import numpy as np
    from mathutils import Matrix

    sys.path.insert(0, str(ROOT))
    from core import schema
    from core.skeleton import BONE_PAIRS, JOINT_NAMES

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

    def material(name, rgba):
        m = bpy.data.materials.new(name)
        m.diffuse_color = rgba
        bsdf = m.node_tree.nodes.get("Principled BSDF") if m.node_tree else None
        if bsdf:
            bsdf.inputs["Base Color"].default_value = rgba
        return m

    for layer in a.layers:
        key, rgba = LAYERS[layer]
        if key not in d:
            print(f"[blender_view] {clip.name}: no {key}, skipping layer {layer}")
            continue
        P = np.asarray(d[key], float)
        coll = bpy.data.collections.new(layer)
        scene.collection.children.link(coll)
        mat = material(layer, rgba)

        # one empty per joint, keyframed every frame
        empties = []
        for j, name in enumerate(JOINT_NAMES):
            e = bpy.data.objects.new(f"{layer}.{name}", None)
            e.empty_display_type = "SPHERE"
            e.empty_display_size = JOINT_SIZE
            coll.objects.link(e)
            for t in range(T):
                if np.isfinite(P[t, j]).all():
                    e.location = P[t, j]
                    e.keyframe_insert("location", frame=t + 1)
            empties.append(e)

        # bones: a mesh whose vertices are hooked to the empties, thickened by a Skin modifier
        me = bpy.data.meshes.new(f"{layer}.bones")
        me.from_pydata([(0.0, 0.0, 0.0)] * len(JOINT_NAMES), [tuple(map(int, p)) for p in BONE_PAIRS], [])
        ob = bpy.data.objects.new(f"{layer}.skeleton", me)
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

    # floor
    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "floor"
    floor.data.materials.append(material("floor", (0.8, 0.8, 0.8, 1.0)))

    # camera: fit2d camera (OpenCV axes x right, y down, z forward) -> Blender camera (-z forward, +y up)
    if "fit2d_camera_pos" in d:
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
