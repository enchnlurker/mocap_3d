"""Stage 3A2: collision shapes for what the body touches (seat, bed, head pad,
held objects), so physics (3B) has something to sit or lie on.

    python -m core.env_extract sit01 [--body sit01] [--method code|llm|auto]

Code method: the body is the ruler. Over the clip, the lowest underside of the
hips, chest, legs and head gives the support heights.
  - seat: feet on the floor plus a low, still pelvis for >= 0.5 s -> a box from
    the floor up to the hips' underside, under the hips and thighs.
  - lying (trunk tilted > 60 deg) for >= 0.5 s -> a box under the body at the
    hips / chest / legs underside, plus a head pad if the head rests higher.
    The box edges come from the depth surface: depth points at the support
    height around the body (core.depth_surface).
  Supports lower than 8 cm are the floor itself: no box.

LLM method (opt-in, --method llm; needs OPENAI_API_KEY in the environment or .env):
one call gets key frames, a depth image, a 3D point grid around the body and
the body parts (posed geoms and support heights), and returns Python that
builds the meshes. That code runs in a separate restricted process (no imports,
no file access names, CPU-time limit) and its output is checked: finite, sizes
within reason, supports under the body without cutting into it, held objects
attached to a real body part; the posed body may not cut more than 5 cm into a
support (MuJoCo contact depth). Held objects ride on that body part. Any failure
falls back to the code method. This is the one LLM step in the pipeline; it
only shapes props, never the motion.

Writes data/<clip>/props_<body>.json and data/<clip>/scene_<body>.xml (the
body plus the props as static / attached geoms).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import textwrap

import numpy as np

from core import schema
from core.bodygen import BODIES
from core.filters import runs

PARTS = {"hips": ["pelvis"], "chest": ["abdomen", "chest"],
         "legs": ["l_thigh", "r_thigh", "l_shin", "r_shin"], "head": ["head"]}
FEET = ["l_foot", "r_foot"]
MIN_SEGMENT_S = 0.5
FOOT_ON_FLOOR_M = 0.03
LOW_PELVIS_FRAC = 0.8          # pelvis below this fraction of its standing height
THIGH_ON_SEAT = 0.6            # fraction of the thigh from the hip that rests on a seat
STILL_MS = 0.15
FLOOR_SUPPORT_M = 0.08         # supports lower than this are the floor
HEAD_PAD_MIN_M = 0.03
DEPTH_BAND_M = 0.04
DEPTH_REACH_M = 1.2
MAX_BED_M = (2.2, 2.0)
LLM_MODEL = os.environ.get("MOCAP_LLM_MODEL", "gpt-6-astra")
LLM_TIMEOUT_S = 180
SANDBOX_CPU_S = 10
MAX_VERTS = 4000
MAX_PROP_M = 3.0
MAX_HELD_M = 0.6
SINK_TOL_M = 0.05


# ---------------------------------------------------------------- body geometry

def geom_extent(kind: str, size: np.ndarray, xmat: np.ndarray) -> np.ndarray:
    """[...,3] half extents of a posed geom along world X, Y, Z (support function)."""
    if kind == "capsule":
        r, h = size[0], size[1]
        return np.abs(xmat[..., :, 2]) * h + r
    if kind == "ellipsoid":
        return np.sqrt(((xmat * size) ** 2).sum(-1))
    if kind == "box":
        return (np.abs(xmat) * size).sum(-1)
    if kind == "sphere":
        return np.full(xmat.shape[:-1], size[0])
    raise ValueError(kind)


def part_bounds(shapes: dict) -> dict:
    """Per part: lo [T,3], hi [T,3] world AABB over the part's geoms."""
    out = {}
    names = list(shapes["geom_names"])
    for part, geoms in {**PARTS, "feet": FEET}.items():
        los, his = [], []
        for g in geoms:
            i = names.index(g)
            e = geom_extent(str(shapes["geom_type"][i]), shapes["geom_size"][i], shapes["geom_xmat"][:, i])
            los.append(shapes["geom_xpos"][:, i] - e)
            his.append(shapes["geom_xpos"][:, i] + e)
        out[part] = (np.min(los, 0), np.max(his, 0))
    return out


def segments(mask: np.ndarray, fps: float) -> list[tuple[int, int]]:
    n = int(round(MIN_SEGMENT_S * fps))
    return [(a, b) for a, b in runs(mask) if b - a >= n]


def _box(name, lo, hi, yaw_deg=0.0, kind="support", **extra) -> dict:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return dict(name=name, kind=kind, shape="box", center=((lo + hi) / 2).round(4).tolist(),
                half_size=((hi - lo) / 2).round(4).tolist(), yaw_deg=round(float(yaw_deg), 2), **extra)


def merge_boxes(props: list[dict], tol: float = 0.05) -> list[dict]:
    """Merge same-named-kind boxes (seat0, seat1, ...) whose tops are within tol
    and whose footprints overlap: one sofa, not one box per sitting spell."""
    out = []
    for p in props:
        lo, hi = np.subtract(p["center"], p["half_size"]), np.add(p["center"], p["half_size"])
        for q in out:
            qlo, qhi = np.subtract(q["center"], q["half_size"]), np.add(q["center"], q["half_size"])
            same = q["name"].rstrip("0123456789") == p["name"].rstrip("0123456789")
            if same and abs(qhi[2] - hi[2]) < tol and (qlo[:2] < hi[:2]).all() and (lo[:2] < qhi[:2]).all():
                mhi = np.maximum(hi, qhi)
                mhi[2] = (hi[2] + qhi[2]) / 2
                m = _box(q["name"], np.minimum(lo, qlo), mhi, source=q["source"], frames=q["frames"] + p["frames"])
                q.clear()
                q.update(m)
                break
        else:
            out.append(dict(p))
    return out


# ---------------------------------------------------------------- code method

def code_props(clip: schema.Clip, shapes: dict, clean: dict, spec: dict) -> list[dict]:
    fps = float(shapes["fps"])
    B = part_bounds(shapes)
    bodies = list(shapes["body_names"])
    pel = shapes["body_xpos"][:, bodies.index("pelvis")]
    speed = np.zeros(len(pel))
    speed[1:] = np.linalg.norm(np.diff(pel, axis=0), axis=1) * fps
    from core.filters import lowpass

    speed = lowpass(speed, fps, 2.0)
    stand_pelvis = spec["ankle_height_m"] + spec["lengths_m"]["shin"] + spec["lengths_m"]["thigh"]
    feet_down = B["feet"][0][:, 2] < FOOT_ON_FLOOR_M
    act = np.asarray(clean["activity"]).astype(str)
    lying = act == "lie"
    props = []

    seat = feet_down & (pel[:, 2] < LOW_PELVIS_FRAC * stand_pelvis) & (speed < STILL_MS) & ~lying
    for n, (a, b) in enumerate(segments(seat, fps)):
        top = float(np.median(B["hips"][0][a:b, 2]))
        if top < FLOOR_SUPPORT_M:
            continue
        tlo, thi = _upper_legs(shapes, a, b)
        lo = np.minimum(B["hips"][0][a:b], tlo).min(0)
        hi = np.maximum(B["hips"][1][a:b], thi).max(0)
        props.append(_box(f"seat{n}", [lo[0], lo[1], 0.0], [hi[0], hi[1], top],
                          frames=[int(a), int(b)], source="code"))

    for n, (a, b) in enumerate(segments(lying, fps)):
        tops = {p: float(np.median(B[p][0][a:b, 2])) for p in ("hips", "chest", "legs")}
        top = float(np.median(list(tops.values())))
        lo = np.min([B[p][0][a:b].min(0) for p in ("hips", "chest", "legs", "head")], 0)
        hi = np.max([B[p][1][a:b].max(0) for p in ("hips", "chest", "legs", "head")], 0)
        if top >= FLOOR_SUPPORT_M:
            lo, hi = _depth_edges(clip, (a + b) // 2, top, lo, hi)
            props.append(_box(f"bed{n}", [lo[0], lo[1], 0.0], [hi[0], hi[1], top],
                              frames=[int(a), int(b)], source="code"))
        else:
            top = 0.0
        head = float(np.median(B["head"][0][a:b, 2]))
        if head - top > HEAD_PAD_MIN_M:
            hl, hh = B["head"][0][a:b].min(0), B["head"][1][a:b].max(0)
            props.append(_box(f"head_pad{n}", [hl[0], hl[1], top], [hh[0], hh[1], head],
                              frames=[int(a), int(b)], source="code"))
    return merge_boxes(props)


def _upper_legs(shapes, a, b):
    """AABB [lo, hi] over frames a..b of the thighs' upper part (what rests on a
    seat; the shins hang in front of it from the knees)."""
    names, bodies = list(shapes["geom_names"]), list(shapes["body_names"])
    los, his = [], []
    for g in ("l_thigh", "r_thigh"):
        r = shapes["geom_size"][names.index(g)][0]
        hip = shapes["body_xpos"][a:b, bodies.index(g)]        # the thigh body sits at the hip
        knee = 2 * shapes["geom_xpos"][a:b, names.index(g)] - hip
        seg = np.stack([hip, hip + THIGH_ON_SEAT * (knee - hip)], 1)
        los.append(seg.min(1) - r)
        his.append(seg.max(1) + r)
    return np.min(los, 0), np.max(his, 0)


def _depth_edges(clip, frame, top, lo, hi):
    """Grow a support footprint to the depth surface at its height around the body."""
    try:
        from core.depth_surface import points_world

        pts, _, info = points_world(clip, frame, colors=False)
    except Exception as e:  # noqa: BLE001  (no depth: keep the body footprint)
        print(f"    depth edges unavailable ({e}); using the body footprint")
        return lo, hi
    if info["person"] is not None:
        pts = pts[~info["person"]]
    near = (np.abs(pts[:, 2] - top) < DEPTH_BAND_M)
    near &= (pts[:, 0] > lo[0] - DEPTH_REACH_M) & (pts[:, 0] < hi[0] + DEPTH_REACH_M)
    near &= (pts[:, 1] > lo[1] - DEPTH_REACH_M) & (pts[:, 1] < hi[1] + DEPTH_REACH_M)
    if near.sum() < 50:
        return lo, hi
    plo, phi = np.percentile(pts[near, :2], 5, axis=0), np.percentile(pts[near, :2], 95, axis=0)
    lo2, hi2 = lo.copy(), hi.copy()
    lo2[:2], hi2[:2] = np.minimum(lo[:2], plo), np.maximum(hi[:2], phi)
    c, half = (lo2[:2] + hi2[:2]) / 2, (hi2[:2] - lo2[:2]) / 2
    cap = np.sort(MAX_BED_M) / 2                      # longer cap on the longer side
    half = np.minimum(half, cap if half[0] < half[1] else cap[::-1])
    lo2[:2], hi2[:2] = c - half, c + half
    return lo2, hi2


# ---------------------------------------------------------------- LLM method

def _api_key() -> str | None:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    env = schema.ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "OPENAI_API_KEY" and v.strip():
                return v.strip().strip('"').strip("'")
    return None


def _jpeg_b64(img_rgb: np.ndarray, width: int = 768) -> str:
    import cv2

    h, w = img_rgb.shape[:2]
    img = cv2.resize(img_rgb, (width, int(h * width / w)))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def llm_inputs(clip: schema.Clip, shapes: dict, code: list[dict], n_key: int = 4) -> dict:
    """Key frames, a depth image, a 3D point grid and the body parts."""
    import cv2

    from core import video

    T = len(shapes["geom_xpos"])
    key = sorted({int((p["frames"][0] + p["frames"][1]) // 2) for p in code if "frames" in p})
    key = (key + [int(i) for i in video.evenly_spaced(T, n_key)])[:n_key]
    key = sorted(set(key))
    images = {i: _jpeg_b64(rgb) for i, rgb in video.frames(clip.video, key)}
    B = part_bounds(shapes)
    parts = {}
    for i in key:
        parts[i] = dict(
            geoms=[dict(name=str(n), type=str(t), size=np.round(s, 3).tolist(),
                        center=np.round(shapes["geom_xpos"][i, g], 3).tolist(),
                        rotation=np.round(shapes["geom_xmat"][i, g], 3).tolist())
                   for g, (n, t, s) in enumerate(zip(shapes["geom_names"], shapes["geom_type"], shapes["geom_size"]))],
            bodies={str(n): dict(pos=np.round(shapes["body_xpos"][i, b], 3).tolist(),
                                 rotation=np.round(shapes["body_xmat"][i, b], 3).tolist())
                    for b, n in enumerate(shapes["body_names"]) if n != "world"})
    lowest = {p: round(float(B[p][0][:, 2].min()), 3) for p in PARTS}
    depth_img, grid = None, None
    try:
        from core.depth_surface import points_world

        mid = key[len(key) // 2]
        pts, _, info = points_world(clip, mid, colors=False)
        c = shapes["body_xpos"][mid, list(shapes["body_names"]).index("pelvis")]
        near = np.linalg.norm(pts[:, :2] - c[:2], axis=1) < 2.5
        sel = pts[near][:: max(1, near.sum() // 600)]
        grid = dict(frame=info["frame"], points_xyz_m=np.round(sel, 2).tolist())
        d = schema.load(clip.depth)
        k = int(np.argmin(np.abs(d["frames"] - mid)))
        dm = d["depth_maps"][k].astype(np.float32)
        lo, hi = np.percentile(dm, [2, 98])
        cm = cv2.applyColorMap((255 * np.clip((dm - lo) / (hi - lo + 1e-6), 0, 1)).astype(np.uint8),
                               cv2.COLORMAP_TURBO)
        depth_img = (int(d["frames"][k]), _jpeg_b64(cv2.cvtColor(cm, cv2.COLOR_BGR2RGB), 512))
    except Exception as e:  # noqa: BLE001
        print(f"    no depth for the LLM ({e})")
    return dict(key=key, images=images, parts=parts, lowest=lowest, grid=grid, depth_img=depth_img)


PROMPT = """You build collision shapes for a physics replay of a person filmed on video.
World frame: metres, Z up, floor at z = 0. The person's body is a mannequin of posed
geoms (capsule size = [radius, half_length] along local z; ellipsoid / box size = half
extents; rotation = 3x3 world rotation, columns are local axes).

Give shapes only for what the body touches or holds: seats, beds, sofas, tables leaned
on, steps, and small held objects (cup, phone, bag). Nothing else in the room.
Supports must sit under the body parts resting on them, top at the part's underside
(lowest undersides over the clip: {lowest}), and must not cut into the body.
Code-method guesses, from body heights alone (use or improve): {code}

Answer with ONLY a Python function, no prose, no imports (`np` = numpy and `math` exist):

def build():
    return [dict(name=str, kind="support" or "held",
                 attach=None or a body name (held only, e.g. "l_forearm"),
                 frame=key frame index the held vertices are posed at (held only),
                 vertices=[[x, y, z], ...],   # world coordinates
                 faces=[[i, j, k], ...]),    # triangles, outward normals
            ...]

Return [] if the body touches nothing but the floor.

Key frames: {key}. Body parts per key frame: {parts}
3D points from the depth map near the body (world, metres): {grid}
"""


def ask_llm(inputs: dict, code: list[dict], key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=key, timeout=LLM_TIMEOUT_S)
    text = PROMPT.format(lowest=json.dumps(inputs["lowest"]), code=json.dumps(code),
                         key=inputs["key"], parts=json.dumps(inputs["parts"]),
                         grid=json.dumps(inputs["grid"]))
    content = [{"type": "input_text", "text": text}]
    for i, b64 in inputs["images"].items():
        content += [{"type": "input_text", "text": f"video frame {i}:"},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}]
    if inputs["depth_img"]:
        f, b64 = inputs["depth_img"]
        content += [{"type": "input_text", "text": f"depth map of frame {f} (blue near, red far):"},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}]
    resp = client.responses.create(model=LLM_MODEL, input=[{"role": "user", "content": content}])
    return resp.output_text


SANDBOX = textwrap.dedent("""
    import json, math, resource, sys
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
    import numpy as np
    src = sys.stdin.read()
    safe = {{k: __builtins__.__dict__[k] if hasattr(__builtins__, "__dict__") else __builtins__[k]
             for k in ("range", "len", "min", "max", "abs", "float", "int", "list", "dict", "tuple",
                       "zip", "enumerate", "round", "sum", "sorted", "map", "isinstance", "bool",
                       "str", "reversed", "any", "all", "ValueError")}}
    ns = {{"__builtins__": safe, "np": np, "math": math}}
    exec(compile(src, "<llm>", "exec"), ns)
    out = ns["build"]()
    def plain(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, (np.floating, np.integer)): return x.item()
        if isinstance(x, dict): return {{k: plain(v) for k, v in x.items()}}
        if isinstance(x, (list, tuple)): return [plain(v) for v in x]
        return x
    json.dump(plain(out), sys.stdout)
""").format(cpu=SANDBOX_CPU_S)
FORBIDDEN = ("import", "__", "open(", "exec", "eval", "compile", "globals", "locals", "getattr", "setattr",
             "np.load", "np.save", "fromfile", "tofile", "np.lib", "np.ctypeslib", "memmap", "genfromtxt",
             "loadtxt", "savetxt", "np.os", "np.sys")


def run_sandboxed(src: str) -> list:
    """Run the LLM's build() in a separate restricted Python process."""
    src = src.strip()
    if src.startswith("```"):
        src = src.split("\n", 1)[1].rsplit("```", 1)[0]
    bad = [w for w in FORBIDDEN if w in src]
    if bad:
        raise ValueError(f"code uses forbidden names {bad}")
    r = subprocess.run([sys.executable, "-I", "-c", SANDBOX], input=src, capture_output=True, text=True,
                       timeout=SANDBOX_CPU_S + 5, cwd=schema.DATA, env={"PATH": "/usr/bin:/bin"})
    if r.returncode != 0:
        raise RuntimeError(f"code failed: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}")
    return json.loads(r.stdout)


def check_props(raw: list, shapes: dict) -> list[dict]:
    """Validate the LLM's shapes; raises on anything off."""
    if not isinstance(raw, list):
        raise ValueError("build() did not return a list")
    bodies = [str(b) for b in shapes["body_names"]]
    out = []
    for n, p in enumerate(raw):
        v = np.asarray(p.get("vertices"), float)
        f = np.asarray(p.get("faces"), int)
        name = str(p.get("name", f"prop{n}"))[:40]
        if v.ndim != 2 or v.shape[1] != 3 or not 4 <= len(v) <= MAX_VERTS or not np.isfinite(v).all():
            raise ValueError(f"{name}: bad vertices")
        if f.ndim != 2 or f.shape[1] != 3 or f.min() < 0 or f.max() >= len(v):
            raise ValueError(f"{name}: bad faces")
        size = np.ptp(v, axis=0)
        kind = p.get("kind")
        if kind == "held":
            att = p.get("attach")
            if att not in bodies or att == "world":
                raise ValueError(f"{name}: attach {att!r} is not a body part")
            if size.max() > MAX_HELD_M:
                raise ValueError(f"{name}: held object {size.max():.2f} m")
            fr = int(p.get("frame", 0))
            if not 0 <= fr < len(shapes["body_xpos"]):
                raise ValueError(f"{name}: frame {fr} out of range")
            b = bodies.index(att)
            R, c = shapes["body_xmat"][fr, b], shapes["body_xpos"][fr, b]
            out.append(dict(name=name, kind="held", shape="mesh", attach=att,
                            vertices=np.round((v - c) @ R, 4).tolist(), faces=f.tolist(), source="llm"))
        elif kind == "support":
            if size.max() > MAX_PROP_M or v[:, 2].min() < -0.05:
                raise ValueError(f"{name}: support {size.round(2).tolist()} m, bottom {v[:, 2].min():.2f}")
            out.append(dict(name=name, kind="support", shape="mesh", vertices=np.round(v, 4).tolist(),
                            faces=f.tolist(), source="llm"))
        else:
            raise ValueError(f"{name}: kind {kind!r}")
    return out


# body geom -> the joint at its far end; a geom whose joint is inferred is a rule pose, not capture
GEOM_JOINT = {"pelvis": "pelvis", "abdomen": "spine", "chest": "neck", "head": "head",
              **{f"{s}_{g}": f"{s}_{j}" for s in "lr" for g, j in (("upper_arm", "elbow"), ("forearm", "wrist"),
                                                                      ("thigh", "knee"), ("shin", "ankle"),
                                                                      ("foot", "toe"))}}


def penetration(scene: str, qpos: np.ndarray, inferred: np.ndarray | None = None, every: int = 3) -> dict:
    """95th-percentile depth (m) by which the captured body cuts into each support prop."""
    import mujoco

    from core.skeleton import J

    m = mujoco.MjModel.from_xml_string(scene)
    d = mujoco.MjData(m)
    props = {g: m.geom(g).name for g in range(m.ngeom) if m.geom(g).name.startswith("prop_")
             and m.geom_bodyid[g] == 0}
    depth = {n: [] for n in props.values()}
    for t in range(0, len(qpos), every):
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(m, d)
        frame = {n: 0.0 for n in props.values()}
        for c in d.contact[: d.ncon]:
            for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                j = GEOM_JOINT.get(m.geom(other).name)
                if g in props and j and (inferred is None or not inferred[t, J[j]]):
                    frame[props[g]] = max(frame[props[g]], -float(c.dist))
        for n, v in frame.items():
            depth[n].append(v)
    return {n: float(np.percentile(v, 95)) if v else 0.0 for n, v in depth.items()}


def check_penetration(scene: str, qpos: np.ndarray, inferred: np.ndarray | None = None) -> None:
    for name, pen in penetration(scene, qpos, inferred).items():
        if pen > SINK_TOL_M:
            raise ValueError(f"{name}: the body cuts {pen * 100:.0f} cm into it")


# ---------------------------------------------------------------- scene

def scene_xml(body_xml, props: list[dict]) -> str:
    import mujoco

    spec = mujoco.MjSpec.from_file(str(body_xml))
    for n, p in enumerate(props):
        nm = f"prop_{n}_{p['name']}"
        if p["shape"] == "box":
            yaw = np.radians(p["yaw_deg"])
            spec.worldbody.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_BOX, size=p["half_size"],
                                    pos=p["center"], quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
                                    rgba=[0.55, 0.42, 0.3, 1], contype=0, conaffinity=1)
            continue
        spec.add_mesh(name=nm, uservert=np.asarray(p["vertices"], float).ravel().tolist(),
                      userface=np.asarray(p["faces"], int).ravel().tolist())
        parent = spec.worldbody if p["kind"] == "support" else spec.body(p["attach"])
        held = p["kind"] == "held"
        parent.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=nm,
                        rgba=[0.3, 0.5, 0.8, 1] if held else [0.55, 0.42, 0.3, 1],
                        contype=0 if not held else 1, conaffinity=1 if not held else 0,
                        mass=0.2 if held else 0)
    spec.compile()
    return spec.to_xml()


# ---------------------------------------------------------------- entry

def run(clip: schema.Clip, body: str | None = None, method: str = "code") -> dict:
    body = body or clip.name
    shapes = schema.load(clip / f"bodyshapes_{body}.npz")
    clean = schema.load(clip.clean)
    spec = json.loads((BODIES / body / "spec.json").read_text())
    code = code_props(clip, shapes, clean, spec)
    key = _api_key()
    if method == "auto":
        method = "llm" if key else "code"
    used, reason, props = "code", "", code
    if method == "llm":
        if not key:
            reason = "no OPENAI_API_KEY"
        else:
            try:
                inputs = llm_inputs(clip, shapes, code)
                src = ask_llm(inputs, code, key)
                (clip / f"props_{body}_llm.py").write_text(src)
                props = check_props(run_sandboxed(src), shapes)
                check_penetration(scene_xml(BODIES / body / "body.xml", props),
                                  schema.load(clip / f"retarget_{body}.npz")["qpos"], clean["inferred"].astype(bool))
                used = "llm"
            except Exception as e:  # noqa: BLE001  (any failure falls back to the code method)
                reason = f"{type(e).__name__}: {e}"
                props = code
        if used != "llm":
            print(f"  LLM method failed ({reason}); using the code method")
    out = dict(clip=clip.name, body=body, method=used, fallback_reason=reason, props=props)
    (clip / f"props_{body}.json").write_text(json.dumps(out, indent=1))
    xml = scene_xml(BODIES / body / "body.xml", props)
    (clip / f"scene_{body}.xml").write_text(xml)
    print(f"{clip.name}: {len(props)} props ({used}): "
          + ", ".join(f"{p['name']} ({p['kind']})" for p in props) + f" -> props_{body}.json, scene_{body}.xml")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--body", help="default: the clip name")
    ap.add_argument("--method", default="code", choices=["code", "llm", "auto"],
                    help="code (default); llm; auto: llm when OPENAI_API_KEY is set, else code")
    a = ap.parse_args(argv)
    run(schema.Clip(a.clip), a.body, a.method)


if __name__ == "__main__":
    main()
