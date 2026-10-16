"""Stage 3A retarget: 2c joint targets -> joint angles of a MuJoCo body (mink IK).

    python -m core.retarget walk01 [--body walk01]

Each frame, mink pulls each body site toward its 2c joint (positions_fit2d):
  weights   pelvis strongest (50): the legs reach and the body doesn't move.
            Ends (head, wrists, ankles, toes) medium. Knees and elbows low: they
            only hint at the bend direction. Inferred joints x0.1.
  hard      joint ranges; the four sole corners of each foot box stay at or
            above z = 0. A small pull toward the previous frame keeps motion
            continuous.
  fixes     chest orientation target: stops the two 3-hinge spine joints from
            winding against each other; only while within 60 deg of the pelvis.
            Weak neutral pull on ankle roll, hip twist and spine side-bend /
            twist, which the targets barely see.
            Speed limit (360 deg/s) on the twist and ankle hinges, which can
            otherwise flip from limit to limit while a limb is straight.
            Unstick: an elbow or knee stuck straight while its target bends
            starts the frame at the target's bend, held there for the first
            half of the frame's iterations so the shoulder / hip can turn the
            bend into the right plane (a straight limb is a stable dead point).

Writes data/<clip>/retarget_<body>.npz:
    qpos [T,nq], qpos_names [nq], site_names [20], ik_residual [T,20] (m),
    joint_pops [T] (frames where a hinge jumps more than POP_DEG)
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from core import schema
from core.bodygen import BODIES, TWIST_JOINTS
from core.skeleton import J, JOINT_NAMES

W_PELVIS = 50.0
W_ENDS = 10.0       # head, wrists, ankles, toes
W_MID = 4.0         # hips, shoulders, neck, spine
W_HINT = 1.0        # knees, elbows: bend direction only; heels (the foot box is rigid, the toe sets it)
INFERRED_FACTOR = 0.1
W_PELVIS_ORIENT = 5.0
W_CHEST_ORIENT = 2.0
CHEST_MAX_DEG = 60.0          # chest target only while within this of the pelvis
W_NEUTRAL = 0.05              # on twist / roll / side-bend hinges
W_PREVIOUS = 0.02             # pull toward the previous frame, all hinges
W_HOLD = 10.0                 # unstick: hold the bend while the rest of the limb turns
SPEED_LIMIT_DEG_S = 360.0     # twist and ankle hinges
STRAIGHT_DEG = 5.0            # unstick: hinge this straight ...
TARGET_BEND_DEG = 20.0        # ... while the target bends at least this much
POP_DEG = 30.0                # per-frame hinge jump reported as a pop (at 30 fps)
ITERS_FIRST, ITERS = 150, 12
SOLVER = "daqp"

ENDS = ["head", "l_wrist", "r_wrist", "l_ankle", "r_ankle", "l_toe", "r_toe"]
HINTS = ["l_knee", "r_knee", "l_elbow", "r_elbow", "l_heel", "r_heel"]
HINGE_PAIRS = {"l_knee": ("l_hip", "l_knee", "l_ankle"), "r_knee": ("r_hip", "r_knee", "r_ankle"),
               "l_elbow": ("l_shoulder", "l_elbow", "l_wrist"), "r_elbow": ("r_shoulder", "r_elbow", "r_wrist")}
NEUTRAL = ["abdomen_side", "abdomen_twist", "chest_side", "chest_twist", "l_hip_rot", "r_hip_rot",
           "l_ankle_roll", "r_ankle_roll"]


def site_weights(inferred: np.ndarray) -> np.ndarray:
    """[T,20] position cost per site and frame."""
    w = np.full(len(JOINT_NAMES), W_MID)
    w[J["pelvis"]] = W_PELVIS
    w[[J[n] for n in ENDS]] = W_ENDS
    w[[J[n] for n in HINTS]] = W_HINT
    return w[None] * np.where(inferred, INFERRED_FACTOR, 1.0)


def frame_rotation(right: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Rotation whose X is the body's right, Z up (orthogonalised), Y forward."""
    x = right / np.linalg.norm(right)
    z = up - (up @ x) * x
    z /= np.linalg.norm(z)
    return np.stack([x, np.cross(z, x), z], 1)


def body_frames(P: np.ndarray):
    """Pelvis and chest rotations [T,3,3] from 2c joints."""
    pel = [frame_rotation(p[J["r_hip"]] - p[J["l_hip"]], p[J["spine"]] - p[J["pelvis"]]) for p in P]
    che = [frame_rotation(p[J["r_shoulder"]] - p[J["l_shoulder"]], p[J["neck"]] - p[J["spine"]]) for p in P]
    return np.asarray(pel), np.asarray(che)


def _angle(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def target_bend(P: np.ndarray, a: str, b: str, c: str) -> float:
    u, v = P[J[b]] - P[J[a]], P[J[c]] - P[J[b]]
    return float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))


class FloorLimit:
    """Sole corners stay at or above z = 0: z + J_z dq >= 0."""

    def __init__(self, model, site_names):
        import mujoco

        self.ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in site_names]

    def compute_qp_inequalities(self, configuration, dt):
        import mujoco
        from mink.limits.limit import Constraint

        m, d = configuration.model, configuration.data
        G, h = [], []
        jac = np.zeros((3, m.nv))
        for i in self.ids:
            mujoco.mj_jacSite(m, d, jac, None, i)
            G.append(-jac[2].copy())
            h.append(d.site_xpos[i, 2])
        return Constraint(G=np.asarray(G), h=np.asarray(h))


def retarget(xml_path, P: np.ndarray, inferred: np.ndarray, fps: float, verbose: bool = True) -> dict:
    import mink
    import mujoco
    from mink.lie import SE3, SO3

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    cfg = mink.Configuration(model)
    T = len(P)
    dt = 1.0 / fps
    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    dof = lambda n: model.jnt_dofadr[jid(n)]
    qadr = lambda n: model.jnt_qposadr[jid(n)]
    hinges = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]

    sites = [mink.FrameTask(n, "site", position_cost=1.0, orientation_cost=0.0, lm_damping=1e-3)
             for n in JOINT_NAMES]
    pelvis_o = mink.FrameTask("pelvis", "body", position_cost=0.0, orientation_cost=W_PELVIS_ORIENT)
    chest_o = mink.FrameTask("chest", "site", position_cost=0.0, orientation_cost=W_CHEST_ORIENT)
    neutral_cost = np.zeros(model.nv)
    for n in NEUTRAL:
        neutral_cost[dof(n)] = W_NEUTRAL
    neutral = mink.PostureTask(model, cost=neutral_cost)
    neutral.set_target(model.qpos0)
    previous = mink.PostureTask(model, cost=W_PREVIOUS)
    hold = mink.PostureTask(model, cost=np.zeros(model.nv))
    fast = [n for n in hinges if any(n.endswith(t) for t in TWIST_JOINTS) or "ankle" in n]
    sole = [model.site(i).name for i in range(model.nsite) if "_sole" in model.site(i).name]
    floor = FloorLimit(model, sole)

    weights = site_weights(inferred)
    R_pel, R_che = body_frames(P)
    q = model.qpos0.copy()
    q[0:3] = P[0, J["pelvis"]]
    q[3:7] = SO3.from_matrix(R_pel[0]).wxyz
    cfg.update(q)

    out_q = np.zeros((T, model.nq))
    resid = np.zeros((T, len(JOINT_NAMES)))
    sid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in JOINT_NAMES]
    chest_on = np.zeros(T, bool)
    unstuck = np.zeros(T, bool)
    t0 = time.time()
    for t in range(T):
        for k, task in enumerate(sites):
            task.set_position_cost(weights[t, k])
            task.set_target(SE3.from_translation(P[t, k]))
        pelvis_o.set_target(SE3.from_rotation(SO3.from_matrix(R_pel[t])))
        chest_on[t] = _angle(R_pel[t].T @ R_che[t]) < CHEST_MAX_DEG
        chest_o.set_orientation_cost(W_CHEST_ORIENT if chest_on[t] else 0.0)
        chest_o.set_target(SE3.from_rotation(SO3.from_matrix(R_che[t])))
        # unstick: a straight elbow / knee whose target bends starts at the target's bend
        q = cfg.q.copy()
        hold_cost = np.zeros(model.nv)
        for n, (a, b, c) in HINGE_PAIRS.items():
            bend = target_bend(P[t], a, b, c)
            if abs(np.degrees(q[qadr(n)])) < STRAIGHT_DEG and bend > TARGET_BEND_DEG:
                lo, hi = model.jnt_range[jid(n)]
                q[qadr(n)] = np.clip(np.radians(bend), lo, hi)
                hold_cost[dof(n)] = W_HOLD
                unstuck[t] = True
        cfg.update(q)
        previous.set_target(q)
        hold.set_cost(hold_cost)
        hold.set_target(q)
        first = t == 0
        n_it = ITERS_FIRST if first else ITERS
        step = dt / n_it
        limits = [mink.ConfigurationLimit(model), floor]
        if not first:
            limits.append(mink.VelocityLimit(model, {n: np.radians(SPEED_LIMIT_DEG_S) for n in fast}))
        for it in range(n_it):
            tasks = [*sites, pelvis_o, chest_o, neutral, previous] + ([hold] if it < n_it // 2 else [])
            try:
                v = mink.solve_ik(cfg, tasks, step, SOLVER, damping=1e-3, limits=limits)
            except mink.exceptions.NoSolutionFound:
                v = mink.solve_ik(cfg, tasks, step, SOLVER, damping=1e-3, limits=limits[:1])
            cfg.integrate_inplace(v, step)
        out_q[t] = cfg.q
        resid[t] = np.linalg.norm(cfg.data.site_xpos[sid] - P[t], axis=1)
    hq = np.array([out_q[:, qadr(n)] for n in hinges]).T
    jump = np.degrees(np.abs(np.diff(hq, axis=0))) * fps / 30.0   # per frame at 30 fps
    pops = np.zeros(T, bool)
    pops[1:] = (jump > POP_DEG).any(1)
    names = [""] * model.nq
    for i in range(model.njnt):
        a = model.jnt_qposadr[i]
        n = model.joint(i).name
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            for k, s in enumerate(("x", "y", "z", "qw", "qx", "qy", "qz")):
                names[a + k] = f"{n}_{s}"
        else:
            names[a] = n
    if verbose:
        print(f"  IK {T} frames in {time.time() - t0:.0f}s; residual median {np.median(resid) * 100:.1f} cm, "
              f"p95 {np.percentile(resid, 95) * 100:.1f} cm; pops {pops.sum()}; unstuck {unstuck.sum()} frames; "
              f"chest target on {chest_on.mean():.0%}")
    return dict(qpos=out_q, qpos_names=np.array(names), site_names=np.array(JOINT_NAMES),
                ik_residual=resid, joint_pops=pops, chest_target_on=chest_on, unstuck=unstuck,
                hinge_names=np.array(hinges))


def run(clip: schema.Clip, body: str | None = None) -> dict:
    body = body or clip.name
    xml = BODIES / body / "body.xml"
    if not xml.exists():
        raise SystemExit(f"{xml} missing; run core.bodygen first")
    d = schema.load(clip.clean)
    if "positions_fit2d" not in d:
        raise SystemExit(f"{clip.name}: run core.fit2d first")
    print(f"{clip.name}: retarget onto {body}")
    out = retarget(xml, np.asarray(d["positions_fit2d"], float), d["inferred"].astype(bool), d["fps"])
    path = clip / f"retarget_{body}.npz"
    schema.save(path, d["fps"], JOINT_NAMES, body=body, xml=str(xml), **out)
    print(f"  -> {path}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--body", help="bodies/<body>/body.xml (default: the clip name)")
    a = ap.parse_args(argv)
    run(schema.Clip(a.clip), a.body)


if __name__ == "__main__":
    main()
