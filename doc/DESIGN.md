# Design: Video Mocap → Retarget → Physics Validation

Status: draft v1, 2026-09-22 Owner: mjsohn Purpose: portfolio demo covering mocap, rigging/retargeting, robotics sim and dataset ETL

---

## 1. Summary

Turn a single video into clean motion, retarget it onto different bodies (humans of other proportions, Unitree G1), and test in physics whether each body can actually perform it. Every clip gets quality metrics and a pass/fail, so the output is a dataset, not a pile of files.

Deterministic code does everything. No LLM in the pipeline (see 3.2).

## 2. Goals / non-goals

Goals

- Cleanup: no jitter, fixed bone lengths, feet locked to a detected floor.
- Retarget onto any body with IK: hands/feet locked, knees/elbows as pole hints.
- Physics check with a per-frame "impossibility meter".
- Per-clip metrics, USD + glTF export, static web viewer.
- Mac-first; 4090 only for better capture and RL.

Non-goals

- Upload-your-video web app, real-time capture, jumps/flips, large dataset (5–10 clips is enough).

## 3. Pipeline

```
video ─┬─ 1a MediaPipe (Mac) → adapter → 2 cleanup → 2b → 2c fit to 2D ─┐
       │      (1c floor depth, 1d per-joint depth feed 2 and 2c)        ├→ clean.npz → 3A retarget → 3B physics → export → viewer
       └─ 1b GVHMR (4090) ──→ SMPL FK → 2 light cleanup ────────────────┘   (merge)     mink IK      tier 1–3     USD       Three.js
                                                                                                     metrics      glTF
```

Two capture branches, one merge point. Everything from 3A on reads only `clean.npz` and never knows which branch produced it.

| Stage | Does | Runs on |
|---|---|---|
| 1a Capture: MediaPipe | video → 33 landmarks + confidence per frame; adapter maps to canonical joints | Mac |
| 1b Capture: GVHMR | video → SMPL shape + rotations + root, gravity-aligned; SMPL FK → canonical joints | 4090, WSL2 Ubuntu |
| 1c Depth (optional) | Depth Pro on 8 evenly spaced frames (default) → floor plane from ground pixels (camera pitch/height), body distance, torso surface samples. `--stride 1` for per-frame | Mac (MPS, ~3.4 s/frame; ~35 s per clip sparse) |
| 1d Joint depth (for 2c) | `core/joint_depth.py`: Depth Pro on every 2nd frame, nearest-surface depth at each landmark's pixel → `depth_joints.npz`. Leaves `depth.npz` alone | Mac (~4.5 min per clip) |
| 2 Cleanup | MediaPipe: swap fix, mask (joints hidden *in frame* keep MediaPipe's guess, flagged `occluded`), gap fill, zero-phase smoothing, floor plane, constant ground offset, 2D hip bob/sway, contacts, bone lengths. GVHMR: foot-skate fix only | Mac |
| 2b Proportions | direction-preserving rebuild with ratio lengths → `positions_proportional` (start point / prior for 2c) | Mac |
| 2c Fit to 2D | `core/fit2d.py`: per clip, solve pose + camera + one length per bone so joints project onto MediaPipe 2D, with per-joint relative depth, 2b as a weak prior → `positions_fit2d` (best result so far) | Mac (~15–35 s per clip) |
| 3A Retarget | body generator + weighted full-body IK → joint angles | Mac |
| 3B Physics | tier 1 balance check, tier 2 PD + pelvis spring, tier 3 RL (optional) | Mac (1–2) / 4090 (3) |
| Export | UsdSkel, robot articulation USD, glTF | Mac |
| Viewer | video + human + bodies side by side, synced metric plots | Mac, GitHub Pages |

### 3.1 Canonical skeleton (merge contract)

20 joints both sources can fill (defined in `core/skeleton.py`): `pelvis, spine, neck, head, l/r_hip, l/r_knee, l/r_ankle, l/r_heel, l/r_toe, l/r_shoulder, l/r_elbow, l/r_wrist`. Heels are included for contact detection (heel strike vs toe-off).

| Joint | MediaPipe | GVHMR |
|---|---|---|
| Most joints | direct landmark | SMPL FK joint |
| pelvis | midpoint of hips | SMPL pelvis |
| neck | midpoint of shoulders | SMPL neck |
| spine | interpolated pelvis → neck | SMPL spine2 |
| head | midpoint of ears | SMPL head |

Rules

- Build the target body from the same source it tracks. MediaPipe hips are surface points (wider than SMPL joint centers), so proportions differ per source. Never mix.
- `clean.npz` records `source` (`mediapipe` / `gvhmr`) for the M11 comparison.
- GVHMR joint rotations (forearm/head twist) are dropped for now; optional low-weight orientation targets in IK later.

### 3.2 LLM annotation (not used)

No LLM stage in the pipeline. Asking a vision LLM for activity, floor vs furniture, supports, holds or hidden-limb angles would only shape joints the camera never saw, and the deterministic path covers those cases better:

- captured joints, floor, camera and proportions come from measurement, not from labels;
- a hidden hand is better found by a detector (MediaPipe Hands) than guessed;
- LLM answers vary between runs, cost money and time per clip, need network credentials, and send frames off the machine.

Where an LLM may come in later: semantic labels for contacts/props (M3/M4), always behind code checks.

## 4. Key decisions

| Topic | Decision |
|---|---|
| Sim body format | Author MJCF (from a Python spec), convert to USD for Isaac. Not the reverse. |
| MJCF features | Common subset only: hinge joints, capsules/boxes/convex meshes, position actuators. No tendons or equality constraints. |
| Motion format | npz keyed by joint name, independent of any rig |
| Units / axes | meters, radians, Z-up, quaternions wxyz (convert to degrees only at the USD boundary) |
| Retarget | weighted IK priorities: feet in contact > pelvis > hands > rest; knee/elbow low weight |
| Contacts | per effector: world-space (floor, table) or body-relative (hand on hip) |
| Physics tiers | 1 CoM-in-support check, 2 PD tracking + pelvis spring (log helper force), 3 RL policy (4090 only) |
| G1 model | Menagerie MJCF, converted ourselves so both sims use the same model |
| LLM role | None in the pipeline (3.2). If one returns, it labels only and code verifies; never coordinates |
| Packaging | one repo; `core` (all Mac code), `capture` (CUDA), `isaac` (Isaac Lab); machines share files only |
| 4090 PC (Windows) | GVHMR in WSL2 Ubuntu (official install is Linux-only: pytorch3d wheel is linux_x86_64). Isaac Sim/Lab native Windows. Skip DPVO. |
| Partial body | Joints never visible stay NaN (not guessed; `tracked` mask). Feet seen (visible, not just guessed in frame) in < 50% of frames → no foot floor fit, ground snap or foot contacts. Floor direction: depth > feet > none. Body height: feet > depth hip height > unknown. The body never sinks below the floor: if the depth hip height would put any joint (inferred feet included) under it, the body is lifted. |
| Hidden limbs (full body) | `core/infer_hidden.py` fills never-seen joints as a flagged prior (`inferred [T,J]`), never as capture. Legs: body-relative hip flexion/abduction + knee flexion from activity rules (lie / sit / stand from geometry), attached to the captured pelvis frame; hidden wrist continues the upper arm; long gaps interpolated. Inferred bone lengths = anthropometric × height from trunk. Inferred legs are rigidly swung above the support plane. Metrics use captured joints only. |
| Proportions (stage 2b) | `core/proportions.py`, on by default: direction-preserving rebuild from the pelvis with generic adult ratios × height (trunk / 0.288), left = right; widths, head and foot follow MediaPipe's landmark spacing (hips 0.40 × trunk, shoulders 0.64 × trunk); floor re-snap for clips with feet. Saved as a separate `positions_proportional`; `positions` untouched. End points move, so contacts are 3A's job. |
| Depth floor search | Whole image, pixels whose local normal is level; lowest parallel surface = floor (sofa seats and table tops lose). Clip floor valid only if ≥ 50% of sampled frames find one. |
| Depth (1c) | Use for the floor direction only; ground pixels give a more reliable tilt than the feet. Torso lean from body-surface depth is off by default: at ~3 m it is inside Depth Pro's error. |
| Floor fit (no depth) | RANSAC on the lowest foot point per frame; default pitch-only (no sideways tilt). A free plane picks up false sideways tilt from left/right leg-length errors. |
| Ground snap | `ground_mode="constant"`: ONE vertical offset per clip (median lower ankle – 0.039 H). A per-frame snap would turn heel/toe noise into body bounce. Feet meet the floor later via IK, not by moving the body |
| Hip-centre motion | bob (Z) + sway (X) from MediaPipe's 2D hip track, band-passed 0.4–6 Hz, px → m at the depth distance. No forward travel (Y = 0) |
| Occluded vs off-screen | low visibility + presence ≥ 0.5 + 2D inside the frame (3 % margin) = hidden in frame → keep MediaPipe's guess (`occluded`). Off-screen → rules. A joint kept in < 50 % of frames loses its occluded samples (no revival from edge frames) |
| Stage 2c (fit to 2D) | Lengths: one per bone group per clip (L = R), pulled to 2b (±12 %). Directions free, 2b prior weak (0.5) for captured bones, strong for inferred. Shoulder/hip/spine pairs tied by construction. Knees don't bend forward; knees/ankles stay ≥ 5 cm apart along the hip axis (legs don't cross); hip–shoulder twist ≤ 30°; hip/shoulder axes held near 2b (~9°) and turn smoothly; knees don't twist sideways (≤ 15°), hip line ≤ 15° from square to the spine, upright feet stay under the hips (≤ 12 cm sideways). Depth: relative to the frame's other visible joints (σ 3 cm); joint covered by another limb, the lap (thigh toward camera), the torso, the near side of an overlapping left/right pair, or hidden = lower bound only. Torso toward/away lean held to 2b (σ 0.1). Depth offset anchored over the clip. Static camera (background < 0.5 px/frame) → one camera, body moves as fitted; moving camera → on 2b's pelvis track |
| Frontend | static read-only viewer, built last |
| Cloud | only for parallel tier 3 runs; RTX GPUs only (g6e), personal AWS account |

## 5. Data contract

One file per stage in `data/<clip>/`. Every file carries `schema_version`, `fps`, `joint_names`.

| File | Contents |
|---|---|
| `capture_mediapipe.npz` | `landmarks [T,33,3]` world, `landmarks_2d [T,33,2]`, `confidence [T,33]`, video info |
| `capture_gvhmr.npz` | SMPL `betas`, `body_pose`, `global_orient`, `transl`, predicted contacts, video info |
| `depth.npz` | per processed frame: `floor_normal_cam`, `floor_d`, `camera_pitch_deg`, `torso_points_cam`, `pelvis_distance_m`, `focal_px`, 1/4-res `depth_maps` |
| `clean.npz` (2b keys) | `positions_proportional [T,20,3]`, `bone_lengths_proportional`, `proportional_height_m` |
| `clean.npz` (2 keys, new) | `occluded [T,20]`, `uv_2d [T,20,2]` (normalized, swap-consistent), `presence [T,20]`, `root_motion [T,3]` |
| `clean.npz` (2c keys, added by `fit2d.py`) | `positions_fit2d [T,20,3]`, `bone_lengths_fit2d [19]`, `fit2d_camera_pos [T,3]`, `fit2d_focal_px`; summary in `fit2d_depth_summary.json`. Re-run `fit2d.py` after `clean.py` |
| `depth_joints.npz` | `frames`, `z_surface [N,33]` nearest-surface depth per landmark, `on_mask`, `in_frame`, `focal_px`, 1/4-res `depth_maps` |
| `clean.npz` | canonical `positions [T,20,3]` floor at Z = 0, `positions_raw` (same frame, uncleaned), `root_pos [T,3]`, `contacts [T,4]` (l/r heel/toe), `bone_names/bone_pairs/bone_lengths` (symmetric medians), `floor_R/floor_offset`, `dropped/filled/swapped` masks, `source` |
| `retarget_<body>.npz` | `qpos [T,nq]`, `joint_names`, per-frame IK residuals |
| `sim_<body>_<tier>.npz` | simulated `qpos`, `helper_force [T]`, `fell` flag, metrics |
| `metrics.json` | all metrics per body/tier + pass/fail + reason |

Bodies live in `bodies/<name>/`: `spec.yaml` (source of truth) → `body.xml` (MJCF) → `body.usd`.

## 6. Metrics (quality gate)

| Metric | Stage | Pass (initial guess, tune later) |
|---|---|---|
| Bone length variation | 3A (reported in 2) | < 1% (stage 2 only measures; the fixed-length body locks it) |
| Foot skate during contact | M6, 3B | < 2 cm |
| Floor penetration | 2, 3B | < 1 cm |
| IK residual (effectors) | 3A | < 3 cm |
| Joint limit violations | 3A | 0 frames |
| Helper force | 3B tier 2 | < 10% body weight (mean) |
| Tracking error vs reference | 3B | < 5 cm mean |
| Fell | 3B | no |

## 7. Repo layout


```
rnd_mocap_sim/
  core/        schema, cleanup, bodygen, retarget, physics, metrics, export, cli
  capture/     GVHMR wrapper (4090, WSL2, own env)
  isaac/       Isaac Lab tier 3 tasks (4090, own env)
  bodies/      body specs + generated MJCF/USD
  configs/     per-clip / per-body YAML
  viewer/      Three.js static site
  doc/         design
  log/         daily logs + glossary
  data/        gitignored
```

## 8. Milestones and progress

Legend: [ ] todo, [~] in progress, [x] done

| # | Milestone | Deliverable | Machine | Status |
|---|---|---|---|---|
| M0 | Setup | project folders, uv env, test clip picked | Mac | [~] env done; test clips to pick |
| M1 | Capture | `capture_mediapipe.npz` + adapter + landmark overlay video | Mac | [x] |
| M2 | Cleanup | `clean.npz` + before/after plots; 2b proportions; 2c fit to 2D | Mac | [x] code + synthetic tests; run on test clips |
| M3 | Body + retarget | actor-matched MJCF, `retarget_*.npz`, kinematic playback | Mac | [ ] build on 2c |

