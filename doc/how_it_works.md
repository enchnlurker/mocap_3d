# How the current result works (stages 1–3A2)

The motion is MediaPipe's 2D keypoints, which are accurate, turned into 3D by a solver (2c). That solver uses bone lengths, per-joint depth (Sapiens2, scaled to Depth Pro) and MediaPipe's own 3D pose as supporting evidence. Everything before 2c prepares those inputs; everything after it puts the motion on a MuJoCo body (3A) and gives that body something to sit, lie on or hold (3A2).

## Data flow

```text
source.mp4
 ├─ 1a capture_mediapipe.py → capture_mediapipe.npz   2D points, 3D points, visibility, presence
 ├─ 1c depth.py             → depth.npz               floor direction, camera height, focal length (8 frames)
 └─ 1d joint_depth.py       → depth_joints.npz        depth at every landmark pixel (every 2nd frame)
        │
 2  clean.py   (reads 1a + 1c) → clean_depth.npz:
      positions               stage 2: cleaned MediaPipe 3D
      positions_proportional  stage 2b: same directions, realistic lengths
 2c fit2d.py   (reads clean + 1a + 1c + 1d + video) → adds to clean_depth.npz:
      positions_fit2d, bone_lengths_fit2d, fit2d_camera_pos
 3A bodygen.py  (2c lengths) → bodies/<clip>/spec.json → body.xml
    retarget.py (positions_fit2d + body) → retarget_<body>.npz        joint angles
    playback.py → playback_<body>.mp4, bodyshapes_<body>.npz          posed geoms per frame
 3A2 env_extract.py (body shapes + depth + video) → props_<body>.json, scene_<body>.xml
```

(`clean.py` writes `clean.npz` instead when run without depth.)

## Step 1: measure

**1a: MediaPipe.** For each frame it gives 33 landmarks in three forms:

- **2D pixel positions:** accurate, and the basis for everything that follows.
- **3D world positions in metres:** centred on the hips every frame, so the hips never move. The shape is unreliable in depth; arms come out short and tilted.
- **Visibility and presence:** visibility means "is this point seen". Presence means "is it in the image at all". This pair later separates "hidden behind the head" from "off the bottom of the frame".

Frames where MediaPipe finds nobody are skipped by 1c and 1d.

**1c: floor depth.** Depth Pro (an Apple model that estimates metric depth from one image) runs on 8 frames. Each full frame is cached, so a re-run only computes frames it hasn't seen. From those it finds:

- **Floor direction:** the lowest flat surface among pixels that aren't the person. Sofa seats and table tops lose because they're higher.
- **Camera tilt and height.**
- **Focal length:** used later as the camera lens.

**1d: joint depth.** Sapiens2 (a Meta model for people that predicts a 3D point per pixel) runs on every 4th frame (every 2nd gives the same 2c result at twice the time) plus 1c's frames; only its depth (Z) is used. Its own scale is ignored: Sapiens2 also runs on 1c's 8 frames, and the ratio of Depth Pro's median body depth to Sapiens2's gives one scale factor for the clip. For each landmark it reads the nearest surface in a small patch around its pixel, using only pixels on the person; a patch with no person pixels (a cup or table in front) gives no reading, because Sapiens2's depth off the body isn't trusted. On 1c's frames, a joint with no person pixels in its patch gets Depth Pro's reading instead (a fallback, counted at half weight in 2c). The reading is skin, not the joint centre. If another body part or a held object covers the joint, it's that, which is why 2c treats some readings as "at least this far" only. A re-run keeps the frames already computed.

## Step 2: clean, proportions, fit

**2: cleanup (`clean.py`).**

1. **Map 33 landmarks to 20 joints.** Most map directly; the pelvis, neck and head are midpoints of the hips, shoulders and ears, and the spine sits halfway between pelvis and neck. **Drop shrunk-body frames:** a person walking very close to the camera, with only hips and legs in shot, still comes back as a whole, confident person squeezed into the visible part. Each frame's on-screen size per metre comes from its least foreshortened bone (projection can only shorten a bone); a frame below 1/1.6 of the surrounding second's size counts as undetected. Real turns dip at most about 1.4×.
2. **Fix left/right swaps** using 2D continuity between frames.
3. **Drop bad points.** Low-visibility points are dropped, unless they're occluded: present, inside the frame, and kept in at least half the frames. Occluded points keep MediaPipe's guess and are flagged `occluded`. Spikes are dropped either way.
4. **Fill and smooth.** Short gaps are filled with shape-preserving interpolation, then a lag-free low-pass filter runs, heavier on depth.
5. **Level the floor:** rotate the skeleton so the floor from 1c is flat at Z = 0.
6. **Set the height.** One offset for the whole clip puts the planted ankle at ankle height. Clips without seen feet (off-frame, under a blanket or a table) use the depth-measured hip height instead, lifted if that would put any part of the body, inferred feet included, under the floor.
7. **Add hip motion.** Bob and sway come from the 2D hip track, band-passed and converted from pixels to metres at the body's distance. Nothing is added toward or away from the camera.
8. **Fill hidden limbs with rules.** Legs that are never seen get poses from rules for sitting, lying or standing, and are flagged `inferred`.
9. Detect foot contacts and save the per-joint 2D points (`uv_2d`) for 2c.

Every threshold lives in `CleanConfig`; `configs/<clip>/clean.json` overrides any of them for one clip. Each run writes `clean_summary.json` (counts, the depth reference, torso lean stats, the config used). The torso's lean toward the camera is compared with Depth Pro's torso surface; `--lean-fix` rotates the upper body to match, off by default because at ~3 m the difference is inside Depth Pro's error.

**2b: proportions.** Body height comes from torso length ÷ 0.288. Every bone is set to a standard fraction of that height, left equal to right. Widths, head and foot use MediaPipe's landmark spacing (hips 0.40 × trunk, shoulders 0.64 × trunk), which sits near the joints rather than at the body's outer edges. The skeleton is then rebuilt outward from the pelvis, keeping every bone's direction. Lengths become realistic, but MediaPipe's directions (including its depth errors) stay. 2b is the start point and the reference pose for 2c.

**2c: the fit (`fit2d.py`)**

Unknowns, solved for all frames at once:

- the pelvis position in camera space, per frame;
- one direction per bone, per frame;
- one length per bone group for the whole clip, with left and right sharing it.

That's roughly 15,000 numbers for a 300-frame clip.

**Fixed structure.** Joint positions are built from the pelvis outward: each joint = parent + length × direction. Both shoulders hang off the neck in opposite directions, both hips off the pelvis, and the two spine halves are in line. So the neck and pelvis stay true midpoints.

**Camera.** The camera is a pinhole at the origin, with Depth Pro's focal length (from 1c) and the principal point at the image centre. A joint's pixel is: pixel = focal × (x, y) / z + centre.

**Cost terms.** The solver minimizes their sum; each term encodes one fact:

| Term | What it says | Strength |
|---|---|---|
| 2D | projected joints sit on MediaPipe's 2D points | main term; about 7 px of noise expected; hidden joints count about ⅓ |
| Depth (visible) | a joint's depth relative to the other visible joints in the frame matches the depth map | about 3 cm of noise expected; relative, so Depth Pro's absolute scale doesn't matter |
| Depth (covered) | a joint behind a hand, another limb or the lap is at least as far as that surface | one-sided |
| Depth anchor | the gap between the fit and the depth map stays constant over the clip | stops the body drifting toward or away from the camera |
| Prior | bone directions stay near 2b's | weak for captured bones, strong for inferred legs |
| Lean | the torso's toward/away tilt stays near 2b's | about 6° of slack |
| Facing | the hip and shoulder axes stay near 2b's, so the body doesn't spin to fake a narrower width | about 9° of slack |
| Lengths | each bone length stays near 2b's | about 12% of slack |
| Widths | shoulder and hip width = the widest on-screen span (95th percentile, at the fit's own distance), if the person faced the camera at some point (span ≥ 0.75 of 2b's width); otherwise 2b's ratio width | about 4% of slack; stops the solver faking a body turn to get the projected width right |
| Smoothness | joint acceleration stays low; the pelvis's camera distance even more so | keeps motion clean |
| Knee | knees don't bend forward | only when legs are captured |
| Legs apart | knees and ankles stay at least 5 cm apart along the hip axis, so legs don't cross | only when both legs are captured |
| Twist | hips and shoulders turn against each other by at most 30° | one-sided |
| Knee sideways | the shin stays within 15° of the thigh's bending plane | only when legs are captured |
| Hip tilt | the hip line stays within 15° of square to the spine | one-sided |
| Feet under hips | when upright, each ankle stays within 12 cm of its hip, sideways | fades out as the trunk tilts past 20–45° |
| Turn smoothness | the hip and shoulder axes turn smoothly, so depth noise can't jitter the body's facing | keeps facing clean |
| Covered turn | while a shoulder has no visible depth reading, the shoulder line's turn per frame may change by at most 0.6° per frame | 2D gives how much it turns, not which way; a spin carries on, but a sudden reversal into the mirror pose doesn't |
| Side view | hip adduction ≤ 15°, knee hinge ≤ 35° off the hip line (once the knee bends), pelvis roll ≤ 12° and ankle midpoint ≤ 5 cm sideways from the pelvis when upright | only with captured legs; above what front views show, so they act only when nothing else pins the pose |

All errors use a soft-L1 cost, so one bad reading can't drag the whole body.

**Solving, in two phases:**

1. **Rigid placement:** the 2b pose is placed as a solid body, one rotation and translation per frame. This gives a good start, and it's also the "before" number in the reports.
2. **Free fit:** directions, pelvis and lengths are all solved together. It starts from the rotated 2b directions, so limbs begin on the right side and don't flip toward or away from the camera.

The optimizer is L-BFGS, a gradient method that handles thousands of unknowns, restarted until the loss stops dropping.

Which depth readings count as "covered" (at least this far, only):

- a joint whose pixel lies behind a nearer limb (a ~6 cm thick tube) or behind the torso (the shoulder–hip quad), and the farther of a left/right pair that overlaps on screen. In a side view that is the far hip, far shoulder and far arm, which MediaPipe still marks visible;
- a joint within 12 cm of a hand centre (wrist plus finger points): the reading may be the hand or what it holds (a cup, a bag, a phone), which the person mask may not cover;
- the nearer end of an impossible bone: a bone of length L whose ends are some span apart on screen can only reach √((1.15 L)² − span²) + 8 cm toward or away from the camera, so when its two readings differ by more, one is wrong.

The check runs on the rigid 2b pose, and again on the fitted pose before a second free fit. Depth exists every 4th frame; the frames between get linearly interpolated readings, visible only if both samples were, and covered if either was (a limb that covered a joint at one sample may still cover it a few frames later). The interpolated rows share the samples' total weight, so they add no new evidence.

**Output placement.**

- **Static or moving:** background features outside the person box are tracked in three horizontal bands, and the fastest band decides. In a dolly shot the nearby ground moves fast while distant buildings, often richer in features, barely move, so one median over the whole background would call it static.
- **Static camera** (fastest band under 0.5 px/frame): one offset for the whole clip moves the solved body into the floor frame. The camera stays put, and the body moves exactly as solved.
- **Moving camera:** the solved pose is placed on 2b's hip track each frame. Here the camera-space motion is the camera following the actor, not body motion.
- **Feet:** clips with feet get one more constant ankle offset.
- **Saved:** the camera position, so Blender can overlay the footage exactly.

## Why the result looks right

- 2D pins everything you can see in the image.
- Depth decides what the image can't: toward or away, and how far each joint is in front of or behind the others.
- The prior and lean terms cover only what neither 2D nor depth can tell.
- Lengths are solved per clip, so wrong proportions can't hide as fake depth twists.

The remaining errors are where all three sources are weak: for example a wrist hidden behind the head, or small, blurry ankles and feet.

## Step 3: embody

**3A body (`bodygen.py`).** The 2c bone lengths become a MuJoCo body. `bodies/<clip>/spec.json` is the editable source of truth; the MJCF is generated from it. One body per bone, hinges only (no ball joints), so every axis gets its own human range (knee 0–150°, hip flexion −30 to 125°). Masses are de Leva segment fractions of a BMI-22 body weight. Shapes are mannequin-like: ellipsoid pelvis, trunk and head, capsule limbs, box feet. Every canonical joint has a site with the same name, so 2c targets map one-to-one.

**3A retarget (`retarget.py`).** Each frame, mink IK pulls each site toward its 2c joint:

- **Weights:** the pelvis is strongest (50), so the legs reach and the body doesn't move. Ends (head, wrists, ankles, toes) are medium. Knees, elbows and heels are low: they only hint at the bend (the foot box is rigid; the toe sets it). Inferred joints get ×0.1.
- **Hard limits:** joint ranges, and the four sole corners of each foot box stay at or above z = 0. A small pull toward the previous frame keeps motion continuous.
- **Fixes for specific failures:** a chest orientation target stops the two 3-hinge spine joints winding against each other (only while within 60° of the pelvis); a weak neutral pull on ankle roll, hip twist and spine side-bend/twist, which the targets barely see; a 360°/s speed limit on twist and ankle hinges, which could flip from limit to limit while a limb was straight; and an unstick: an elbow or knee stuck straight while its target bends starts the frame at the target's bend, held there for half the frame's iterations so the shoulder or hip can turn the bend into the right plane.

**Playback (`playback.py`)** needs no physics: it sets the joint angles each frame and renders an mp4 or opens the MuJoCo viewer, with captured targets as green spheres and inferred ones orange. It also exports per-frame body shapes for Blender and 3A2.

**3A2 props (`env_extract.py`).** Collision shapes for what the body touches.

- **Code method:** the body is the ruler. The lowest underside of the hips, chest, legs and head over the clip gives the support heights. Feet on the floor plus a low, still pelvis give a seat box under the hips and upper thighs. Lying gives a box under the body plus a head pad, with edges grown to the depth surface at that height.
- **LLM method** (opt-in, `--method llm`; not hooked up yet): one call gets key frames, a depth image, a 3D point grid and the body parts, and returns Python that builds meshes. It runs in a separate restricted process and is checked; held objects are attached to a body part. Any failure falls back to the code method.

**Viewing tools.** `depth_surface.py` exports one depth frame as a coloured PLY lined up with the Blender camera, to check where a seat really was. `blender_view.py` can show the retargeted body (`--retarget`), the props (`--props`), two clean files side by side (`--compare`), the solved camera over the footage (`--camera`) and a PLY (`--ply`).

## Notes

- Without `depth_joints.npz`, 2c still runs, but toward/away then comes only from the 2b prior (raised from 0.5 to 4 in that case). On the synthetic test, no depth gives 11.4 → 10.3 cm MPJPE, and with depth it gives 11.4 → 2.8 cm. Run 1d.
