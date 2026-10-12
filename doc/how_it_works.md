# How the current result works (stages 1–2c)

The current result is MediaPipe's 2D keypoints, which are accurate, turned into 3D by a solver. That solver uses bone lengths, Depth Pro depth and MediaPipe's own 3D pose as supporting evidence. Everything before the last stage exists to prepare those inputs.

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
```

(`clean.py` writes `clean.npz` instead when run without depth.)

## Step 1: measure

**1a: MediaPipe.** For each frame it gives 33 landmarks in three forms:

- **2D pixel positions:** accurate, and the basis for everything that follows.
- **3D world positions in metres:** centred on the hips every frame, so the hips never move. The shape is unreliable in depth; arms come out short and tilted.
- **Visibility and presence:** visibility means "is this point seen". Presence means "is it in the image at all". This pair later separates "hidden behind the head" from "off the bottom of the frame".

**1c: floor depth.** Depth Pro (an Apple model that estimates metric depth from one image) runs on 8 frames. From those it finds:

- **Floor direction:** the lowest flat surface among pixels that aren't the person. Sofa seats and table tops lose because they're higher.
- **Camera tilt and height.**
- **Focal length:** used later as the camera lens.

**1d: joint depth.** Depth Pro runs on every 2nd frame. For each landmark it reads the nearest surface in a small patch around its pixel, preferring pixels on the person. That's skin, not the joint centre. If something covers the joint, it's the covering object, which is why 2c treats some readings as "at least this far" only.

## Step 2: clean, proportions, fit

**2: cleanup (`clean.py`).**

1. **Map 33 landmarks to 20 joints.** Most map directly; the pelvis, neck and head are midpoints of the hips, shoulders and ears, and the spine sits halfway between pelvis and neck.
2. **Fix left/right swaps** using 2D continuity between frames.
3. **Drop bad points.** Low-visibility points are dropped, unless they're occluded: present, inside the frame, and kept in at least half the frames. Occluded points keep MediaPipe's guess and are flagged `occluded`. Spikes are dropped either way.
4. **Fill and smooth.** Short gaps are filled with shape-preserving interpolation, then a lag-free low-pass filter runs, heavier on depth.
5. **Level the floor:** rotate the skeleton so the floor from 1c is flat at Z = 0.
6. **Set the height.** One offset for the whole clip puts the planted ankle at ankle height. Clips without seen feet (off-frame, under a blanket or a table) use the depth-measured hip height instead, lifted if that would put any part of the body, inferred feet included, under the floor.
7. **Add hip motion.** Bob and sway come from the 2D hip track, band-passed and converted from pixels to metres at the body's distance. Nothing is added toward or away from the camera.
8. **Fill hidden limbs with rules.** Legs that are never seen get poses from rules for sitting, lying or standing, and are flagged `inferred`.
9. Detect foot contacts and save the per-joint 2D points (`uv_2d`) for 2c.

**2b: proportions.** Body height comes from torso length ÷ 0.288. Every bone is set to a standard fraction of that height, left equal to right. Widths, head and foot use MediaPipe's landmark spacing (hips 0.40 × trunk, shoulders 0.64 × trunk), which sits near the joints rather than at the body's outer edges. The skeleton is then rebuilt outward from the pelvis, keeping every bone's direction. Lengths become realistic, but MediaPipe's directions (including its depth errors) stay. 2b is the start point and the reference pose for 2c.

**2c: the fit (`fit2d.py`)**

Unknowns, solved for all frames at once:

- the pelvis position in camera space, per frame;
- one direction per bone, per frame;
- one length per bone group for the whole clip, with left and right sharing it.

That's roughly 15,000 numbers for a 300-frame clip.

**Fixed structure.** Joint positions are built from the pelvis outward: each joint = parent + length × direction. Both shoulders hang off the neck in opposite directions, both hips off the pelvis, and the two spine halves are in line. So the neck and pelvis stay true midpoints.

**Camera.** The camera is a pinhole at the origin, with Depth Pro's focal length and the principal point at the image centre. A joint's pixel is: pixel = focal × (x, y) / z + centre.

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
| Smoothness | joint acceleration stays low; the pelvis's camera distance even more so | keeps motion clean |
| Knee | knees don't bend forward | only when legs are captured |
| Legs apart | knees and ankles stay at least 5 cm apart along the hip axis, so legs don't cross | only when both legs are captured |
| Twist | hips and shoulders turn against each other by at most 30° | one-sided |
| Knee sideways | the shin stays within 15° of the thigh's bending plane | only when legs are captured |
| Hip tilt | the hip line stays within 15° of square to the spine | one-sided |
| Feet under hips | when upright, each ankle stays within 12 cm of its hip, sideways | fades out as the trunk tilts past 20–45° |
| Turn smoothness | the hip and shoulder axes turn smoothly, so depth noise can't jitter the body's facing | keeps facing clean |

All errors use a soft-L1 cost, so one bad reading can't drag the whole body.

**Solving, in two phases:**

1. **Rigid placement:** the 2b pose is placed as a solid body, one rotation and translation per frame. This gives a good start, and it's also the "before" number in the reports.
2. **Free fit:** directions, pelvis and lengths are all solved together. It starts from the rotated 2b directions, so limbs begin on the right side and don't flip toward or away from the camera.

The optimizer is L-BFGS, a gradient method that handles thousands of unknowns, restarted until the loss stops dropping.

Which depth readings count as "covered": a joint whose pixel lies behind a nearer limb (a ~6 cm thick tube) or behind the torso (the shoulder–hip quad), and the farther of a left/right pair that overlaps on screen. In a side view that is the far hip, far shoulder and far arm, which MediaPipe still marks visible. The check runs on the rigid 2b pose, and again on the fitted pose before a second free fit.

**Output placement.**

- **Static camera** (background moves less than 0.5 px/frame): one offset for the whole clip moves the solved body into the floor frame. The camera stays put, and the body moves exactly as solved.
- **Moving camera:** the solved pose is placed on 2b's hip track each frame. Here the camera-space motion is the camera following the actor, not body motion.
- **Feet:** clips with feet get one more constant ankle offset.
- **Saved:** the camera position, so Blender can overlay the footage exactly.

## Why the result looks right

- 2D pins everything you can see in the image.
- Depth decides what the image can't: toward or away, and how far each joint is in front of or behind the others.
- The prior and lean terms cover only what neither 2D nor depth can tell.
- Lengths are solved per clip, so wrong proportions can't hide as fake depth twists.

The remaining errors are where all three sources are weak: for example a wrist hidden behind the head, or small, blurry ankles and feet.

## Notes

- Without `depth_joints.npz`, 2c still runs, but toward/away then comes only from the 2b prior (raised from 0.5 to 4 in that case). On the synthetic test, no depth gives 11.7 → 10.7 cm MPJPE, and with depth it gives 11.7 → 3.1 cm. Run 1d.
