# rnd_mocap_sim

Single video → clean motion → retarget onto other bodies → physics validation.
Design: [doc/DESIGN.md](doc/DESIGN.md). How stages 1–2c work: [doc/how_it_works.md](doc/how_it_works.md).

Status: M0–M2 done (capture, depth, cleanup, proportions, fit to 2D). M3 next.

## Setup (Mac)

```sh
brew install uv ffmpeg
brew install --cask blender   # for core.blender_view
uv sync                      # Python 3.12 venv with mediapipe, torch (MPS), Depth Pro
mkdir -p data/models
curl -L -o data/models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
curl -L -o data/models/depth_pro.pt https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt   # 1.9 GB
uv run python -m core.check_env
```

## Run a clip

```sh
uv run python -m core.pipeline clip01 --video ~/Movies/clip01.mp4   # 1a → 1c → 1d → 2/2b → 2c → check images
```

or stage by stage (each reads/writes `data/<clip>/`):

| Stage | Command | Output |
|---|---|---|
| 1a | `python -m core.capture_mediapipe clip01 --video v.mp4 --overlay` | `capture_mediapipe.npz`, `overlay_mediapipe.mp4` |
| 1c | `python -m core.depth clip01` (`--stride 1` per-frame) | `depth.npz` |
| 1d | `python -m core.joint_depth clip01` | `depth_joints.npz` |
| 2 + 2b | `python -m core.clean clip01` | `clean_depth.npz` (or `clean.npz` with `--no-depth`), `clean_before_after.png` |
| 2c | `python -m core.fit2d clip01` | adds `positions_fit2d` … to the clean file, `fit2d_depth_summary.json` |
| checks | `python -m core.plots clip01` / `--all` | `camera_check.jpg`, `fullbody_check.jpg`, `data/camera_check_all.jpg` |
| 3D view | `python -m core.blender_view clip01 [--save clip01.blend]` | opens Blender: animated skeletons, floor, solved camera over the video |

Re-run 2c after 2.

## Tests

`uv run pytest` builds a synthetic walking-in-place clip seen by a known pitched camera and checks the swap fix, spike drop, floor fit, ground snap, 2b, hidden-leg inference, and that 2c recovers the true 3D pose.

## Layout

```
core/     all Mac code
bodies/   body specs + generated MJCF/USD (M3)
configs/  per-clip / per-body YAML
doc/      design
log/      daily logs + glossary
tests/
data/     gitignored: clips, models
```
