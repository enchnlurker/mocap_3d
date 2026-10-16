# rnd_mocap_sim

Single video → clean motion → retarget onto other bodies → physics validation.
Design: [doc/DESIGN.md](doc/DESIGN.md). How stages 1–3A2 work: [doc/how_it_works.md](doc/how_it_works.md).

Status: M0–M3 done (capture, depth, cleanup, proportions, fit to 2D, body + retarget + playback, props). Physics (3B) next.

## Setup (Mac)

```sh
brew install uv ffmpeg
brew install --cask blender   # for core.blender_view
uv sync                      # Python 3.12 venv with mediapipe, torch (MPS), Depth Pro, transformers, mujoco, mink
mkdir -p data/models
curl -L -o data/models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
curl -L -o data/models/depth_pro.pt https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt   # 1.9 GB
uv run python -m core.check_env
```

Stage 1d downloads Sapiens2 (`facebook/sapiens2-pointmap-0.4b`, ~2 GB) into `data/models/hf/` on first run.

Stage 3A2 builds props with the code method. An LLM method exists but isn't hooked up yet (opt-in: `--method llm`, needs `OPENAI_API_KEY` in `.env`).

## Run a clip

```sh
uv run python -m core.pipeline clip01 --video ~/Movies/clip01.mp4   # 1a → 1c → 1d → 2/2b → 2c → checks → 3A → 3A2
uv run python -m core.pipeline clip01 --from bodygen                  # just 3A + 3A2
```

or stage by stage (each reads/writes `data/<clip>/`):

| Stage | Command | Output |
|---|---|---|
| 1a | `python -m core.capture_mediapipe clip01 --video v.mp4 --overlay` | `capture_mediapipe.npz`, `overlay_mediapipe.mp4` |
| 1c | `python -m core.depth clip01` (`--stride 1` per-frame) | `depth.npz`, `cache/depth_pro/` |
| 1d | `python -m core.joint_depth clip01` (needs 1c; `--stride 4`) | `depth_joints.npz` |
| 2 + 2b | `python -m core.clean clip01` (`--config c.json`, `--lean-fix`) | `clean_depth.npz` (or `clean.npz` with `--no-depth`), `clean_summary.json`, `clean_before_after.png` |
| 2c | `python -m core.fit2d clip01` | adds `positions_fit2d` … to the clean file, `fit2d_depth_summary.json` |
| checks | `python -m core.plots clip01` / `--all` | `camera_check.jpg`, `fullbody_check.jpg`, `data/camera_check_all.jpg` |
| 3A body | `python -m core.bodygen clip01` | `bodies/clip01/spec.json` (editable), `bodies/clip01/body.xml` |
| 3A retarget | `python -m core.retarget clip01` | `retarget_clip01.npz` |
| 3A playback | `python -m core.playback clip01` (`--viewer` under `mjpython`, `--scene` with props) | `playback_clip01.mp4`, `bodyshapes_clip01.npz` |
| 3A2 props | `python -m core.env_extract clip01` (`--method code\|llm`) | `props_clip01.json`, `scene_clip01.xml` |
| depth check | `python -m core.depth_surface clip01 [--frame 90]` | `depth_surface_f<frame>.ply` in the world frame |
| 3D view | `python -m core.blender_view clip01 [--camera] [--retarget] [--props] [--compare other] [--ply f.ply] [--save clip01.blend]` | opens Blender: animated skeletons and floor, plus the chosen extras |

Re-run 2c after 2, and 3A after 2c. Per-clip cleanup thresholds go in `configs/<clip>/clean.json` (any `CleanConfig` field).

## Tests

`uv run pytest` builds a synthetic walking-in-place clip seen by a known pitched camera and checks the swap fix, spike drop, shrunk-body check, floor fit, ground snap, 2b, hidden-leg inference, the 2c depth rules (impossible depth, hand cover, carried cover, widths from the image), the banded camera check, and that 2c recovers the true 3D pose. Stage 3 tests build a body, retarget the synthetic walk, find a seat under a synthetic sitting pose, and run the props sandbox.

## Layout

```
core/     all Mac code
bodies/   body specs (spec.json) + generated MJCF
configs/  per-clip overrides (configs/<clip>/clean.json)
doc/      design
log/      daily logs + glossary
tests/
data/     gitignored: clips, models
```
