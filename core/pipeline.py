"""Run stages 1a -> 2c for one clip.

    python -m core.pipeline clip01 --video path/to/video.mp4
    python -m core.pipeline clip01 --skip-depth          # 1a, 2, 2b, 2c only
    python -m core.pipeline clip01 --from clean          # re-run from stage 2
"""

from __future__ import annotations

import argparse

from core import capture_mediapipe, clean, depth, fit2d, joint_depth, plots, schema

STAGES = ["capture", "depth", "joint_depth", "clean", "fit2d", "plots"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--video")
    ap.add_argument("--from", dest="start", default="capture", choices=STAGES)
    ap.add_argument("--skip-depth", action="store_true", help="skip 1c and 1d")
    ap.add_argument("--camera", default="auto", choices=["auto", "static", "moving"])
    a = ap.parse_args(argv)
    clip = schema.Clip(a.clip)
    todo = STAGES[STAGES.index(a.start):]
    if a.skip_depth:
        todo = [s for s in todo if s not in ("depth", "joint_depth")]
    if "capture" in todo:
        capture_mediapipe.main([a.clip, "--overlay"] + (["--video", a.video] if a.video else []))
    if "depth" in todo:
        depth.main([a.clip])
    if "joint_depth" in todo:
        joint_depth.main([a.clip])
    if "clean" in todo:
        clean.run(clip, use_depth=not a.skip_depth)
    if "fit2d" in todo:
        fit2d.run(clip, a.camera)
    if "plots" in todo:
        plots.main([a.clip])


if __name__ == "__main__":
    main()
