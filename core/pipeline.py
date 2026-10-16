"""Run stages 1a -> 3A2 for one clip.

    python -m core.pipeline clip01 --video path/to/video.mp4
    python -m core.pipeline clip01 --skip-depth          # 1a, 2, 2b, 2c, 3A only
    python -m core.pipeline clip01 --from clean          # re-run from stage 2
    python -m core.pipeline clip01 --from retarget --to playback
"""

from __future__ import annotations

import argparse

from core import (bodygen, capture_mediapipe, clean, depth, env_extract, fit2d, joint_depth, playback, plots,
                  retarget, schema)

STAGES = ["capture", "depth", "joint_depth", "clean", "fit2d", "plots", "bodygen", "retarget", "playback", "env"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--video")
    ap.add_argument("--from", dest="start", default="capture", choices=STAGES)
    ap.add_argument("--to", dest="stop", default=STAGES[-1], choices=STAGES)
    ap.add_argument("--skip-depth", action="store_true", help="skip 1c and 1d")
    ap.add_argument("--camera", default="auto", choices=["auto", "static", "moving"])
    ap.add_argument("--body", help="body name for 3A (default: the clip name)")
    ap.add_argument("--props", default="code", choices=["code", "llm", "auto"], help="3A2 method")
    a = ap.parse_args(argv)
    clip = schema.Clip(a.clip)
    body = a.body or clip.name
    todo = STAGES[STAGES.index(a.start):STAGES.index(a.stop) + 1]
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
    if "bodygen" in todo:
        bodygen.build(body, clip)
    if "retarget" in todo:
        retarget.run(clip, body)
    if "playback" in todo:
        playback.run(clip, body)
    if "env" in todo:
        env_extract.run(clip, body, a.props)


if __name__ == "__main__":
    main()
