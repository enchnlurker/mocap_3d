"""Stage 1a: video -> 33 MediaPipe landmarks + confidence per frame.

    python -m core.capture_mediapipe clip01 [--video path.mp4] [--overlay]

Writes capture_mediapipe.npz:
    landmarks     [T,33,3] world metres, hips-centred every frame, camera-aligned
                  axes (x right, y down, z away from camera)
    landmarks_2d  [T,33,2] pixels
    confidence    [T,33]   visibility: "is this point seen"
    presence      [T,33]   "is it in the image at all"
    video info    video_path, width, height, n_frames
Frames without a detection are NaN.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from core import schema, video
from core.skeleton import BONE_PAIRS, JOINT_NAMES, map_mediapipe


def capture(video_path: Path, model: str = "heavy") -> dict:
    vi = video.info(video_path)
    T, W, H, fps = vi["n_frames"], vi["width"], vi["height"], vi["fps"]
    lm3 = np.full((T, 33, 3), np.nan, np.float32)
    lm2 = np.full((T, 33, 2), np.nan, np.float32)
    vis = np.zeros((T, 33), np.float32)
    pres = np.zeros((T, 33), np.float32)
    n = 0
    with video.pose_landmarker(video_mode=True, model=model) as lm:
        for i, rgb in video.frames(video_path):
            if i >= T:  # container frame counts can be short; grow
                grow = i - T + 1
                lm3 = np.concatenate([lm3, np.full((grow, 33, 3), np.nan, np.float32)])
                lm2 = np.concatenate([lm2, np.full((grow, 33, 2), np.nan, np.float32)])
                vis = np.concatenate([vis, np.zeros((grow, 33), np.float32)])
                pres = np.concatenate([pres, np.zeros((grow, 33), np.float32)])
                T = i + 1
            res = lm.detect_for_video(video.mp_image(rgb), int(round(i * 1000 / fps)))
            n = i + 1
            if not res.pose_landmarks:
                continue
            for k, (p, w) in enumerate(zip(res.pose_landmarks[0], res.pose_world_landmarks[0])):
                lm2[i, k] = (p.x * W, p.y * H)
                lm3[i, k] = (w.x, w.y, w.z)
                vis[i, k] = p.visibility or 0.0
                pres[i, k] = p.presence or 0.0
    T = n
    return dict(landmarks=lm3[:T], landmarks_2d=lm2[:T], confidence=vis[:T], presence=pres[:T],
                video_path=str(video_path), width=W, height=H, n_frames=T, fps=fps)


def write_overlay(video_path: Path, cap: dict, out: Path) -> None:
    """Landmark overlay video: canonical bones, joints coloured by visibility."""
    uv = map_mediapipe(cap["landmarks_2d"])
    vis = map_mediapipe(cap["confidence"], "min")
    vi = video.info(video_path)
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), vi["fps"], (vi["width"], vi["height"]))
    for i, rgb in video.frames(video_path):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if i < len(uv) and np.isfinite(uv[i]).all():
            for a, b in BONE_PAIRS:
                cv2.line(bgr, tuple(int(x) for x in uv[i, a]), tuple(int(x) for x in uv[i, b]),
                         (255, 255, 255), 2, cv2.LINE_AA)
            for j in range(len(JOINT_NAMES)):
                v = float(vis[i, j])
                color = (0, int(255 * v), int(255 * (1 - v)))  # green seen, red not seen
                cv2.circle(bgr, tuple(int(x) for x in uv[i, j]), 5, color, -1, cv2.LINE_AA)
        cv2.putText(bgr, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        wr.write(bgr)
    wr.release()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--video", help="source video; copied to data/<clip>/source.<ext> if given")
    ap.add_argument("--model", default="heavy", choices=["heavy", "full", "lite"])
    ap.add_argument("--overlay", action="store_true", help="also write overlay_mediapipe.mp4")
    a = ap.parse_args(argv)
    clip = schema.Clip(a.clip)
    clip.dir.mkdir(parents=True, exist_ok=True)
    if a.video:
        src = Path(a.video)
        dst = clip / f"source{src.suffix.lower()}"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    vpath = clip.video
    cap = capture(vpath, a.model)
    fps = cap.pop("fps")
    schema.save(clip.capture, fps, [f"mp{k}" for k in range(33)], **cap)
    found = np.isfinite(cap["landmarks"][:, 0, 0]).mean()
    print(f"{clip.name}: {cap['n_frames']} frames, pose found in {found:.0%} -> {clip.capture}")
    if a.overlay:
        write_overlay(vpath, cap, clip / "overlay_mediapipe.mp4")


if __name__ == "__main__":
    main()
