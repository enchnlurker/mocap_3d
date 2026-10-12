"""Video reading and the MediaPipe pose models shared by the capture and depth stages."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.schema import MODELS


def info(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    out = dict(path=str(path),
               width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
               fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
               n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return out


def frames(path: str | Path, indices=None):
    """Yield (index, RGB uint8 frame). indices=None reads every frame."""
    cap = cv2.VideoCapture(str(path))
    wanted = None if indices is None else set(int(i) for i in indices)
    last = None if wanted is None else max(wanted)
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok or (last is not None and i > last):
            break
        if wanted is None or i in wanted:
            yield i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()


def evenly_spaced(n_frames: int, n: int) -> np.ndarray:
    n = min(n, n_frames)
    return np.unique(np.round(np.linspace(0, n_frames - 1, n)).astype(int))


def pose_landmarker(video_mode: bool, model: str = "heavy", segmentation: bool = False):
    from mediapipe.tasks.python import BaseOptions, vision

    path = MODELS / f"pose_landmarker_{model}.task"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; see README (models)")
    mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(path)), running_mode=mode,
        num_poses=1, output_segmentation_masks=segmentation)
    return vision.PoseLandmarker.create_from_options(opts)


def mp_image(rgb: np.ndarray):
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


class PersonMasker:
    """Per-frame person segmentation (MediaPipe pose mask), used to keep the
    floor search off the body and to prefer body pixels for joint depth."""

    def __init__(self, model: str = "heavy"):
        self.lm = pose_landmarker(video_mode=False, model=model, segmentation=True)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        res = self.lm.detect(mp_image(rgb))
        if not res.segmentation_masks:
            return np.zeros(rgb.shape[:2], bool)
        return res.segmentation_masks[0].numpy_view().squeeze() > 0.5

    def close(self):
        self.lm.close()
