"""Print versions, device and model files.  python -m core.check_env"""

from __future__ import annotations

import importlib
import platform
import shutil

from core.schema import MODELS


def main():
    print(f"python {platform.python_version()} on {platform.platform()}")
    for mod in ("numpy", "scipy", "cv2", "mediapipe", "torch", "depth_pro", "matplotlib"):
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:<11} {getattr(m, '__version__', 'ok')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {mod:<11} MISSING ({e})")
    try:
        import torch

        print(f"  mps available: {torch.backends.mps.is_available()}")
    except Exception:  # noqa: BLE001
        pass
    print(f"  ffmpeg: {shutil.which('ffmpeg') or 'not found'}")
    for name in ("pose_landmarker_heavy.task", "pose_landmarker_full.task", "depth_pro.pt"):
        p = MODELS / name
        print(f"  {name:<28} {'%.0f MB' % (p.stat().st_size / 1e6) if p.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
