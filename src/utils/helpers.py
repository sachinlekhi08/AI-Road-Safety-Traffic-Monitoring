"""
src/utils/helpers.py
====================
Shared utility functions used across the project.
"""

import os
import time
import cv2
import sys


def get_device():
    """
    Return the best available PyTorch device string.
    Prefers Apple MPS → CUDA → CPU.
    """
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def ensure_dir(path: str) -> str:
    """Create directory (and parents) if it does not exist. Returns path."""
    os.makedirs(path, exist_ok=True)
    return path


def allowed_video_extension(filename: str, allowed: set) -> bool:
    """Return True if *filename* has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in allowed
    )


def safe_filename(filename: str) -> str:
    """
    Return a timestamped, space-free version of *filename* to avoid collisions
    and path-traversal issues.

    Example:
        'my video.mp4'  →  '1720000000_my_video.mp4'
    """
    base, ext = os.path.splitext(filename)
    base = "".join(c if c.isalnum() or c in "-_." else "_" for c in base)
    ts = int(time.time())
    return f"{ts}_{base}{ext}"


def format_duration(seconds: float) -> str:
    """Return a human-readable 'mm:ss' string from *seconds*."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def draw_label(
    frame,
    text: str,
    x: int,
    y: int,
    color=(0, 200, 0),
    bg_color=(0, 0, 0),
    font_scale: float = 0.55,
    thickness: int = 1,
):
    """
    Draw a label with a filled background rectangle on *frame*.
    Uses OpenCV — no external dependencies.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # Background rectangle
    cv2.rectangle(frame, (x, y - th - baseline - 2), (x + tw + 2, y + baseline), bg_color, cv2.FILLED)
    # Text
    cv2.putText(frame, text, (x + 1, y), font, font_scale, color, thickness, cv2.LINE_AA)


def get_class_color(class_id: int) -> tuple:
    """Return a consistent BGR colour for a given COCO class ID."""
    palette = {
        0:  (255, 178, 102),   # person  – light blue
        1:  (102, 255, 178),   # bicycle – green-ish
        2:  (0,   200, 255),   # car     – yellow
        3:  (255,  51, 153),   # motorcycle – pink
        5:  (51,  153, 255),   # bus     – orange
        7:  (153,  51, 255),   # truck   – purple
        9:  (0,   255,   0),   # traffic light – bright green
    }
    return palette.get(class_id, (200, 200, 200))

