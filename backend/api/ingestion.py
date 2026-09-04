from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2


def inspect_video(source: Path, output_dir: Path, sample_every_seconds: float = 2.0) -> dict:
    """Extract lightweight, reviewable metadata without placing video bytes in SQL."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("The supplied file could not be opened as a video")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0.0
    step = max(1, int(fps * sample_every_seconds)) if fps else 1
    frames = []
    index = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            frame_name = f"frame-{uuid4().hex[:10]}.jpg"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            frames.append({"path": str(frame_path), "frame_index": index, "timestamp_seconds": round(index / fps, 3) if fps else 0})
        index += 1
    capture.release()
    return {"fps": round(fps, 3), "frame_count": frame_count, "duration_seconds": round(duration, 3), "frames": frames}
