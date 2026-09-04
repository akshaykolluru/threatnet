from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import cv2
import numpy as np

from .vector_search import EMBEDDING_DIMENSION


@dataclass(frozen=True)
class FaceDescriptor:
    """A reviewable image descriptor with enough provenance to reproduce a comparison."""

    vector: np.ndarray
    bbox: tuple[int, int, int, int]
    backend: str
    face_detected: bool


class FaceEmbeddingExtractor:
    """Extract 512-d face embeddings with InsightFace when available.

    The local fallback uses OpenCV detection plus a normalized visual descriptor. It
    keeps the V2 demo runnable without downloading a model, while marking its output
    clearly so it cannot be mistaken for a biometric identity decision.
    """

    def __init__(self) -> None:
        self._insightface_app: Any | None = None
        self._insightface_attempted = False
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade_factory = getattr(cv2, "CascadeClassifier", None)
        # OpenCV 5 wheels can omit the legacy Haar wrapper. InsightFace remains the
        # preferred detector; local visual fallback still works when it is absent.
        self._cascade = cascade_factory(str(cascade_path)) if cascade_factory and cascade_path.exists() else None

    def extract(self, image: np.ndarray) -> list[FaceDescriptor]:
        """Return detected faces or one visual fallback descriptor for review workflows."""

        if image is None or image.size == 0:
            raise ValueError("The supplied image could not be decoded")

        insightface_faces = self._extract_with_insightface(image)
        if insightface_faces:
            return insightface_faces

        rectangles = self._detect_with_opencv(image)
        if rectangles:
            return [
                FaceDescriptor(
                    vector=self._opencv_visual_descriptor(image, rectangle),
                    bbox=rectangle,
                    backend="opencv_visual_descriptor",
                    face_detected=True,
                )
                for rectangle in rectangles
            ]

        height, width = image.shape[:2]
        # Keep demo/test ingestion operational, but make the weaker signal explicit.
        fallback = self._center_crop_bbox(width, height)
        return [
            FaceDescriptor(
                vector=self._opencv_visual_descriptor(image, fallback),
                bbox=fallback,
                backend="opencv_visual_descriptor_fallback",
                face_detected=False,
            )
        ]

    def extract_reference(self, content: bytes) -> FaceDescriptor:
        """Decode a reference photo and choose the first detected face-like descriptor."""

        encoded = np.frombuffer(content, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        descriptors = self.extract(image)
        if not descriptors:
            raise ValueError("No usable face or visual descriptor could be extracted from the reference photo")
        return descriptors[0]

    def _extract_with_insightface(self, image: np.ndarray) -> list[FaceDescriptor]:
        app = self._get_insightface_app()
        if app is None:
            return []
        try:
            faces = app.get(image)
        except Exception:
            return []

        descriptors: list[FaceDescriptor] = []
        for face in faces:
            vector = np.asarray(getattr(face, "normed_embedding", getattr(face, "embedding", [])), dtype=np.float32)
            if vector.size != EMBEDDING_DIMENSION:
                continue
            normalized = self._normalize(vector)
            bbox_values = np.asarray(getattr(face, "bbox", []), dtype=np.float32).astype(int).tolist()
            if len(bbox_values) != 4:
                continue
            descriptors.append(
                FaceDescriptor(
                    vector=normalized,
                    bbox=tuple(int(value) for value in bbox_values),
                    backend="insightface_buffalo_l",
                    face_detected=True,
                )
            )
        return descriptors

    def _get_insightface_app(self) -> Any | None:
        if self._insightface_attempted:
            return self._insightface_app
        self._insightface_attempted = True
        try:
            from insightface.app import FaceAnalysis  # type: ignore[import-not-found]

            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=-1, det_size=(320, 320))
            self._insightface_app = app
        except Exception:
            self._insightface_app = None
        return self._insightface_app

    def _detect_with_opencv(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self._cascade is None or self._cascade.empty():
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        return [tuple(int(value) for value in rectangle) for rectangle in detected]

    def _opencv_visual_descriptor(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = self._clamp_bbox(bbox, image.shape[1], image.shape[0])
        crop = image[y : y + height, x : x + width]
        if crop.size == 0:
            raise ValueError("The face crop is empty")
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        # 32 x 16 grayscale samples are a deterministic 512-d local visual descriptor.
        descriptor = cv2.resize(equalized, (32, 16), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
        descriptor = descriptor - float(descriptor.mean())
        return self._normalize(descriptor)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            # Constant visual frames still need a deterministic non-zero descriptor.
            vector = np.ones(EMBEDDING_DIMENSION, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
        return vector.astype(np.float32) / norm

    @staticmethod
    def _center_crop_bbox(width: int, height: int) -> tuple[int, int, int, int]:
        crop_width = max(1, int(width * 0.72))
        crop_height = max(1, int(height * 0.72))
        return ((width - crop_width) // 2, (height - crop_height) // 2, crop_width, crop_height)

    @staticmethod
    def _clamp_bbox(bbox: tuple[int, int, int, int], image_width: int, image_height: int) -> tuple[int, int, int, int]:
        x, y, width, height = bbox
        x = max(0, min(x, image_width - 1))
        y = max(0, min(y, image_height - 1))
        width = max(1, min(width, image_width - x))
        height = max(1, min(height, image_height - y))
        return x, y, width, height


_extractor: Optional[FaceEmbeddingExtractor] = None


def get_face_extractor() -> FaceEmbeddingExtractor:
    """Return one lazily initialized extractor per API process."""

    global _extractor
    if _extractor is None:
        _extractor = FaceEmbeddingExtractor()
    return _extractor


def inspect_video(
    source: Path,
    output_dir: Path,
    sample_every_seconds: float = 2.0,
    recorded_at: datetime | None = None,
    location: str = "",
) -> dict[str, Any]:
    """Sample CCTV frames and derive 512-d descriptors without storing video bytes in SQL."""

    if sample_every_seconds <= 0:
        raise ValueError("sample_every_seconds must be greater than zero")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("The supplied file could not be opened as a video")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0.0
    step = max(1, int(fps * sample_every_seconds)) if fps else 1
    frames: list[dict[str, Any]] = []
    embeddings: list[dict[str, Any]] = []
    extractor = get_face_extractor()
    index = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                timestamp_seconds = round(index / fps, 3) if fps else 0.0
                frame_name = f"frame-{uuid4().hex[:10]}.jpg"
                frame_path = output_dir / frame_name
                if not cv2.imwrite(str(frame_path), frame):
                    raise ValueError("A sampled CCTV frame could not be written to storage")
                captured_at = recorded_at + timedelta(seconds=timestamp_seconds) if recorded_at else None
                descriptors = extractor.extract(frame)
                frame_metadata = {
                    "path": str(frame_path),
                    "frame_index": index,
                    "timestamp_seconds": timestamp_seconds,
                    "face_candidates": len(descriptors),
                }
                frames.append(frame_metadata)
                for descriptor in descriptors:
                    embeddings.append(
                        {
                            "frame_path": str(frame_path),
                            "frame_index": index,
                            "timestamp_seconds": timestamp_seconds,
                            "captured_at": captured_at.isoformat() if captured_at else None,
                            "location": location,
                            "bbox": list(descriptor.bbox),
                            "embedding_backend": descriptor.backend,
                            "face_detected": descriptor.face_detected,
                            "vector": descriptor.vector.tolist(),
                        }
                    )
            index += 1
    finally:
        capture.release()
    return {
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "duration_seconds": round(duration, 3),
        "frames": frames,
        "face_embeddings": embeddings,
        "embedding_dimension": EMBEDDING_DIMENSION,
    }


def public_ingestion_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Remove raw vectors before metadata is stored in evidence rows or returned by the API."""

    return {key: value for key, value in result.items() if key != "face_embeddings"}
