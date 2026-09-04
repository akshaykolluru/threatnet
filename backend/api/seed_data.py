from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sqlalchemy import func, select

from .contradictions import detect_spatiotemporal_contradictions
from .database import (
    AuditLog,
    Case,
    Entity,
    Event,
    Evidence,
    FaceEmbedding,
    FaceMatch,
    SessionLocal,
)
from .extraction import persist_extraction
from .ingestion import get_face_extractor, inspect_video
from .vector_search import FaceVectorIndex


DEMO_CASE_ID = "CASE-101"
DEMO_TEXT_EVIDENCE_ID = "TXT-V2-001"
DEMO_CCTV_EVIDENCE_ID = "CCTV-V2-001"
DEMO_FACE_MATCH_ID = "FM-V2-001"
DEMO_FACE_MATCH_EVENT_ID = "EVT-FM-V2-001"
DEMO_CAPTURED_AT = datetime(2026, 9, 4, 20, 3)
DEMO_LOCATION = "Jubilee Hills"
DEMO_INTELLIGENCE_NOTE = (
    "Witness note: Arjun Rao stated he was at Banjara Hills at 2026-09-04T20:00:00. "
    "Arjun Rao called Priya Nair at 2026-09-04T20:01:00 from Banjara Hills. "
    "Arjun Rao owns a white hatchback."
)


def _storage_root() -> Path:
    return Path(os.getenv("STORAGE_DIR", str(Path(__file__).resolve().parent.parent / "storage")))


def _storage_reference(path: Path) -> str:
    return (Path("storage") / path.resolve().relative_to(_storage_root().resolve())).as_posix()


def _face_index() -> FaceVectorIndex:
    return FaceVectorIndex(_storage_root() / "face-index")


def ensure_v2_demo_data() -> None:
    """Idempotently seed the V2 narrative, mocked CCTV dataset, match, and alert."""

    assets = _ensure_mock_cctv_assets()
    with SessionLocal() as session:
        if session.get(Case, DEMO_CASE_ID) is None:
            return
        if session.get(Evidence, DEMO_TEXT_EVIDENCE_ID) is None:
            persist_extraction(
                session=session,
                case_id=DEMO_CASE_ID,
                raw_text=DEMO_INTELLIGENCE_NOTE,
                source="Synthetic witness and phone-log note",
                source_type="statement",
                evidence_id=DEMO_TEXT_EVIDENCE_ID,
            )
            session.flush()

        arjun = session.scalar(
            select(Entity).where(
                Entity.case_id == DEMO_CASE_ID,
                func.lower(Entity.canonical_name) == "arjun rao",
            )
        )
        if arjun is None:
            return
        _ensure_demo_frame_embeddings(session, arjun, assets)
        session.flush()
        _ensure_demo_face_match(session, arjun, assets)
        session.flush()
        detect_spatiotemporal_contradictions(session, DEMO_CASE_ID, arjun.id)
        if not session.scalar(
            select(AuditLog).where(
                AuditLog.action == "seeded_v2_intelligence_demo",
                AuditLog.target == DEMO_CASE_ID,
            )
        ):
            session.add(AuditLog(actor="demo-seed", action="seeded_v2_intelligence_demo", target=DEMO_CASE_ID))
        session.commit()


def _ensure_mock_cctv_assets() -> dict[str, Path]:
    """Generate a local mock portrait and AVI clip without downloading biometric data."""

    root = _storage_root() / "cctv" / "v2-demo"
    frames_dir = root / "frames"
    root.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    reference_path = root / "arjun-rao-reference.jpg"
    video_path = root / "mock-jubilee-hills.avi"
    if not reference_path.exists():
        cv2.imwrite(str(reference_path), _mock_portrait())
    if not video_path.exists():
        portrait = _mock_portrait()
        height, width = portrait.shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            5.0,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("The local OpenCV video writer could not create the V2 mock CCTV clip")
        try:
            for index in range(15):
                frame = portrait.copy()
                cv2.putText(
                    frame,
                    f"CAM-04 {index / 5:.1f}s",
                    (8, 228),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (190, 220, 240),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(frame)
        finally:
            writer.release()
    return {
        "root": root,
        "frames_dir": frames_dir,
        "reference_path": reference_path,
        "video_path": video_path,
    }


def _mock_portrait() -> np.ndarray:
    """Create a clearly synthetic portrait for the demo, never a real person's image."""

    image = np.full((240, 240, 3), (34, 44, 58), dtype=np.uint8)
    cv2.rectangle(image, (0, 190), (240, 240), (21, 31, 45), thickness=-1)
    cv2.ellipse(image, (120, 112), (63, 82), 0, 0, 360, (111, 155, 184), thickness=-1)
    cv2.ellipse(image, (120, 60), (58, 29), 0, 180, 360, (31, 45, 59), thickness=-1)
    cv2.circle(image, (96, 104), 8, (21, 31, 38), thickness=-1)
    cv2.circle(image, (145, 104), 8, (21, 31, 38), thickness=-1)
    cv2.ellipse(image, (120, 147), (25, 10), 0, 0, 180, (61, 78, 90), thickness=3)
    cv2.line(image, (120, 112), (116, 133), (62, 87, 105), thickness=3)
    cv2.putText(image, "SYNTHETIC DEMO", (51, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 220, 240), 1)
    return image


def _ensure_demo_frame_embeddings(session: Any, arjun: Entity, assets: dict[str, Path]) -> None:
    """Sample mock CCTV once and keep the durable index recoverable on later starts."""

    existing = list(
        session.scalars(
            select(FaceEmbedding).where(FaceEmbedding.evidence_id == DEMO_CCTV_EVIDENCE_ID).order_by(FaceEmbedding.id)
        ).all()
    )
    index = _face_index()
    if not existing:
        sampled = inspect_video(
            assets["video_path"],
            assets["frames_dir"],
            sample_every_seconds=2.0,
            recorded_at=DEMO_CAPTURED_AT,
            location=DEMO_LOCATION,
        )
        session.add(
            Evidence(
                id=DEMO_CCTV_EVIDENCE_ID,
                case_id=DEMO_CASE_ID,
                title="Synthetic CCTV sample: Jubilee Hills CAM-04",
                type="cctv",
                notes="Synthetic demo footage. Frame similarity is an indicator only and requires verification.",
                confidence=None,
                source=_storage_reference(assets["video_path"]),
                metadata_json=json.dumps(
                    {
                        "synthetic_demo": True,
                        "recorded_at": DEMO_CAPTURED_AT.isoformat(),
                        "location": DEMO_LOCATION,
                        "frame_count": len(sampled["frames"]),
                        "embedding_dimension": sampled["embedding_dimension"],
                    }
                ),
            )
        )
        for offset, descriptor in enumerate(sampled["face_embeddings"], start=1):
            embedding_id = f"FEM-V2-{offset:03d}"
            captured_at = datetime.fromisoformat(descriptor["captured_at"]) if descriptor["captured_at"] else None
            frame_path = _storage_reference(Path(descriptor["frame_path"]))
            embedding = FaceEmbedding(
                id=embedding_id,
                case_id=DEMO_CASE_ID,
                evidence_id=DEMO_CCTV_EVIDENCE_ID,
                entity_id=arjun.id,
                frame_path=frame_path,
                timestamp_seconds=float(descriptor["timestamp_seconds"]),
                captured_at=captured_at,
                location=DEMO_LOCATION,
                bbox_json=json.dumps(descriptor["bbox"]),
                embedding_backend=descriptor["embedding_backend"],
            )
            session.add(embedding)
            index.upsert(
                embedding_id,
                descriptor["vector"],
                {
                    "case_id": DEMO_CASE_ID,
                    "evidence_id": DEMO_CCTV_EVIDENCE_ID,
                    "frame_path": frame_path,
                    "timestamp_seconds": descriptor["timestamp_seconds"],
                    "captured_at": descriptor["captured_at"],
                    "location": DEMO_LOCATION,
                    "face_detected": descriptor["face_detected"],
                    "embedding_backend": descriptor["embedding_backend"],
                    "synthetic_demo": True,
                },
            )
        return

    # The relational store is source of truth; rebuild index entries if its sidecar was removed.
    extractor = get_face_extractor()
    for embedding in existing:
        index_path = _storage_root() / Path(*Path(embedding.frame_path).parts[1:])
        image = cv2.imread(str(index_path))
        if image is None:
            continue
        descriptor = extractor.extract(image)[0]
        index.upsert(
            embedding.id,
            descriptor.vector,
            {
                "case_id": DEMO_CASE_ID,
                "evidence_id": DEMO_CCTV_EVIDENCE_ID,
                "frame_path": embedding.frame_path,
                "timestamp_seconds": embedding.timestamp_seconds,
                "captured_at": embedding.captured_at.isoformat() if embedding.captured_at else None,
                "location": embedding.location,
                "face_detected": descriptor.face_detected,
                "embedding_backend": descriptor.backend,
                "synthetic_demo": True,
            },
        )


def _ensure_demo_face_match(session: Any, arjun: Entity, assets: dict[str, Path]) -> None:
    """Run the same index search used by the endpoint to create the seeded indicator."""

    if session.get(FaceMatch, DEMO_FACE_MATCH_ID):
        return
    reference = get_face_extractor().extract_reference(assets["reference_path"].read_bytes())
    candidates = _face_index().search(reference.vector, limit=1, case_id=DEMO_CASE_ID)
    if not candidates:
        return
    embedding = session.get(FaceEmbedding, candidates[0].item_id)
    if embedding is None:
        return
    similarity = round(max(0.0, candidates[0].similarity), 4)
    session.add(
        FaceMatch(
            id=DEMO_FACE_MATCH_ID,
            case_id=DEMO_CASE_ID,
            embedding_id=embedding.id,
            entity_id=arjun.id,
            reference_label="Synthetic Arjun Rao reference portrait",
            similarity=similarity,
            label="Match Indicator (Requires Verification)",
            interpretation=f"Similarity Score {similarity * 100:.1f}% - Indicator Only, Not Proof. Synthetic demo data.",
        )
    )
    if embedding.captured_at and session.get(Event, DEMO_FACE_MATCH_EVENT_ID) is None:
        session.add(
            Event(
                id=DEMO_FACE_MATCH_EVENT_ID,
                case_id=DEMO_CASE_ID,
                time=embedding.captured_at,
                label="Face match indicator for Arjun Rao",
                location=embedding.location,
                source_evidence_id=embedding.evidence_id,
                entity_id=arjun.id,
                kind="face_match",
                metadata_json=json.dumps(
                    {
                        "face_match_id": DEMO_FACE_MATCH_ID,
                        "embedding_id": embedding.id,
                        "similarity": similarity,
                        "synthetic_demo": True,
                        "requires_human_verification": True,
                    }
                ),
            )
        )
