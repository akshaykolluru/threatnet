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
    Contradiction,
    Entity,
    Event,
    Evidence,
    FaceEmbedding,
    FaceMatch,
    Ranking,
    Relation,
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
RIYA_CASE_ID = "CASE-RIYA-001"
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

    # Keep the presentation case independent from the older CASE-101 fixtures.
    ensure_riya_demo_data()
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


def ensure_riya_demo_data() -> None:
    """Create the presentation-ready fictional case without touching user cases."""

    with SessionLocal() as session:
        if session.get(Case, RIYA_CASE_ID) is None:
            return
        _ensure_riya_assets()
        _insert_riya_records(session)
        if not session.scalar(select(AuditLog).where(AuditLog.action == "seeded_riya_presentation_demo", AuditLog.target == RIYA_CASE_ID)):
            session.add(AuditLog(actor="demo-seed", action="seeded_riya_presentation_demo", target=RIYA_CASE_ID))
        session.commit()


def _ensure_riya_assets() -> None:
    """Generate only fictional local presentation assets; no network or real data."""

    root = _storage_root() / "demo-assets"
    root.mkdir(parents=True, exist_ok=True)
    workbook_path = root / "case-riya-suspect-screening.xlsx"
    cctv_root = root / "riya-cctv"
    cctv_root.mkdir(parents=True, exist_ok=True)
    video_path = cctv_root / "riverside-cam-07.avi"
    frames: list[np.ndarray] = []
    for second in (0, 4, 8, 12):
        frame_path = cctv_root / f"riverside-cam-07-{second:02d}s.jpg"
        if not frame_path.exists():
            image = np.full((360, 640, 3), (49, 58, 67), dtype=np.uint8)
            cv2.rectangle(image, (0, 235), (640, 360), (31, 36, 42), thickness=-1)
            cv2.line(image, (0, 300), (640, 300), (115, 125, 133), thickness=3)
            x = 330 - second * 12
            cv2.rectangle(image, (x, 204), (x + 155, 270), (40, 45, 53), thickness=-1)
            cv2.rectangle(image, (x + 23, 180), (x + 118, 218), (52, 59, 67), thickness=-1)
            cv2.rectangle(image, (x + 38, 245), (x + 125, 263), (220, 220, 215), thickness=-1)
            cv2.putText(image, "TS 10 DEMO 8831", (x + 40, 259), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(image, "SYNTHETIC DEMO  |  RIVERSIDE CAM-07", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (225, 235, 240), 1, cv2.LINE_AA)
            cv2.putText(image, f"2026-08-17 21:{2 + second:02d}:00", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (225, 235, 240), 1, cv2.LINE_AA)
            cv2.imwrite(str(frame_path), image)
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is not None:
            frames.append(image)
    if frames and not video_path.exists():
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 1.0, (640, 360))
        if writer.isOpened():
            try:
                for image in frames:
                    writer.write(image)
            finally:
                writer.release()
    if not workbook_path.exists():
        try:
            from openpyxl import Workbook

            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Suspects"
            sheet.append(["Suspect ID", "Name", "Area", "Vehicle", "Route Match", "Call Link Count"])
            sheet.append(["SUS-RM-001", "Rohan Mehta", "Banjara Hills", "TS 09 DEMO 4172", "High", 8])
            sheet.append(["SUS-VS-002", "Vikram Sethi", "Madhapur", "TS 10 DEMO 8831", "High", 6])
            sheet.append(["SUS-AN-003", "Anil Kumar", "Jubilee Hills", "TS 08 DEMO 2904", "Medium", 1])
            sheet.append(["SUS-PK-004", "Priya Kapoor", "Gachibowli", "TS 11 DEMO 6310", "Low", 0])
            workbook.save(workbook_path)
        except Exception:
            # The rest of the case is still usable when the optional Excel writer
            # is absent; the UI exposes the expected workbook schema.
            pass


def _insert_riya_records(session: Any) -> None:
    """Insert a coherent, intentionally fictional chain of reviewable records."""

    entities = [
        ("ENT-RIYA", "Riya Sharma (fictional victim)", "riya sharma", "person", {"role": "victim", "estimated_age": "20", "body_type": "Average", "mobile_number": "90000-00002", "fictional_demo": True}),
        ("ENT-ROHAN", "Rohan Mehta (fictional suspect)", "rohan mehta", "person", {"role": "possible organizer", "estimated_age": "22–25", "body_type": "Medium build", "mobile_number": "90000-00001", "vehicle_number": "TS 09 DEMO 4172", "fictional_demo": True}),
        ("ENT-VIKRAM", "Vikram Sethi (fictional suspect)", "vikram sethi", "person", {"role": "possible direct perpetrator", "estimated_age": "28–32", "body_type": "Stocky build", "mobile_number": "90000-00003", "vehicle_number": "TS 10 DEMO 8831", "fictional_demo": True}),
        ("ENT-RIYA-CAR", "Riya's college commute — TS 12 DEMO 1648", "ts 12 demo 1648", "vehicle", {"owner": "Riya Sharma", "fictional_demo": True}),
        ("ENT-ROHAN-CAR", "Rohan's sedan — TS 09 DEMO 4172", "ts 09 demo 4172", "vehicle", {"owner": "Rohan Mehta", "fictional_demo": True}),
        ("ENT-VIKRAM-CAR", "Vikram's hatchback — TS 10 DEMO 8831", "ts 10 demo 8831", "vehicle", {"owner": "Vikram Sethi", "fictional_demo": True}),
        ("ENT-COLLEGE", "Sapphire College", "sapphire college", "location", {"fictional_demo": True}),
        ("ENT-HOME", "Riya's residence — Lakeview Apartments", "lakeview apartments", "location", {"fictional_demo": True}),
        ("ENT-PARTY", "Aurora Rooftop Cafe", "aurora rooftop cafe", "location", {"fictional_demo": True}),
        ("ENT-CRIME", "Riverside service road", "riverside service road", "location", {"fictional_demo": True}),
    ]
    for entity_id, label, canonical_name, entity_type, metadata in entities:
        existing_entity = session.get(Entity, entity_id)
        if existing_entity is None:
            session.add(Entity(id=entity_id, case_id=RIYA_CASE_ID, label=label, canonical_name=canonical_name, type=entity_type, metadata_json=json.dumps(metadata)))
        elif existing_entity.metadata_json != json.dumps(metadata):
            # Keep the fictional seed profile complete after a code update without
            # altering investigator-created entities.
            existing_entity.metadata_json = json.dumps(metadata)

    cctv_metadata = {
        "synthetic_demo": True,
        "location": "Riverside service road",
        "frames": [
            {
                "path": _storage_reference(_storage_root() / "demo-assets" / "riya-cctv" / f"riverside-cam-07-{second:02d}s.jpg"),
                "frame_index": second,
                "timestamp_seconds": float(second),
                "face_detected": False,
            }
            for second in (0, 4, 8, 12)
        ],
    }
    evidence = [
        ("RIYA-EV-001", "Case theory — fictional demonstration", "document", "Review hypothesis: rejection, repeated contact, route surveillance, contact with Vikram, and a payment trail form a source-linked lead chain. This is not a finding of guilt.", "demo/case-theory", 0.0),
        ("RIYA-EV-002", "Victim routine and college attendance", "statement", "Riya's fictional routine: Lakeview Apartments to Sapphire College on weekdays; weekend visits to Aurora Rooftop Cafe. Source record retained for review.", "statement/RIYA-W1", 0.74),
        ("RIYA-EV-003", "Rohan vehicle registry record", "vehicle", "Fictional registry: TS 09 DEMO 4172, silver sedan, registered to Rohan Mehta.", "registry/TS-09-DEMO-4172", 0.93),
        ("RIYA-EV-004", "Vikram vehicle registry record", "vehicle", "Fictional registry: TS 10 DEMO 8831, dark hatchback, registered to Vikram Sethi.", "registry/TS-10-DEMO-8831", 0.93),
        ("RIYA-EV-005", "Three-day route surveillance log", "location", "Fictional ANPR/CCTV correlation places Rohan's sedan near Riya's college-to-home route on three consecutive days. Requires source verification.", "cctv/route-surveillance-log", 0.84),
        ("RIYA-EV-006", "Rohan–Riya call-detail records", "call-log", "Fictional call log records repeated unanswered calls from Rohan to Riya following the rejected proposal. Review timing and attribution.", "cdr/ROHAN-RIYA-DEMO", 0.82),
        ("RIYA-EV-007", "Rohan–Vikram contact records", "call-log", "Fictional call log records repeated contacts between Rohan and Vikram before the incident. Content is not inferred from call metadata.", "cdr/ROHAN-VIKRAM-DEMO", 0.88),
        ("RIYA-EV-008", "Financial transfer review", "document", "Fictional bank review shows an INR 250,000 transfer from Rohan's fictional account to Vikram's fictional account shortly before the incident. Requires banking-record verification.", "bank/FT-DEMO-250000", 0.91),
        ("RIYA-EV-009", "Synthetic Riverside CAM-07 vehicle sighting", "cctv", "Fictional roadside CCTV frames record Vikram's fictional hatchback near Riverside service road during the incident window. Plate reading and vehicle attribution require review.", _storage_reference(_storage_root() / "demo-assets" / "riya-cctv" / "riverside-cam-07.avi"), 0.86),
        ("RIYA-EV-010", "Witness observation", "statement", "Fictional witness statement: a dark hatchback paused near Riverside service road shortly before the emergency call. Requires corroboration.", "statement/RIYA-W2", 0.67),
        ("RIYA-EV-011", "Suspect screening workbook", "document", "Synthetic workbook for the Suspect screening feature. Download or upload backend/storage/demo-assets/case-riya-suspect-screening.xlsx.", "storage/demo-assets/case-riya-suspect-screening.xlsx", 0.0),
    ]
    for evidence_id, title, evidence_type, notes, source, confidence in evidence:
        if session.get(Evidence, evidence_id) is None:
            metadata = {"fictional_demo": True, "requires_human_verification": True}
            if evidence_id == "RIYA-EV-009":
                metadata.update(cctv_metadata)
            session.add(Evidence(id=evidence_id, case_id=RIYA_CASE_ID, title=title, type=evidence_type, notes=notes, source=source, confidence=confidence, metadata_json=json.dumps(metadata)))

    event_specs = [
        ("RIYA-EVT-001", "2026-08-14T08:20:00", "Riya travels from home to Sapphire College", "Lakeview Apartments → Sapphire College", "ENT-RIYA", "presence", "RIYA-EV-002"),
        ("RIYA-EVT-002", "2026-08-14T17:35:00", "Rohan's sedan observed near Riya's return route", "Sapphire College exit", "ENT-ROHAN", "vehicle_sighting", "RIYA-EV-005"),
        ("RIYA-EVT-003", "2026-08-15T17:42:00", "Rohan's sedan observed near Riya's return route", "Lakeview junction", "ENT-ROHAN", "vehicle_sighting", "RIYA-EV-005"),
        ("RIYA-EVT-004", "2026-08-16T18:05:00", "Rohan's sedan observed near Riya's return route", "Sapphire College exit", "ENT-ROHAN", "vehicle_sighting", "RIYA-EV-005"),
        ("RIYA-EVT-005", "2026-08-16T21:18:00", "Rohan repeatedly calls Riya", "Cellular record — fictional", "ENT-ROHAN", "call_record", "RIYA-EV-006"),
        ("RIYA-EVT-006", "2026-08-16T22:10:00", "Rohan contacts Vikram", "Cellular record — fictional", "ENT-ROHAN", "call_record", "RIYA-EV-007"),
        ("RIYA-EVT-007", "2026-08-16T22:24:00", "Fictional INR 250,000 transfer to Vikram is recorded", "Financial review record", "ENT-ROHAN", "financial_record", "RIYA-EV-008"),
        ("RIYA-EVT-008", "2026-08-17T20:45:00", "Riya leaves Aurora Rooftop Cafe", "Aurora Rooftop Cafe", "ENT-RIYA", "presence", "RIYA-EV-002"),
        ("RIYA-EVT-009", "2026-08-17T21:02:00", "Vikram's hatchback recorded near Riverside service road", "Riverside service road", "ENT-VIKRAM", "vehicle_sighting", "RIYA-EV-009"),
        ("RIYA-EVT-010", "2026-08-17T21:16:00", "Emergency call creates crime-scene review window", "Riverside service road", "ENT-RIYA", "incident", "RIYA-EV-010"),
    ]
    for event_id, timestamp, label, location, entity_id, kind, evidence_id in event_specs:
        if session.get(Event, event_id) is None:
            session.add(Event(id=event_id, case_id=RIYA_CASE_ID, time=datetime.fromisoformat(timestamp), label=label, location=location, entity_id=entity_id, kind=kind, source_evidence_id=evidence_id, metadata_json=json.dumps({"fictional_demo": True, "requires_human_verification": True})))

    relation_specs = [
        ("ENT-RIYA", "ENT-COLLEGE", "ROUTINE_DESTINATION"), ("ENT-RIYA", "ENT-HOME", "RESIDES_AT"), ("ENT-RIYA", "ENT-PARTY", "VISITS"),
        ("ENT-ROHAN", "ENT-ROHAN-CAR", "REGISTERED_TO"), ("ENT-VIKRAM", "ENT-VIKRAM-CAR", "REGISTERED_TO"),
        ("ENT-ROHAN", "ENT-RIYA", "REPEATED_CALLS_REVIEW"), ("ENT-ROHAN", "ENT-RIYA", "ROUTE_SURVEILLANCE_LEAD"),
        ("ENT-ROHAN", "ENT-VIKRAM", "CONTACT_RECORDS_REVIEW"), ("ENT-ROHAN", "ENT-VIKRAM", "PAYMENT_RECORD_REVIEW"),
        ("ENT-VIKRAM", "ENT-CRIME", "VEHICLE_SIGHTING_REVIEW"), ("ENT-RIYA", "ENT-CRIME", "INCIDENT_LOCATION"),
    ]
    existing_relations = {(row.source, row.target, row.label) for row in session.scalars(select(Relation).where(Relation.case_id == RIYA_CASE_ID)).all()}
    for source, target, label in relation_specs:
        if (source, target, label) not in existing_relations:
            session.add(Relation(case_id=RIYA_CASE_ID, source=source, target=target, label=label, metadata_json=json.dumps({"fictional_demo": True, "requires_human_verification": True})))

    rankings = [
        ("RIYA-RANK-001", "Financial transfer review", 0.91, "Fictional payment record connects Rohan and Vikram; banking source verification remains required."),
        ("RIYA-RANK-002", "Rohan–Vikram call pattern", 0.88, "Repeated fictional contact before the incident is a review lead, not evidence of content or intent."),
        ("RIYA-RANK-003", "Vikram vehicle near crime scene", 0.86, "Fictional CCTV/ANPR sighting aligns with the incident window; plate attribution requires review."),
        ("RIYA-RANK-004", "Three-day route surveillance", 0.84, "Fictional vehicle sightings align with Riya's stated routine and require camera-source verification."),
    ]
    for ranking_id, label, score, reason in rankings:
        if session.get(Ranking, ranking_id) is None:
            session.add(Ranking(id=ranking_id, case_id=RIYA_CASE_ID, label=label, score=score, reason=reason))

    if session.get(Contradiction, "RIYA-ALERT-001") is None:
        session.add(Contradiction(id="RIYA-ALERT-001", case_id=RIYA_CASE_ID, severity="high", summary="Rohan's fictional interview account places him at home while his registered sedan is recorded on Riya's route during the same review window.", entity_id="ENT-ROHAN", source_event_id="RIYA-EVT-003", reasoning_trace="Synthetic demo alert: compare the stated account with the source-linked vehicle sighting. Human verification is required."))


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
