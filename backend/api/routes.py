from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

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
    json_metadata,
)
from .extraction import ExtractionSourceType, persist_extraction
from .frame_search import frame_index, get_frame_search_embedder, index_frames, search_frames
from .ingestion import get_face_extractor, inspect_video, public_ingestion_metadata
from .screening import screen_workbook
from .services import build_graph_payload, demo_state
from .vector_search import FaceVectorIndex


router = APIRouter(prefix="/api")
MATCH_INDICATOR_THRESHOLD = 0.55


def _storage_root() -> Path:
    return Path(os.getenv("STORAGE_DIR", str(Path(__file__).resolve().parent.parent / "storage")))


def _face_index() -> FaceVectorIndex:
    return FaceVectorIndex(_storage_root() / "face-index")


def _frame_search_index() -> FaceVectorIndex:
    return frame_index(_storage_root())


def _storage_reference(path: Path) -> str:
    """Make a stored file addressable through the existing /storage static mount."""

    try:
        relative_path = path.resolve().relative_to(_storage_root().resolve())
        return (Path("storage") / relative_path).as_posix()
    except ValueError:
        return str(path)


def _remove_storage_file(source: str) -> None:
    source_path = Path(source.replace("\\", "/"))
    if source_path.parts[:1] != ("storage",):
        return
    stored_file = _storage_root() / Path(*source_path.parts[1:])
    if stored_file.is_file():
        stored_file.unlink()


def _parse_timestamp(value: str | None, field_name: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"{field_name} must be a valid ISO timestamp") from exc


def _case_or_404(session: Session, case_id: str) -> Case:
    case = session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


def _entity_or_404(session: Session, case_id: str, entity_id: str) -> Entity:
    entity = session.get(Entity, entity_id)
    if not entity or entity.case_id != case_id:
        raise HTTPException(status_code=422, detail="Entity does not belong to this case")
    return entity


def _match_identifier(case_id: str, embedding_id: str, entity_id: str, reference_label: str) -> str:
    digest = hashlib.sha256(
        f"{case_id}:{embedding_id}:{entity_id}:{reference_label.casefold()}".encode("utf-8")
    ).hexdigest()[:16].upper()
    return f"FM-{digest}"


def _face_match_event_identifier(face_match_id: str) -> str:
    return f"EVT-FM-{face_match_id.removeprefix('FM-')}"


class CaseCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    summary: str = Field(default="New investigation case", max_length=10_000)
    priority: Literal["Low", "Medium", "High"] = "Medium"


class CaseStatusUpdate(BaseModel):
    status: Literal["Open", "Closed"]


class EvidenceCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    type: str = Field(default="source", max_length=64)
    notes: str = Field(default="", max_length=50_000)
    source: str = Field(default="investigator entry", max_length=10_000)


class EventCreate(BaseModel):
    label: str = Field(min_length=1, max_length=255)
    location: str = Field(default="", max_length=255)
    time: Optional[str] = None
    entity_id: Optional[str] = None
    kind: Literal["observation", "presence"] = "observation"
    coordinates: Optional[tuple[float, float]] = None


class IntelligenceExtractionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)
    source: str = Field(default="investigator text entry", max_length=10_000)
    source_type: ExtractionSourceType = "statement"
    default_time: Optional[str] = None
    default_location: str = Field(default="", max_length=255)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "demo", "intelligence_engine": "v2"}


@router.get("/cases")
def cases() -> dict[str, list[dict[str, str]]]:
    demo_state()
    with SessionLocal() as session:
        return {
            "items": [
                {
                    "id": case.id,
                    "title": case.title,
                    "status": case.status,
                    "priority": case.priority,
                    "summary": case.summary,
                }
                for case in session.scalars(select(Case).order_by(Case.id)).all()
            ]
        }


@router.post("/cases")
def create_case(payload: CaseCreate) -> dict[str, str]:
    demo_state()
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Case title is required")
    with SessionLocal() as session:
        existing = session.scalars(select(Case.id)).all()
        next_number = max(
            [
                int(case_id.split("-")[1])
                for case_id in existing
                if case_id.startswith("CASE-") and case_id.split("-")[1].isdigit()
            ]
            or [100]
        ) + 1
        case = Case(
            id=f"CASE-{next_number}",
            title=title,
            status="Open",
            priority=payload.priority,
            summary=payload.summary.strip() or "New investigation case",
        )
        session.add(case)
        session.add(AuditLog(actor="demo-investigator", action="created_case", target=case.id))
        session.commit()
        return {
            "id": case.id,
            "title": case.title,
            "status": case.status,
            "priority": case.priority,
            "summary": case.summary,
        }


@router.patch("/case/{case_id}")
def update_case_status(case_id: str, payload: CaseStatusUpdate) -> dict[str, str]:
    demo_state()
    with SessionLocal() as session:
        case = _case_or_404(session, case_id)
        case.status = payload.status
        session.add(AuditLog(actor="demo-investigator", action=f"status_changed_to_{payload.status.lower()}", target=case_id))
        session.commit()
        return {
            "id": case.id,
            "title": case.title,
            "status": case.status,
            "priority": case.priority,
            "summary": case.summary,
        }


@router.delete("/cases/{case_id}")
def delete_case(case_id: str) -> dict[str, str]:
    demo_state()
    if case_id in {"CASE-101", "CASE-202", "CASE-303"}:
        raise HTTPException(status_code=409, detail="Built-in demo cases cannot be deleted; close the case instead")
    with SessionLocal() as session:
        case = _case_or_404(session, case_id)
        evidence_sources = [item.source for item in session.scalars(select(Evidence).where(Evidence.case_id == case_id)).all()]
        embedding_ids = [item.id for item in session.scalars(select(FaceEmbedding).where(FaceEmbedding.case_id == case_id)).all()]
        for model in (FaceMatch, FaceEmbedding, Contradiction, Relation, Event, Entity, Ranking, Evidence):
            for item in session.scalars(select(model).where(model.case_id == case_id)).all():
                session.delete(item)
        session.add(AuditLog(actor="demo-investigator", action="deleted_case", target=case_id))
        session.delete(case)
        session.commit()
    _face_index().remove(embedding_ids)
    _frame_search_index().remove_by_metadata(case_id=case_id)
    for source in evidence_sources:
        _remove_storage_file(source)
    return {"deleted": case_id}


@router.post("/case/{case_id}/evidence")
def create_evidence(case_id: str, payload: EvidenceCreate) -> dict[str, str]:
    demo_state()
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Evidence title is required")
    with SessionLocal() as session:
        _case_or_404(session, case_id)
        item = Evidence(
            id=f"EV-{uuid4().hex[:8].upper()}",
            case_id=case_id,
            title=title,
            type=payload.type.strip() or "source",
            notes=payload.notes.strip(),
            source=payload.source.strip() or "investigator entry",
        )
        session.add(item)
        session.add(AuditLog(actor="demo-investigator", action="added_evidence", target=case_id))
        session.commit()
        return {
            "id": item.id,
            "case_id": case_id,
            "title": item.title,
            "type": item.type,
            "notes": item.notes,
            "source": item.source,
        }


@router.post("/case/{case_id}/evidence/image")
async def create_image_evidence(
    case_id: str,
    title: str = Form(...),
    type: str = Form("image"),
    notes: str = Form(""),
    source: str = Form("image upload"),
    image: UploadFile = File(...),
) -> dict[str, Any]:
    demo_state()
    clean_title = title.strip()
    if not clean_title:
        raise HTTPException(status_code=422, detail="Evidence title is required")
    if not image.filename or not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a supported image file")
    extension = Path(image.filename).suffix.lower()
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    if extension not in allowed_extensions:
        raise HTTPException(status_code=415, detail="Use JPG, PNG, WEBP, or GIF images")
    content = await image.read()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Evidence images are limited to 15 MB")
    with SessionLocal() as session:
        _case_or_404(session, case_id)
        evidence_id = f"IMG-{uuid4().hex[:8].upper()}"
        relative_source = Path("storage") / "cases" / case_id / "evidence" / f"{evidence_id}{extension}"
        stored_file = _storage_root() / relative_source.relative_to("storage")
        stored_file.parent.mkdir(parents=True, exist_ok=True)
        stored_file.write_bytes(content)
        metadata = {
            "kind": "image",
            "filename": image.filename,
            "content_type": image.content_type,
            "size_bytes": len(content),
        }
        item = Evidence(
            id=evidence_id,
            case_id=case_id,
            title=clean_title,
            type=type.strip() or "image",
            notes=notes.strip(),
            source=str(relative_source).replace("\\", "/"),
            metadata_json=json.dumps(metadata),
        )
        session.add(item)
        session.add(AuditLog(actor="demo-investigator", action="added_image_evidence", target=case_id))
        session.commit()
        return {
            "id": item.id,
            "case_id": case_id,
            "title": item.title,
            "type": item.type,
            "notes": item.notes,
            "source": item.source,
            "metadata": metadata,
        }


def _delete_event_dependencies(session: Session, event: Event) -> None:
    for relation in session.scalars(
        select(Relation).where(
            Relation.case_id == event.case_id,
            or_(Relation.source == event.id, Relation.target == event.id, Relation.event_id == event.id),
        )
    ).all():
        session.delete(relation)
    for contradiction in session.scalars(
        select(Contradiction).where(
            Contradiction.case_id == event.case_id,
            or_(
                Contradiction.source_event_id == event.id,
                Contradiction.conflicting_event_id == event.id,
            ),
        )
    ).all():
        session.delete(contradiction)


@router.delete("/case/{case_id}/evidence/{evidence_id}")
def delete_evidence(case_id: str, evidence_id: str) -> dict[str, str]:
    demo_state()
    with SessionLocal() as session:
        _case_or_404(session, case_id)
        item = session.get(Evidence, evidence_id)
        if not item or item.case_id != case_id:
            raise HTTPException(status_code=404, detail="Evidence record not found")
        source = item.source
        embeddings = list(session.scalars(select(FaceEmbedding).where(FaceEmbedding.evidence_id == evidence_id)).all())
        embedding_ids = [embedding.id for embedding in embeddings]
        if embedding_ids:
            for face_match in session.scalars(select(FaceMatch).where(FaceMatch.embedding_id.in_(embedding_ids))).all():
                session.delete(face_match)
            for embedding in embeddings:
                session.delete(embedding)
        for event in session.scalars(select(Event).where(Event.source_evidence_id == evidence_id)).all():
            if event.kind in {"presence", "extracted_relation", "face_match"}:
                _delete_event_dependencies(session, event)
                session.delete(event)
            else:
                event.source_evidence_id = None
        for relation in session.scalars(
            select(Relation).where(
                Relation.case_id == case_id,
                or_(Relation.source == evidence_id, Relation.target == evidence_id),
            )
        ).all():
            session.delete(relation)
        session.delete(item)
        session.add(AuditLog(actor="demo-investigator", action=f"deleted_evidence:{evidence_id}", target=case_id))
        session.commit()
    _face_index().remove(embedding_ids)
    _frame_search_index().remove_by_metadata(evidence_id=evidence_id)
    _remove_storage_file(source)
    return {"deleted": evidence_id, "case_id": case_id}


@router.post("/case/{case_id}/events")
def create_event(case_id: str, payload: EventCreate) -> dict[str, Any]:
    demo_state()
    label = payload.label.strip()
    if not label:
        raise HTTPException(status_code=422, detail="Event label is required")
    event_time = _parse_timestamp(payload.time, "Event time") or datetime.utcnow()
    with SessionLocal() as session:
        _case_or_404(session, case_id)
        if payload.entity_id:
            _entity_or_404(session, case_id, payload.entity_id)
        metadata: dict[str, Any] = {}
        if payload.coordinates:
            metadata["coordinates"] = list(payload.coordinates)
        item = Event(
            id=f"EVT-{uuid4().hex[:8].upper()}",
            case_id=case_id,
            time=event_time,
            label=label,
            location=payload.location.strip(),
            entity_id=payload.entity_id,
            kind=payload.kind,
            metadata_json=json.dumps(metadata),
        )
        session.add(item)
        session.flush()
        findings = detect_spatiotemporal_contradictions(session, case_id, payload.entity_id)
        session.add(AuditLog(actor="demo-investigator", action="added_event", target=case_id))
        session.commit()
        return {
            "id": item.id,
            "case_id": case_id,
            "time": item.time.isoformat(),
            "label": item.label,
            "location": item.location,
            "entity_id": item.entity_id,
            "kind": item.kind,
            "contradictions_created": [finding.contradiction_id for finding in findings],
        }


@router.delete("/case/{case_id}/events/{event_id}")
def delete_event(case_id: str, event_id: str) -> dict[str, str]:
    demo_state()
    with SessionLocal() as session:
        _case_or_404(session, case_id)
        item = session.get(Event, event_id)
        if not item or item.case_id != case_id:
            raise HTTPException(status_code=404, detail="Timeline event not found")
        _delete_event_dependencies(session, item)
        session.delete(item)
        session.add(AuditLog(actor="demo-investigator", action=f"deleted_event:{event_id}", target=case_id))
        session.commit()
        return {"deleted": event_id, "case_id": case_id}


@router.post("/case/{case_id}/intelligence/extract")
def extract_intelligence(case_id: str, payload: IntelligenceExtractionRequest) -> dict[str, Any]:
    """Turn a source narrative into human-reviewable graph triples and timeline events."""

    demo_state()
    default_time = _parse_timestamp(payload.default_time, "default_time")
    with SessionLocal() as session:
        _case_or_404(session, case_id)
        try:
            result = persist_extraction(
                session=session,
                case_id=case_id,
                raw_text=payload.text,
                source=payload.source,
                source_type=payload.source_type,
                default_time=default_time,
                default_location=payload.default_location.strip(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        session.flush()
        findings = detect_spatiotemporal_contradictions(session, case_id)
        session.add(AuditLog(actor="demo-investigator", action="extracted_intelligence", target=case_id))
        session.commit()
        return {
            "case_id": case_id,
            "evidence_id": result.evidence_id,
            "triples": [
                {
                    "subject": triple.subject,
                    "relation": triple.relation,
                    "object": triple.object,
                    "timestamp": triple.timestamp.isoformat() if triple.timestamp else None,
                    "location": triple.location,
                    "confidence": triple.confidence,
                    "source_text": triple.source_text,
                    "interpretation": "Extracted relationship requires investigator verification.",
                }
                for triple in result.triples
            ],
            "entity_ids": result.entity_ids,
            "event_ids": result.event_ids,
            "contradictions_created": [finding.contradiction_id for finding in findings],
        }


def _rewrite_ingestion_paths(result: dict[str, Any]) -> None:
    for frame in result.get("frames", []):
        frame["path"] = _storage_reference(Path(frame["path"]))
    for embedding in result.get("face_embeddings", []):
        embedding["frame_path"] = _storage_reference(Path(embedding["frame_path"]))


@router.post("/case/{case_id}/cctv/inspect")
async def inspect_cctv(
    case_id: str,
    file: UploadFile = File(...),
    recorded_at: str | None = Form(None),
    location: str = Form(""),
    sample_every_seconds: float = Form(2.0),
    query: str = Form(""),
    reference_image: UploadFile | None = File(None),
) -> dict[str, Any]:
    """Sample CCTV frames, create 512-d descriptors, and index them with provenance."""

    demo_state()
    captured_at = _parse_timestamp(recorded_at, "recorded_at")
    with SessionLocal() as session:
        _case_or_404(session, case_id)
    if not file.filename:
        raise HTTPException(status_code=400, detail="A video filename is required")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="The uploaded video is empty")
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Demo uploads are limited to 100 MB")
    reference_content = await _validated_reference_image(reference_image)
    if (query.strip() or reference_content) and not get_frame_search_embedder().available:
        raise HTTPException(
            status_code=503,
            detail="Semantic frame search is unavailable. Install the optional open_clip_torch dependency and ensure its pretrained model can be downloaded once.",
        )

    job_dir = _storage_root() / "cctv" / uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    source = job_dir / Path(file.filename).name
    source.write_bytes(content)
    try:
        ingestion_result = inspect_video(
            source,
            job_dir / "frames",
            sample_every_seconds=sample_every_seconds,
            recorded_at=captured_at,
            location=location.strip(),
        )
    except ValueError as exc:
        source.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Keep local paths only while the embedding worker reads the generated files.
    # API/database output below is rewritten to the existing /storage public mount.
    evidence_id = f"CCTV-{uuid4().hex[:8].upper()}"
    try:
        indexed_frame_embeddings = index_frames(_frame_search_index(), ingestion_result["frames"], case_id, evidence_id)
    except (RuntimeError, ValueError) as exc:
        # Ingestion is useful without the optional semantic model; a submitted
        # query must fail clearly instead of silently ignoring its inputs.
        if query.strip() or reference_content:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        indexed_frame_embeddings = 0
    _rewrite_ingestion_paths(ingestion_result)
    metadata = public_ingestion_metadata(ingestion_result)
    metadata.update(
        {
            "recorded_at": captured_at.isoformat() if captured_at else None,
            "location": location.strip(),
            "embedding_index": _face_index().backend,
        }
    )

    with SessionLocal() as session:
        _case_or_404(session, case_id)
        session.add(
            Evidence(
                id=evidence_id,
                case_id=case_id,
                title=f"CCTV inspection: {file.filename}",
                type="cctv",
                notes="OpenCV-sampled frames and similarity candidates require investigator review.",
                source=_storage_reference(source),
                metadata_json=json.dumps(metadata),
            )
        )
        vector_index = _face_index()
        for descriptor in ingestion_result.get("face_embeddings", []):
            embedding_id = f"FEM-{uuid4().hex[:10].upper()}"
            embedding_captured_at = _parse_timestamp(descriptor.get("captured_at"), "frame timestamp")
            face_embedding = FaceEmbedding(
                id=embedding_id,
                case_id=case_id,
                evidence_id=evidence_id,
                frame_path=descriptor["frame_path"],
                timestamp_seconds=float(descriptor["timestamp_seconds"]),
                captured_at=embedding_captured_at,
                location=descriptor.get("location", ""),
                bbox_json=json.dumps(descriptor.get("bbox", [])),
                embedding_backend=descriptor["embedding_backend"],
            )
            session.add(face_embedding)
            vector_index.upsert(
                embedding_id,
                descriptor["vector"],
                {
                    "case_id": case_id,
                    "evidence_id": evidence_id,
                    "frame_path": descriptor["frame_path"],
                    "timestamp_seconds": descriptor["timestamp_seconds"],
                    "captured_at": descriptor.get("captured_at"),
                    "location": descriptor.get("location", ""),
                    "face_detected": descriptor.get("face_detected", False),
                    "embedding_backend": descriptor["embedding_backend"],
                },
            )
        session.flush()
        findings = detect_spatiotemporal_contradictions(session, case_id)
        session.add(AuditLog(actor="demo-investigator", action="inspected_cctv", target=case_id))
        session.commit()
    initial_results: list[dict[str, Any]] = []
    if query.strip() or reference_content:
        try:
            initial_results = _public_search_results(
                search_frames(_frame_search_index(), case_id, query, reference_content, evidence_id=evidence_id)
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "evidence_id": evidence_id,
        "case_id": case_id,
        "source": _storage_reference(source),
        "metadata": metadata,
        "indexed_embeddings": len(ingestion_result.get("face_embeddings", [])),
        "indexed_frame_embeddings": indexed_frame_embeddings,
        "query": query.strip(),
        "results": initial_results,
        "contradictions_created": [finding.contradiction_id for finding in findings],
        "interpretation": "Derived frames and similarity candidates are investigative indicators, not proof.",
    }


async def _validated_reference_image(reference_image: UploadFile | None) -> bytes | None:
    if reference_image is None:
        return None
    if not reference_image.filename:
        raise HTTPException(status_code=422, detail="The reference image has no filename")
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    extension = Path(reference_image.filename).suffix.lower()
    if extension not in allowed_extensions or not (reference_image.content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="Reference image must be JPG, PNG, or WEBP")
    content = await reference_image.read()
    if not content:
        raise HTTPException(status_code=422, detail="The reference image is empty")
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Reference images are limited to 15 MB")
    return content


def _public_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for result in results:
        item = dict(result)
        item["frame_path"] = _storage_reference(Path(item["frame_path"]))
        public.append(item)
    return public


@router.post("/case/{case_id}/cctv/search")
async def search_cctv_frames(
    case_id: str,
    query: str = Form(""),
    reference_image: UploadFile | None = File(None),
    evidence_id: str | None = Form(None),
    limit: int = Form(12),
) -> dict[str, Any]:
    """Search already-indexed video frames without re-uploading or reprocessing a video."""

    demo_state()
    reference_content = await _validated_reference_image(reference_image)
    if not query.strip() and not reference_content:
        raise HTTPException(status_code=422, detail="Enter a search prompt or upload a reference image")
    with SessionLocal() as session:
        _case_or_404(session, case_id)
        if evidence_id:
            evidence = session.get(Evidence, evidence_id)
            if not evidence or evidence.case_id != case_id or evidence.type != "cctv":
                raise HTTPException(status_code=422, detail="Select a CCTV clip belonging to this case")
    try:
        results = search_frames(
            _frame_search_index(), case_id, query=query, reference_image=reference_content,
            evidence_id=evidence_id, limit=max(1, min(limit, 25)),
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "case_id": case_id,
        "evidence_id": evidence_id,
        "query": query.strip(),
        "results": _public_search_results(results),
        "interpretation": "Results are ranked visual-similarity candidates for investigator review, not conclusions or identity determinations.",
    }


def _resolve_reference_entity(
    session: Session,
    case_id: str,
    entity_id: str | None,
    reference_label: str,
) -> Entity:
    if entity_id:
        return _entity_or_404(session, case_id, entity_id)
    clean_label = reference_label.strip() or "Unlabeled reference photo"
    canonical_name = clean_label.casefold()
    existing = session.scalar(
        select(Entity).where(
            Entity.case_id == case_id,
            func.lower(Entity.canonical_name) == canonical_name,
        )
    )
    if existing:
        return existing
    entity = Entity(
        id=f"ENT-{uuid4().hex[:8].upper()}",
        case_id=case_id,
        label=clean_label,
        canonical_name=canonical_name,
        type="person",
        metadata_json=json.dumps({"created_by": "reference_photo_match"}),
    )
    session.add(entity)
    session.flush()
    return entity


@router.post("/cctv/match-face")
async def match_face(
    case_id: str = Form(...),
    reference_photo: UploadFile = File(...),
    reference_label: str = Form(""),
    entity_id: str | None = Form(None),
    limit: int = Form(5),
) -> dict[str, Any]:
    """Compare a reference photo with stored frame vectors and persist review alerts."""

    demo_state()
    if not reference_photo.filename:
        raise HTTPException(status_code=400, detail="A reference photo filename is required")
    extension = Path(reference_photo.filename).suffix.lower()
    if not (reference_photo.content_type or "").startswith("image/") and extension not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=415, detail="Upload a supported reference photo")
    content = await reference_photo.read()
    if not content:
        raise HTTPException(status_code=422, detail="The reference photo is empty")
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Reference photos are limited to 15 MB")
    try:
        reference_descriptor = get_face_extractor().extract_reference(content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    safe_limit = min(max(limit, 1), 25)
    candidates = _face_index().search(reference_descriptor.vector, limit=safe_limit, case_id=case_id)
    if not candidates:
        return {
            "case_id": case_id,
            "matches": [],
            "interpretation": "No stored frame vectors are available for this case. This endpoint never treats a missing or low score as proof of exclusion.",
        }

    label = reference_label.strip() or Path(reference_photo.filename).stem
    response_matches: list[dict[str, Any]] = []
    with SessionLocal() as session:
        _case_or_404(session, case_id)
        reference_entity = _resolve_reference_entity(session, case_id, entity_id, label)
        for candidate in candidates:
            embedding = session.get(FaceEmbedding, candidate.item_id)
            if embedding is None or embedding.case_id != case_id:
                continue
            confidence = round(max(0.0, candidate.similarity), 4)
            confidence_percent = round(confidence * 100, 1)
            stored_as_alert = confidence >= MATCH_INDICATOR_THRESHOLD
            match_id: str | None = None
            if stored_as_alert:
                match_id = _match_identifier(case_id, embedding.id, reference_entity.id, label)
                stored_match = session.get(FaceMatch, match_id)
                if stored_match is None:
                    stored_match = FaceMatch(
                        id=match_id,
                        case_id=case_id,
                        embedding_id=embedding.id,
                        entity_id=reference_entity.id,
                        reference_label=label,
                        similarity=confidence,
                        label="Match Indicator (Requires Verification)",
                        interpretation=f"Similarity Score {confidence_percent:.1f}% - Indicator Only, Not Proof.",
                    )
                    session.add(stored_match)
                else:
                    stored_match.similarity = confidence

                if embedding.captured_at and embedding.location:
                    event_id = _face_match_event_identifier(match_id)
                    if session.get(Event, event_id) is None:
                        session.add(
                            Event(
                                id=event_id,
                                case_id=case_id,
                                time=embedding.captured_at,
                                label=f"Face match indicator for {reference_entity.label}",
                                location=embedding.location,
                                source_evidence_id=embedding.evidence_id,
                                entity_id=reference_entity.id,
                                kind="face_match",
                                metadata_json=json.dumps(
                                    {
                                        "face_match_id": match_id,
                                        "embedding_id": embedding.id,
                                        "similarity": confidence,
                                        "timestamp_seconds": embedding.timestamp_seconds,
                                        "requires_human_verification": True,
                                    }
                                ),
                            )
                        )
            response_matches.append(
                {
                    "match_id": match_id,
                    "embedding_id": embedding.id,
                    "frame_path": embedding.frame_path,
                    "timestamp_seconds": embedding.timestamp_seconds,
                    "captured_at": embedding.captured_at.isoformat() if embedding.captured_at else None,
                    "location": embedding.location,
                    "confidence_score": confidence,
                    "similarity_percent": confidence_percent,
                    "label": "Match Indicator (Requires Verification)",
                    "interpretation": f"Similarity Score {confidence_percent:.1f}% - Indicator Only, Not Proof.",
                    "embedding_backend": embedding.embedding_backend,
                    "face_detected": candidate.metadata.get("face_detected", False),
                    "stored_as_alert": stored_as_alert,
                }
            )
        session.flush()
        findings = detect_spatiotemporal_contradictions(session, case_id, reference_entity.id)
        session.add(AuditLog(actor="demo-investigator", action="matched_reference_photo", target=case_id))
        session.commit()
    return {
        "case_id": case_id,
        "reference_entity_id": reference_entity.id,
        "reference_label": reference_entity.label,
        "matches": response_matches,
        "contradictions_created": [finding.contradiction_id for finding in findings],
        "interpretation": "Similarity candidates are ranked for investigator review. They are indicators only and never proof of identity.",
    }


@router.post("/case/{case_id}/screening/upload")
async def upload_screening_sheet(case_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    demo_state()
    with SessionLocal() as session:
        _case_or_404(session, case_id)
    if not file.filename or Path(file.filename).suffix.lower() not in {".xlsx", ".xlsm"}:
        raise HTTPException(status_code=415, detail="Upload an .xlsx or .xlsm workbook")
    storage_root = _storage_root() / "cases" / case_id / "screening"
    storage_root.mkdir(parents=True, exist_ok=True)
    source = storage_root / Path(file.filename).name
    source.write_bytes(await file.read())
    try:
        result = screen_workbook(source)
    except (KeyError, ValueError, TypeError, RuntimeError) as exc:
        source.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Workbook could not be screened: {exc}") from exc
    with SessionLocal() as session:
        session.add(AuditLog(actor="demo-investigator", action="screened_spreadsheet", target=case_id))
        session.commit()
    return {"case_id": case_id, "source": _storage_reference(source), **result}


def _serialize_evidence(item: Evidence) -> dict[str, Any]:
    return {
        "id": item.id,
        "case_id": item.case_id,
        "title": item.title,
        "type": item.type,
        "notes": item.notes,
        "confidence": item.confidence,
        "source": item.source,
        "metadata": json_metadata(item.metadata_json),
    }


def _serialize_event(item: Event) -> dict[str, Any]:
    return {
        "id": item.id,
        "case_id": item.case_id,
        "time": item.time.isoformat(),
        "label": item.label,
        "location": item.location,
        "entity_id": item.entity_id,
        "kind": item.kind,
        "metadata": json_metadata(item.metadata_json),
    }


def _serialize_contradiction(item: Contradiction) -> dict[str, Any]:
    return {
        "id": item.id,
        "severity": item.severity,
        "summary": item.summary,
        "entity_id": item.entity_id,
        "source_event_id": item.source_event_id,
        "conflicting_event_id": item.conflicting_event_id,
        "reasoning_trace": item.reasoning_trace,
        "created_at": item.created_at.isoformat() if item.created_at else None,
    }


def _serialize_face_match(item: FaceMatch, embedding: FaceEmbedding | None) -> dict[str, Any]:
    return {
        "id": item.id,
        "embedding_id": item.embedding_id,
        "entity_id": item.entity_id,
        "reference_label": item.reference_label,
        "similarity": item.similarity,
        "similarity_percent": round(max(0.0, item.similarity) * 100, 1),
        "label": item.label,
        "interpretation": item.interpretation,
        "created_at": item.created_at.isoformat(),
        "frame_path": embedding.frame_path if embedding else "",
        "timestamp_seconds": embedding.timestamp_seconds if embedding else None,
        "captured_at": embedding.captured_at.isoformat() if embedding and embedding.captured_at else None,
        "location": embedding.location if embedding else "",
    }


def _case_alerts(contradictions: list[Contradiction], matches: list[FaceMatch], embeddings: dict[str, FaceEmbedding]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = [
        {
            "id": contradiction.id,
            "type": "contradiction",
            "severity": contradiction.severity,
            "title": "Spatiotemporal review alert",
            "summary": contradiction.summary,
            "reasoning_trace": contradiction.reasoning_trace,
            "created_at": contradiction.created_at.isoformat() if contradiction.created_at else None,
        }
        for contradiction in contradictions
    ]
    alerts.extend(
        {
            "id": face_match.id,
            "type": "face_match",
            "severity": "review",
            "title": face_match.label,
            "summary": face_match.interpretation,
            "reasoning_trace": (
                f"Reference '{face_match.reference_label}' scored "
                f"{max(0.0, face_match.similarity) * 100:.1f}% against frame "
                f"{embeddings.get(face_match.embedding_id).frame_path if embeddings.get(face_match.embedding_id) else 'unavailable'}."
            ),
            "created_at": face_match.created_at.isoformat(),
        }
        for face_match in matches
    )
    return sorted(alerts, key=lambda alert: alert.get("created_at") or "", reverse=True)


@router.get("/case/{case_id}")
def case_detail(case_id: str) -> dict[str, Any]:
    state = demo_state()
    with SessionLocal() as session:
        case = session.get(Case, case_id)
        if case:
            evidence = list(session.scalars(select(Evidence).where(Evidence.case_id == case_id)).all())
            entities = list(session.scalars(select(Entity).where(Entity.case_id == case_id).order_by(Entity.label)).all())
            events = list(session.scalars(select(Event).where(Event.case_id == case_id).order_by(Event.time)).all())
            ranking = list(session.scalars(select(Ranking).where(Ranking.case_id == case_id).order_by(Ranking.score.desc())).all())
            contradictions = list(
                session.scalars(select(Contradiction).where(Contradiction.case_id == case_id).order_by(Contradiction.created_at.desc())).all()
            )
            relations = list(session.scalars(select(Relation).where(Relation.case_id == case_id)).all())
            face_embeddings = list(session.scalars(select(FaceEmbedding).where(FaceEmbedding.case_id == case_id)).all())
            embeddings_by_id = {embedding.id: embedding for embedding in face_embeddings}
            face_matches = list(
                session.scalars(select(FaceMatch).where(FaceMatch.case_id == case_id).order_by(FaceMatch.created_at.desc())).all()
            )
            audit = list(
                session.scalars(select(AuditLog).where(AuditLog.target == case_id).order_by(AuditLog.at)).all()
            )
            graph_nodes = [
                {"id": f"CASE:{case.id}", "label": case.title, "group": "case"},
                *[{"id": item.id, "label": item.title, "group": item.type} for item in evidence],
                *[{"id": item.id, "label": item.label, "group": f"entity-{item.type}"} for item in entities],
                *[{"id": item.id, "label": item.label, "group": "event"} for item in events],
                *[
                    {
                        "id": item.id,
                        "label": f"{item.reference_label} {max(0.0, item.similarity) * 100:.0f}%",
                        "group": "face-match",
                    }
                    for item in face_matches
                ],
            ]
            graph_edges = [
                {"source": f"CASE:{case.id}", "target": node["id"], "label": "contains"}
                for node in graph_nodes[1:]
            ]
            graph_edges.extend({"source": relation.source, "target": relation.target, "label": relation.label} for relation in relations)
            graph_edges.extend(
                {
                    "source": face_match.entity_id or f"CASE:{case.id}",
                    "target": face_match.id,
                    "label": "candidate match",
                }
                for face_match in face_matches
            )
            return {
                "id": case.id,
                "title": case.title,
                "status": case.status,
                "priority": case.priority,
                "summary": case.summary,
                "evidence": [_serialize_evidence(item) for item in evidence],
                "entities": [
                    {
                        "id": item.id,
                        "label": item.label,
                        "canonical_name": item.canonical_name,
                        "type": item.type,
                        "metadata": json_metadata(item.metadata_json),
                    }
                    for item in entities
                ],
                "events": [_serialize_event(item) for item in events],
                "ranking": [{"id": item.id, "label": item.label, "score": item.score, "reason": item.reason} for item in ranking],
                "contradictions": [_serialize_contradiction(item) for item in contradictions],
                "face_matches": [_serialize_face_match(item, embeddings_by_id.get(item.embedding_id)) for item in face_matches],
                "alerts": _case_alerts(contradictions, face_matches, embeddings_by_id),
                "audit": [{"at": item.at.isoformat(), "actor": item.actor, "action": item.action} for item in audit],
                "graph": {"nodes": graph_nodes, "edges": graph_edges},
            }
    case = next((item for item in state["cases"] if item["id"] == case_id), None)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return {**case, "graph": build_graph_payload(state, case_id)}


@router.get("/case/{case_id}/alerts")
def case_alerts(case_id: str) -> dict[str, list[dict[str, Any]]]:
    payload = case_detail(case_id)
    return {"items": payload.get("alerts", [])}


@router.get("/search")
def search(q: str = "") -> dict[str, list[dict[str, Any]]]:
    demo_state()
    query = q.strip().casefold()
    with SessionLocal() as session:
        records = list(session.scalars(select(Evidence)).all())
        hits = [
            _serialize_evidence(item)
            for item in records
            if not query or query in item.title.casefold() or query in item.notes.casefold()
        ]
    return {"items": hits}


@router.get("/case/{case_id}/timeline")
def case_timeline(case_id: str) -> dict[str, list[dict[str, Any]]]:
    case_detail_payload = case_detail(case_id)
    return {"items": case_detail_payload["events"]}


@router.get("/case/{case_id}/audit")
def case_audit(case_id: str) -> dict[str, list[dict[str, Any]]]:
    case_detail_payload = case_detail(case_id)
    return {"items": case_detail_payload["audit"]}


@router.get("/demo/workflow")
def demo_workflow() -> dict[str, list[str]]:
    return {
        "steps": [
            "Ingest a CCTV file and sample reviewable frames with OpenCV",
            "Generate 512-d InsightFace embeddings when available, with an explicit OpenCV fallback for local demos",
            "Persist vectors in FAISS or a durable NumPy matrix alongside frame timestamps and provenance",
            "Upload a reference photo to rank candidate frame similarities as Match Indicators (Requires Verification)",
            "Extract canonical subject-relation-object triples from source text and attach them to the case graph",
            "Run spatiotemporal speed checks against unverified CCTV match indicators",
            "Review alerts, timeline, graph links, source records, and reasoning traces before any decision",
        ]
    }
