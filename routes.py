import json
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from typing import Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select

from .database import AuditLog, Case, Contradiction, Evidence, Event, Ranking, Relation, SessionLocal
from .ingestion import inspect_video
from .screening import screen_workbook
from .services import demo_state, build_graph_payload

router = APIRouter(prefix="/api")


def _storage_root() -> Path:
    return Path(os.getenv("STORAGE_DIR", str(Path(__file__).resolve().parent.parent / "storage")))


def _remove_storage_file(source: str) -> None:
    source_path = Path(source)
    if source_path.parts[:1] != ("storage",):
        return
    stored_file = _storage_root() / Path(*source_path.parts[1:])
    if stored_file.is_file():
        stored_file.unlink()


class CaseCreate(BaseModel):
    title: str
    summary: str = "New investigation case"
    priority: str = "Medium"


class CaseStatusUpdate(BaseModel):
    status: Literal["Open", "Closed"]


class EvidenceCreate(BaseModel):
    title: str
    type: str = "source"
    notes: str = ""
    source: str = "investigator entry"


class EventCreate(BaseModel):
    label: str
    location: str = ""
    time: Optional[str] = None


@router.get("/health")
def health():
    return {"status": "ok", "mode": "demo"}


@router.get("/cases")
def cases():
    demo_state()
    with SessionLocal() as session:
        return {"items": [{"id": c.id, "title": c.title, "status": c.status, "priority": c.priority, "summary": c.summary} for c in session.scalars(select(Case)).all()]}


@router.post("/cases")
def create_case(payload: CaseCreate):
    demo_state()
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Case title is required")
    with SessionLocal() as session:
        existing = session.scalars(select(Case.id)).all()
        next_number = max([int(case_id.split("-")[1]) for case_id in existing if case_id.startswith("CASE-") and case_id.split("-")[1].isdigit()] or [100]) + 1
        case = Case(id=f"CASE-{next_number}", title=title, status="Open", priority=payload.priority, summary=payload.summary.strip() or "New investigation case")
        session.add(case)
        session.add(AuditLog(actor="demo-investigator", action="created_case", target=case.id))
        session.commit()
        return {"id": case.id, "title": case.title, "status": case.status, "priority": case.priority, "summary": case.summary}


@router.patch("/case/{case_id}")
def update_case_status(case_id: str, payload: CaseStatusUpdate):
    demo_state()
    with SessionLocal() as session:
        case = session.get(Case, case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        case.status = payload.status
        session.add(AuditLog(actor="demo-investigator", action=f"status_changed_to_{payload.status.lower()}", target=case_id))
        session.commit()
        return {"id": case.id, "title": case.title, "status": case.status, "priority": case.priority, "summary": case.summary}


@router.delete("/cases/{case_id}")
def delete_case(case_id: str):
    demo_state()
    if case_id in {"CASE-101", "CASE-202", "CASE-303"}:
        raise HTTPException(status_code=409, detail="Built-in demo cases cannot be deleted; close the case instead")
    with SessionLocal() as session:
        case = session.get(Case, case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        evidence_sources = [item.source for item in session.scalars(select(Evidence).where(Evidence.case_id == case_id)).all()]
        for model in (Evidence, Event, Relation, Ranking, Contradiction):
            for item in session.scalars(select(model).where(model.case_id == case_id)).all():
                session.delete(item)
        session.add(AuditLog(actor="demo-investigator", action="deleted_case", target=case_id))
        session.delete(case)
        session.commit()
        for source in evidence_sources:
            _remove_storage_file(source)
        return {"deleted": case_id}


@router.post("/case/{case_id}/evidence")
def create_evidence(case_id: str, payload: EvidenceCreate):
    demo_state()
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Evidence title is required")
    with SessionLocal() as session:
        if not session.get(Case, case_id):
            raise HTTPException(status_code=404, detail="Case not found")
        evidence_id = f"EV-{uuid4().hex[:8].upper()}"
        item = Evidence(id=evidence_id, case_id=case_id, title=title, type=payload.type.strip() or "source", notes=payload.notes.strip(), source=payload.source.strip() or "investigator entry")
        session.add(item)
        session.add(AuditLog(actor="demo-investigator", action="added_evidence", target=case_id))
        session.commit()
        return {"id": item.id, "case_id": case_id, "title": item.title, "type": item.type, "notes": item.notes, "source": item.source}


@router.post("/case/{case_id}/evidence/image")
async def create_image_evidence(
    case_id: str,
    title: str = Form(...),
    type: str = Form("image"),
    notes: str = Form(""),
    source: str = Form("image upload"),
    image: UploadFile = File(...),
):
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
        if not session.get(Case, case_id):
            raise HTTPException(status_code=404, detail="Case not found")
        evidence_id = f"IMG-{uuid4().hex[:8].upper()}"
        relative_source = Path("storage") / "cases" / case_id / "evidence" / f"{evidence_id}{extension}"
        stored_file = _storage_root() / relative_source.relative_to("storage")
        stored_file.parent.mkdir(parents=True, exist_ok=True)
        stored_file.write_bytes(content)
        metadata = {"kind": "image", "filename": image.filename, "content_type": image.content_type, "size_bytes": len(content)}
        item = Evidence(id=evidence_id, case_id=case_id, title=clean_title, type=type.strip() or "image", notes=notes.strip(), source=str(relative_source), metadata_json=json.dumps(metadata))
        session.add(item)
        session.add(AuditLog(actor="demo-investigator", action="added_image_evidence", target=case_id))
        session.commit()
        return {"id": item.id, "case_id": case_id, "title": item.title, "type": item.type, "notes": item.notes, "source": item.source, "metadata": metadata}


@router.delete("/case/{case_id}/evidence/{evidence_id}")
def delete_evidence(case_id: str, evidence_id: str):
    demo_state()
    with SessionLocal() as session:
        if not session.get(Case, case_id):
            raise HTTPException(status_code=404, detail="Case not found")
        item = session.get(Evidence, evidence_id)
        if not item or item.case_id != case_id:
            raise HTTPException(status_code=404, detail="Evidence record not found")
        source = item.source
        for event in session.scalars(select(Event).where(Event.source_evidence_id == evidence_id)).all():
            event.source_evidence_id = None
        for relation in session.scalars(select(Relation).where(Relation.case_id == case_id, (Relation.source == evidence_id) | (Relation.target == evidence_id))).all():
            session.delete(relation)
        session.delete(item)
        session.add(AuditLog(actor="demo-investigator", action=f"deleted_evidence:{evidence_id}", target=case_id))
        session.commit()
        _remove_storage_file(source)
        return {"deleted": evidence_id, "case_id": case_id}


@router.post("/case/{case_id}/events")
def create_event(case_id: str, payload: EventCreate):
    demo_state()
    label = payload.label.strip()
    if not label:
        raise HTTPException(status_code=422, detail="Event label is required")
    with SessionLocal() as session:
        if not session.get(Case, case_id):
            raise HTTPException(status_code=404, detail="Case not found")
        event_id = f"EVT-{uuid4().hex[:8].upper()}"
        if payload.time:
            try:
                event_time = datetime.fromisoformat(payload.time.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="Event time must be a valid ISO timestamp") from exc
        else:
            event_time = datetime.utcnow()
        item = Event(id=event_id, case_id=case_id, time=event_time, label=label, location=payload.location.strip())
        session.add(item)
        session.add(AuditLog(actor="demo-investigator", action="added_event", target=case_id))
        session.commit()
        return {"id": item.id, "case_id": case_id, "time": item.time.isoformat(), "label": item.label, "location": item.location}


@router.delete("/case/{case_id}/events/{event_id}")
def delete_event(case_id: str, event_id: str):
    demo_state()
    with SessionLocal() as session:
        if not session.get(Case, case_id):
            raise HTTPException(status_code=404, detail="Case not found")
        item = session.get(Event, event_id)
        if not item or item.case_id != case_id:
            raise HTTPException(status_code=404, detail="Timeline event not found")
        for relation in session.scalars(select(Relation).where(Relation.case_id == case_id, (Relation.source == event_id) | (Relation.target == event_id))).all():
            session.delete(relation)
        session.delete(item)
        session.add(AuditLog(actor="demo-investigator", action=f"deleted_event:{event_id}", target=case_id))
        session.commit()
        return {"deleted": event_id, "case_id": case_id}


@router.get("/case/{case_id}")
def case_detail(case_id: str):
    state = demo_state()
    with SessionLocal() as session:
        case = session.get(Case, case_id)
        if case:
            evidence = session.scalars(select(Evidence).where(Evidence.case_id == case_id)).all()
            events = session.scalars(select(Event).where(Event.case_id == case_id).order_by(Event.time)).all()
            ranking = session.scalars(select(Ranking).where(Ranking.case_id == case_id).order_by(Ranking.score.desc())).all()
            contradictions = session.scalars(select(Contradiction).where(Contradiction.case_id == case_id)).all()
            relations = session.scalars(select(Relation).where(Relation.case_id == case_id)).all()
            audit = session.scalars(select(AuditLog).where(AuditLog.target == case_id).order_by(AuditLog.at)).all()
            payload = {
                "id": case.id, "title": case.title, "status": case.status, "priority": case.priority, "summary": case.summary,
                "evidence": [{"id": e.id, "case_id": e.case_id, "title": e.title, "type": e.type, "notes": e.notes, "confidence": e.confidence, "source": e.source} for e in evidence],
                "events": [{"id": e.id, "case_id": e.case_id, "time": e.time.isoformat(), "label": e.label, "location": e.location} for e in events],
                "ranking": [{"id": r.id, "label": r.label, "score": r.score, "reason": r.reason} for r in ranking],
                "contradictions": [{"id": c.id, "severity": c.severity, "summary": c.summary} for c in contradictions],
                "audit": [{"at": a.at.isoformat(), "actor": a.actor, "action": a.action} for a in audit],
            }
            graph_nodes = [{"id": f"CASE:{case.id}", "label": case.title, "group": "case"}] + [{"id": e.id, "label": e.title, "group": e.type} for e in evidence] + [{"id": e.id, "label": e.label, "group": "event"} for e in events]
            graph_edges = [{"source": f"CASE:{case.id}", "target": e.id, "label": "contains"} for e in evidence + events] + [{"source": r.source, "target": r.target, "label": r.label} for r in relations]
            graph = {"nodes": graph_nodes, "edges": graph_edges}
            return {**payload, "graph": graph}
    case = next((item for item in state["cases"] if item["id"] == case_id), None)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return {**case, "graph": build_graph_payload(state, case_id)}


@router.get("/search")
def search(q: str = ""):
    state = demo_state()
    ql = q.lower()
    hits = [item for item in state["evidence"] if ql in item["title"].lower() or ql in item["notes"].lower()]
    return {"items": hits}


@router.get("/case/{case_id}/timeline")
def case_timeline(case_id: str):
    case_detail_payload = case_detail(case_id)
    return {"items": case_detail_payload["events"]}


@router.get("/case/{case_id}/audit")
def case_audit(case_id: str):
    case_detail_payload = case_detail(case_id)
    return {"items": case_detail_payload["audit"]}


@router.post("/case/{case_id}/cctv/inspect")
async def inspect_cctv(case_id: str, file: UploadFile = File(...)):
    demo_state()
    with SessionLocal() as session:
        if not session.get(Case, case_id):
            raise HTTPException(status_code=404, detail="Case not found")
    if not file.filename:
        raise HTTPException(status_code=400, detail="A video filename is required")
    storage_root = Path(os.getenv("STORAGE_DIR", str(Path(__file__).resolve().parent.parent / "storage")))
    job_dir = storage_root / "cctv" / uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    source = job_dir / Path(file.filename).name
    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Demo uploads are limited to 100 MB")
    source.write_bytes(content)
    try:
        metadata = inspect_video(source, job_dir / "frames")
    except ValueError as exc:
        source.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    evidence_id = f"CCTV-{uuid4().hex[:8].upper()}"
    with SessionLocal() as session:
        session.add(Evidence(id=evidence_id, case_id=case_id, title=f"CCTV inspection: {file.filename}", type="cctv", notes="OpenCV-sampled frames. Any face or similarity result requires investigator review.", source=str(source), metadata_json=json.dumps(metadata)))
        session.add(AuditLog(actor="demo-investigator", action="inspected_cctv", target=case_id))
        session.commit()
    return {"evidence_id": evidence_id, "case_id": case_id, "source": str(source), "metadata": metadata, "interpretation": "Derived metadata and frames are investigative indicators, not proof."}


@router.post("/case/{case_id}/screening/upload")
async def upload_screening_sheet(case_id: str, file: UploadFile = File(...)):
    demo_state()
    with SessionLocal() as session:
        if not session.get(Case, case_id):
            raise HTTPException(status_code=404, detail="Case not found")
    if not file.filename or Path(file.filename).suffix.lower() not in {".xlsx", ".xlsm"}:
        raise HTTPException(status_code=415, detail="Upload an .xlsx or .xlsm workbook")
    storage_root = Path(os.getenv("STORAGE_DIR", str(Path(__file__).resolve().parent.parent / "storage"))) / "cases" / case_id / "screening"
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
    return {"case_id": case_id, "source": str(source), **result}


@router.get("/demo/workflow")
def demo_workflow():
    return {
        "steps": [
            "Ingest a CCTV file path or uploaded clip",
            "Sample frames with OpenCV",
            "Extract faces and optional embeddings",
            "Index reference images and compare via vector search",
            "Attach evidence references, timestamps, and provenance",
            "Review timeline, contradictions, and graph relationships"
        ]
    }
