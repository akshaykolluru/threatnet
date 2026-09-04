from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'demo.db'}")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for the local investigation store."""


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="Open")
    priority: Mapped[str] = mapped_column(String(32), default="Medium")
    summary: Mapped[str] = mapped_column(Text, default="")
    evidence: Mapped[list[Evidence]] = relationship(back_populates="case", cascade="all, delete-orphan")
    events: Mapped[list[Event]] = relationship(back_populates="case", cascade="all, delete-orphan")
    entities: Mapped[list[Entity]] = relationship(back_populates="case", cascade="all, delete-orphan")


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    type: Mapped[str] = mapped_column(String(64))
    notes: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(Text, default="")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    case: Mapped[Case] = relationship(back_populates="evidence")


class Entity(Base):
    """Canonical person, location, organization, vehicle, or other graph entity."""

    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    label: Mapped[str] = mapped_column(String(255))
    canonical_name: Mapped[str] = mapped_column(String(255), index=True)
    type: Mapped[str] = mapped_column(String(64), default="unknown")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    case: Mapped[Case] = relationship(back_populates="entities")
    events: Mapped[list[Event]] = relationship(back_populates="entity")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    time: Mapped[datetime] = mapped_column(DateTime)
    label: Mapped[str] = mapped_column(String(255))
    location: Mapped[str] = mapped_column(String(255), default="")
    source_evidence_id: Mapped[Optional[str]] = mapped_column(ForeignKey("evidence.id"), nullable=True)
    entity_id: Mapped[Optional[str]] = mapped_column(ForeignKey("entities.id"), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(64), default="observation")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    case: Mapped[Case] = relationship(back_populates="events")
    entity: Mapped[Optional[Entity]] = relationship(back_populates="events")


class Relation(Base):
    __tablename__ = "relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    source: Mapped[str] = mapped_column(String(64))
    target: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(64))
    event_id: Mapped[Optional[str]] = mapped_column(ForeignKey("events.id"), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


class Ranking(Base):
    __tablename__ = "rankings"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    label: Mapped[str] = mapped_column(String(255))
    score: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)


class Contradiction(Base):
    """A review alert with a traceable, non-conclusive reasoning record."""

    __tablename__ = "contradictions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    severity: Mapped[str] = mapped_column(String(32))
    summary: Mapped[str] = mapped_column(Text)
    entity_id: Mapped[Optional[str]] = mapped_column(ForeignKey("entities.id"), nullable=True, index=True)
    source_event_id: Mapped[Optional[str]] = mapped_column(ForeignKey("events.id"), nullable=True)
    conflicting_event_id: Mapped[Optional[str]] = mapped_column(ForeignKey("events.id"), nullable=True)
    reasoning_trace: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FaceEmbedding(Base):
    """Metadata for a 512-d frame descriptor held in the durable vector index."""

    __tablename__ = "face_embeddings"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    evidence_id: Mapped[Optional[str]] = mapped_column(ForeignKey("evidence.id"), nullable=True, index=True)
    entity_id: Mapped[Optional[str]] = mapped_column(ForeignKey("entities.id"), nullable=True, index=True)
    frame_path: Mapped[str] = mapped_column(Text)
    timestamp_seconds: Mapped[float] = mapped_column(Float)
    captured_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    location: Mapped[str] = mapped_column(String(255), default="")
    bbox_json: Mapped[str] = mapped_column(Text, default="[]")
    embedding_backend: Mapped[str] = mapped_column(String(128), default="opencv_visual_descriptor")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FaceMatch(Base):
    """A stored candidate comparison for the intelligence-alert feed."""

    __tablename__ = "face_matches"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    embedding_id: Mapped[str] = mapped_column(ForeignKey("face_embeddings.id"), index=True)
    entity_id: Mapped[Optional[str]] = mapped_column(ForeignKey("entities.id"), nullable=True, index=True)
    reference_label: Mapped[str] = mapped_column(String(255))
    similarity: Mapped[float] = mapped_column(Float)
    label: Mapped[str] = mapped_column(String(128), default="Match Indicator (Requires Verification)")
    interpretation: Mapped[str] = mapped_column(Text, default="Similarity output is an indicator only, not proof.")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    actor: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(128))
    target: Mapped[str] = mapped_column(String(128))


_SQLITE_COLUMN_MIGRATIONS: dict[str, dict[str, str]] = {
    "events": {
        "entity_id": "VARCHAR(64)",
        "kind": "VARCHAR(64) NOT NULL DEFAULT 'observation'",
        "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
    },
    "relations": {
        "event_id": "VARCHAR(64)",
        "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
    },
    "contradictions": {
        "entity_id": "VARCHAR(64)",
        "source_event_id": "VARCHAR(64)",
        "conflicting_event_id": "VARCHAR(64)",
        "reasoning_trace": "TEXT NOT NULL DEFAULT ''",
        "created_at": "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP",
    },
}


def ensure_schema() -> None:
    """Create V2 tables and add safe columns to an existing SQLite demo database."""

    Base.metadata.create_all(engine)
    if not DATABASE_URL.startswith("sqlite"):
        return

    inspector = inspect(engine)
    with engine.begin() as connection:
        for table_name, expected_columns in _SQLITE_COLUMN_MIGRATIONS.items():
            if table_name not in inspector.get_table_names():
                continue
            existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, definition in expected_columns.items():
                if column_name not in existing_columns:
                    connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"))


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def _seed_case_id(item: dict[str, Any], seed: dict[str, Any]) -> str:
    """Keep legacy seed rows that omit case_id compatible with the V2 schema."""

    if item.get("case_id"):
        return str(item["case_id"])
    for case in seed["cases"]:
        if str(item.get("id", "")).startswith(case["id"]):
            return str(case["id"])
    return str(seed["cases"][0]["id"])


def _insert_missing(session: Session, model: type[Any], rows: list[dict[str, Any]], id_field: str = "id") -> None:
    existing = set(session.scalars(select(getattr(model, id_field))).all())
    for row in rows:
        if row[id_field] not in existing:
            session.add(model(**row))


def init_db(seed: dict[str, Any]) -> None:
    """Initialize legacy seed records without overwriting investigator-created data."""

    Path(str(BASE_DIR / "data")).mkdir(parents=True, exist_ok=True)
    ensure_schema()
    with SessionLocal() as session:
        _insert_missing(session, Case, list(seed.get("cases", [])))
        _insert_missing(session, Evidence, list(seed.get("evidence", [])))

        event_rows = [{**item, "time": _parse_time(item["time"])} for item in seed.get("events", [])]
        _insert_missing(session, Event, event_rows)

        relation_rows = [
            {**item, "case_id": _seed_case_id(item, seed)}
            for item in seed.get("relations", [])
        ]
        existing_relation_keys = {
            (row.case_id, row.source, row.target, row.label)
            for row in session.scalars(select(Relation)).all()
        }
        for row in relation_rows:
            key = (row["case_id"], row["source"], row["target"], row["label"])
            if key not in existing_relation_keys:
                session.add(Relation(**row))

        ranking_rows = [{**item, "case_id": _seed_case_id(item, seed)} for item in seed.get("ranking", [])]
        _insert_missing(session, Ranking, ranking_rows)

        contradiction_rows = [{**item, "case_id": _seed_case_id(item, seed)} for item in seed.get("contradictions", [])]
        _insert_missing(session, Contradiction, contradiction_rows)

        existing_audit = {(row.actor, row.action, row.target) for row in session.scalars(select(AuditLog)).all()}
        for item in seed.get("audit", []):
            key = (item["actor"], item["action"], item["target"])
            if key not in existing_audit:
                session.add(AuditLog(at=_parse_time(item["at"]), actor=item["actor"], action=item["action"], target=item["target"]))
        session.commit()


def json_metadata(value: str) -> dict[str, Any]:
    """Read metadata defensively so historic malformed rows remain displayable."""

    try:
        return json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
