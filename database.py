from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'demo.db'}")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Case(Base):
    __tablename__ = "cases"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="Open")
    priority: Mapped[str] = mapped_column(String(32), default="Medium")
    summary: Mapped[str] = mapped_column(Text, default="")
    evidence: Mapped[list[Evidence]] = relationship(back_populates="case", cascade="all, delete-orphan")
    events: Mapped[list[Event]] = relationship(back_populates="case", cascade="all, delete-orphan")


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


class Event(Base):
    __tablename__ = "events"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    time: Mapped[datetime] = mapped_column(DateTime)
    label: Mapped[str] = mapped_column(String(255))
    location: Mapped[str] = mapped_column(String(255), default="")
    source_evidence_id: Mapped[Optional[str]] = mapped_column(ForeignKey("evidence.id"), nullable=True)
    case: Mapped[Case] = relationship(back_populates="events")


class Relation(Base):
    __tablename__ = "relations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    source: Mapped[str] = mapped_column(String(64))
    target: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(64))


class Ranking(Base):
    __tablename__ = "rankings"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    label: Mapped[str] = mapped_column(String(255))
    score: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)


class Contradiction(Base):
    __tablename__ = "contradictions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    severity: Mapped[str] = mapped_column(String(32))
    summary: Mapped[str] = mapped_column(Text)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    actor: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(128))
    target: Mapped[str] = mapped_column(String(128))


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def init_db(seed: dict[str, Any]) -> None:
    Path(str(BASE_DIR / "data")).mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        existing_cases = set(session.scalars(select(Case.id)).all())
        if existing_cases:
            for item in seed["cases"]:
                if item["id"] not in existing_cases:
                    session.add(Case(**item))
            existing_evidence = set(session.scalars(select(Evidence.id)).all())
            for item in seed["evidence"]:
                if item["id"] not in existing_evidence:
                    session.add(Evidence(**item))
            existing_events = set(session.scalars(select(Event.id)).all())
            for item in seed["events"]:
                if item["id"] not in existing_events:
                    session.add(Event(**{**item, "time": _parse_time(item["time"])}))
            existing_rankings = set(session.scalars(select(Ranking.id)).all())
            for item in seed["ranking"]:
                if item["id"] not in existing_rankings:
                    case_id = next((case["id"] for case in seed["cases"] if item["id"].startswith(case["id"])), seed["cases"][0]["id"])
                    session.add(Ranking(case_id=case_id, **item))
            existing_contradictions = set(session.scalars(select(Contradiction.id)).all())
            for item in seed["contradictions"]:
                if item["id"] not in existing_contradictions:
                    case_id = next((case["id"] for case in seed["cases"] if item["id"].startswith(case["id"])), seed["cases"][0]["id"])
                    session.add(Contradiction(case_id=case_id, **item))
            session.commit()
            return
        for item in seed["cases"]:
            session.add(Case(**item))
        for item in seed["evidence"]:
            session.add(Evidence(**item))
        for item in seed["events"]:
            session.add(Event(**{**item, "time": _parse_time(item["time"])}))
        case_id = seed["cases"][0]["id"]
        for item in seed["relations"]:
            session.add(Relation(case_id=case_id, **item))
        for item in seed["ranking"]:
            session.add(Ranking(case_id=case_id, **item))
        for item in seed["contradictions"]:
            session.add(Contradiction(case_id=case_id, **item))
        for item in seed["audit"]:
            session.add(AuditLog(at=_parse_time(item["at"]), actor=item["actor"], action=item["action"], target=item["target"]))
        session.commit()


def json_metadata(value: str) -> dict[str, Any]:
    try:
        return json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
