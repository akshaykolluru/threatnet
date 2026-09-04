from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .database import Entity, Event, Evidence, Relation


ExtractionSourceType = Literal["statement", "interrogation-note", "phone-log", "document"]
PERSON = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}"
ISO_TIMESTAMP = r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?"

_SIGHTING_PATTERNS = (
    re.compile(
        rf"(?P<subject>{PERSON})\s+(?:(?:was|were)\s+)?(?:sighted|seen|present|located|observed)"
        rf"\s+(?:at|in|near)\s+(?P<location>[A-Z][A-Za-z0-9' -]*?)(?=\s+(?:at|on)\s+{ISO_TIMESTAMP}|[.,;]|$)"
    ),
    re.compile(
        rf"(?P<subject>{PERSON})\s+(?:said|stated|claimed|reported)\s+(?:that\s+)?(?:he|she|they)?\s*"
        rf"(?:was|were)\s+(?:at|in|near)\s+(?P<location>[A-Z][A-Za-z0-9' -]*?)(?=\s+(?:at|on)\s+{ISO_TIMESTAMP}|[.,;]|$)"
    ),
)
_CALL_PATTERN = re.compile(
    rf"(?P<subject>{PERSON})\s+(?:called|phoned|contacted)\s+(?P<object>{PERSON})"
)
_OWNERSHIP_PATTERN = re.compile(
    rf"(?P<subject>{PERSON})\s+(?:owns|owned|drives|uses)\s+(?P<object>(?:a|an|the)\s+[^.,;]+|[A-Z0-9][^.,;]+)"
)
_TIMESTAMP_PATTERN = re.compile(ISO_TIMESTAMP)
_FROM_LOCATION_PATTERN = re.compile(r"\bfrom\s+(?P<location>[A-Z][A-Za-z0-9' -]*?)(?=[.,;]|$)")


@dataclass(frozen=True)
class CanonicalTriple:
    """A provenance-preserving relation inferred from one portion of source text."""

    subject: str
    relation: str
    object: str
    timestamp: datetime | None
    location: str
    source_text: str
    confidence: float


@dataclass(frozen=True)
class ExtractionResult:
    """Records created from a source document after regex fallback extraction."""

    evidence_id: str
    triples: list[CanonicalTriple]
    entity_ids: list[str]
    event_ids: list[str]


def extract_triples(
    raw_text: str,
    default_time: datetime | None = None,
    default_location: str = "",
) -> list[CanonicalTriple]:
    """Extract a small, transparent set of canonical triples from investigative text.

    This deterministic fallback is intentionally inspectable. A structured LLM adapter
    can be added ahead of it later, but the current service never sends case text to an
    external provider and always retains the exact source sentence for review.
    """

    triples: list[CanonicalTriple] = []
    for sentence in _split_sentences(raw_text):
        timestamp = _timestamp_from_sentence(sentence) or default_time
        sighting = _first_match(_SIGHTING_PATTERNS, sentence)
        if sighting:
            subject = _clean_label(sighting.group("subject"))
            location = _clean_label(sighting.group("location")) or default_location
            if subject and location:
                triples.append(
                    CanonicalTriple(
                        subject=subject,
                        relation="SIGHTED_AT",
                        object=location,
                        timestamp=timestamp,
                        location=location,
                        source_text=sentence,
                        confidence=0.72,
                    )
                )

        call = _CALL_PATTERN.search(sentence)
        if call:
            subject = _clean_label(call.group("subject"))
            target = _clean_label(call.group("object"))
            location_match = _FROM_LOCATION_PATTERN.search(sentence)
            location = _clean_label(location_match.group("location")) if location_match else default_location
            if subject and target:
                triples.append(
                    CanonicalTriple(
                        subject=subject,
                        relation="CALLED",
                        object=target,
                        timestamp=timestamp,
                        location=location,
                        source_text=sentence,
                        confidence=0.68,
                    )
                )

        ownership = _OWNERSHIP_PATTERN.search(sentence)
        if ownership:
            subject = _clean_label(ownership.group("subject"))
            target = _clean_label(ownership.group("object"))
            if subject and target:
                triples.append(
                    CanonicalTriple(
                        subject=subject,
                        relation="OWNS",
                        object=target,
                        timestamp=timestamp,
                        location=default_location,
                        source_text=sentence,
                        confidence=0.64,
                    )
                )
    return _deduplicate(triples)


def persist_extraction(
    session: Session,
    case_id: str,
    raw_text: str,
    source: str,
    source_type: ExtractionSourceType = "statement",
    default_time: datetime | None = None,
    default_location: str = "",
    evidence_id: str | None = None,
) -> ExtractionResult:
    """Save the source text, canonical entities, events, and graph relations in one transaction."""

    text = raw_text.strip()
    if not text:
        raise ValueError("Text extraction requires non-empty source text")
    triples = extract_triples(text, default_time=default_time, default_location=default_location)
    evidence = Evidence(
        id=evidence_id or f"TXT-{uuid4().hex[:8].upper()}",
        case_id=case_id,
        title=f"Extracted intelligence: {source_type}",
        type=source_type,
        notes=text,
        source=source.strip() or "investigator text entry",
        metadata_json=json.dumps(
            {
                "extraction_method": "transparent_regex_fallback",
                "triple_count": len(triples),
                "human_review_required": True,
            }
        ),
    )
    session.add(evidence)
    session.flush()

    entity_ids: list[str] = []
    event_ids: list[str] = []
    for triple in triples:
        subject = _find_or_create_entity(session, case_id, triple.subject, "person")
        object_type = _object_type(triple)
        target = _find_or_create_entity(session, case_id, triple.object, object_type)
        entity_ids.extend((subject.id, target.id))

        event_id = f"EVT-{uuid4().hex[:8].upper()}"
        event_kind = "presence" if triple.relation == "SIGHTED_AT" else "extracted_relation"
        event = Event(
            id=event_id,
            case_id=case_id,
            time=triple.timestamp or datetime.utcnow(),
            label=f"{subject.label} {triple.relation.replace('_', ' ').lower()} {target.label}",
            location=triple.location,
            source_evidence_id=evidence.id,
            entity_id=subject.id,
            kind=event_kind,
            metadata_json=json.dumps(
                {
                    "relation": triple.relation,
                    "object_entity_id": target.id,
                    "source_sentence": triple.source_text,
                    "extraction_confidence": triple.confidence,
                    "requires_human_verification": True,
                }
            ),
        )
        session.add(event)
        session.flush()
        session.add(
            Relation(
                case_id=case_id,
                source=subject.id,
                target=target.id,
                label=triple.relation,
                event_id=event.id,
                metadata_json=json.dumps(
                    {
                        "source_evidence_id": evidence.id,
                        "source_sentence": triple.source_text,
                        "confidence": triple.confidence,
                    }
                ),
            )
        )
        event_ids.append(event.id)

    return ExtractionResult(
        evidence_id=evidence.id,
        triples=triples,
        entity_ids=list(dict.fromkeys(entity_ids)),
        event_ids=event_ids,
    )


def _find_or_create_entity(session: Session, case_id: str, label: str, entity_type: str) -> Entity:
    canonical_name = _canonical_name(label)
    entity = session.scalar(
        select(Entity).where(
            Entity.case_id == case_id,
            func.lower(Entity.canonical_name) == canonical_name,
        )
    )
    if entity is not None:
        return entity
    entity = Entity(
        id=f"ENT-{uuid4().hex[:8].upper()}",
        case_id=case_id,
        label=label,
        canonical_name=canonical_name,
        type=entity_type,
        metadata_json=json.dumps({"created_by": "intelligence_extraction"}),
    )
    session.add(entity)
    session.flush()
    return entity


def _object_type(triple: CanonicalTriple) -> str:
    if triple.relation == "SIGHTED_AT":
        return "location"
    if triple.relation == "CALLED":
        return "person"
    if triple.relation == "OWNS":
        return "asset"
    return "unknown"


def _split_sentences(raw_text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", raw_text) if part.strip()]


def _timestamp_from_sentence(sentence: str) -> datetime | None:
    matched = _TIMESTAMP_PATTERN.search(sentence)
    if not matched:
        return None
    try:
        return datetime.fromisoformat(matched.group(0).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _first_match(patterns: tuple[re.Pattern[str], ...], sentence: str) -> re.Match[str] | None:
    for pattern in patterns:
        matched = pattern.search(sentence)
        if matched:
            return matched
    return None


def _clean_label(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .,:;")


def _canonical_name(value: str) -> str:
    return _clean_label(value).casefold()


def _deduplicate(triples: list[CanonicalTriple]) -> list[CanonicalTriple]:
    seen: set[tuple[str, str, str, datetime | None, str]] = set()
    unique: list[CanonicalTriple] = []
    for triple in triples:
        key = (
            _canonical_name(triple.subject),
            triple.relation,
            _canonical_name(triple.object),
            triple.timestamp,
            _canonical_name(triple.location),
        )
        if key not in seen:
            seen.add(key)
            unique.append(triple)
    return unique
