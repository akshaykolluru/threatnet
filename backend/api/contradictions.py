from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import Contradiction, Entity, Event, json_metadata


DEFAULT_MAX_FEASIBLE_SPEED_KMH = 45.0
# Local demo coordinates keep the rule deterministic and do not invoke external geocoding.
KNOWN_LOCATION_COORDINATES: dict[str, tuple[float, float]] = {
    "banjara hills": (17.4126, 78.4482),
    "jubilee hills": (17.4326, 78.4070),
    "panjagutta": (17.4292, 78.4518),
    "madhapur": (17.4486, 78.3910),
}


@dataclass(frozen=True)
class ContradictionFinding:
    """A newly-created speed-conflict alert and its calculation inputs."""

    contradiction_id: str
    entity_id: str
    distance_km: float
    elapsed_minutes: float
    required_speed_kmh: float


def configured_max_feasible_speed_kmh() -> float:
    """Read a conservative operational threshold without allowing an invalid value."""

    try:
        value = float(os.getenv("MAX_FEASIBLE_SPEED_KMH", str(DEFAULT_MAX_FEASIBLE_SPEED_KMH)))
        return value if value > 0 else DEFAULT_MAX_FEASIBLE_SPEED_KMH
    except ValueError:
        return DEFAULT_MAX_FEASIBLE_SPEED_KMH


def detect_spatiotemporal_contradictions(
    session: Session,
    case_id: str,
    entity_id: str | None = None,
    max_feasible_speed_kmh: float | None = None,
) -> list[ContradictionFinding]:
    """Create review alerts when a presence claim and face-match indicator conflict.

    The detector only considers known/geocoded locations and never concludes identity
    or intent. It compares a reported presence event with a face-match event for the
    same canonical entity and records its full velocity calculation for investigator
    review.
    """

    threshold = max_feasible_speed_kmh or configured_max_feasible_speed_kmh()
    statement = select(Event).where(
        Event.case_id == case_id,
        Event.entity_id.is_not(None),
        Event.location != "",
        Event.kind.in_(("presence", "face_match")),
    )
    if entity_id:
        statement = statement.where(Event.entity_id == entity_id)
    events = list(session.scalars(statement.order_by(Event.entity_id, Event.time)).all())
    entities = {
        entity.id: entity
        for entity in session.scalars(select(Entity).where(Entity.case_id == case_id)).all()
    }
    findings: list[ContradictionFinding] = []
    for candidate_entity_id, entity_events in _group_events(events).items():
        entity = entities.get(candidate_entity_id)
        if entity is None:
            continue
        for first, second in combinations(entity_events, 2):
            presence_event, face_match_event = _presence_and_match(first, second)
            if presence_event is None or face_match_event is None:
                continue
            comparison = _speed_comparison(presence_event, face_match_event)
            if comparison is None:
                continue
            distance_km, elapsed_minutes, required_speed_kmh = comparison
            if required_speed_kmh <= threshold:
                continue
            contradiction_id = _contradiction_id(case_id, entity.id, presence_event.id, face_match_event.id)
            if session.get(Contradiction, contradiction_id):
                continue

            severity = "high" if required_speed_kmh >= threshold * 2 else "medium"
            summary = (
                f"{entity.label} would need to travel {distance_km:.1f} km in {elapsed_minutes:.1f} minutes "
                f"({required_speed_kmh:.1f} km/h), above the {threshold:.1f} km/h review threshold."
            )
            trace = _reasoning_trace(
                entity.label,
                presence_event,
                face_match_event,
                distance_km,
                elapsed_minutes,
                required_speed_kmh,
                threshold,
            )
            session.add(
                Contradiction(
                    id=contradiction_id,
                    case_id=case_id,
                    severity=severity,
                    summary=summary,
                    entity_id=entity.id,
                    source_event_id=presence_event.id,
                    conflicting_event_id=face_match_event.id,
                    reasoning_trace=trace,
                )
            )
            findings.append(
                ContradictionFinding(
                    contradiction_id=contradiction_id,
                    entity_id=entity.id,
                    distance_km=distance_km,
                    elapsed_minutes=elapsed_minutes,
                    required_speed_kmh=required_speed_kmh,
                )
            )
    return findings


def _group_events(events: Iterable[Event]) -> dict[str, list[Event]]:
    grouped: dict[str, list[Event]] = {}
    for event in events:
        if event.entity_id:
            grouped.setdefault(event.entity_id, []).append(event)
    return grouped


def _presence_and_match(first: Event, second: Event) -> tuple[Event | None, Event | None]:
    if first.kind == "presence" and second.kind == "face_match":
        return first, second
    if second.kind == "presence" and first.kind == "face_match":
        return second, first
    return None, None


def _speed_comparison(presence_event: Event, face_match_event: Event) -> tuple[float, float, float] | None:
    coordinates_a = _event_coordinates(presence_event)
    coordinates_b = _event_coordinates(face_match_event)
    if coordinates_a is None or coordinates_b is None:
        return None
    seconds = abs((face_match_event.time - presence_event.time).total_seconds())
    if seconds <= 0:
        return None
    distance_km = _haversine_km(coordinates_a, coordinates_b)
    elapsed_hours = seconds / 3600
    return distance_km, seconds / 60, distance_km / elapsed_hours


def _event_coordinates(event: Event) -> tuple[float, float] | None:
    metadata = json_metadata(event.metadata_json)
    coordinates = metadata.get("coordinates")
    if isinstance(coordinates, list) and len(coordinates) == 2:
        try:
            return float(coordinates[0]), float(coordinates[1])
        except (TypeError, ValueError):
            return None
    return KNOWN_LOCATION_COORDINATES.get(event.location.strip().casefold())


def _haversine_km(origin: tuple[float, float], destination: tuple[float, float]) -> float:
    latitude_a, longitude_a = map(math.radians, origin)
    latitude_b, longitude_b = map(math.radians, destination)
    delta_latitude = latitude_b - latitude_a
    delta_longitude = longitude_b - longitude_a
    area = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(latitude_a) * math.cos(latitude_b) * math.sin(delta_longitude / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(math.sqrt(area))


def _contradiction_id(case_id: str, entity_id: str, presence_event_id: str, face_match_event_id: str) -> str:
    digest = hashlib.sha256(
        f"{case_id}:{entity_id}:{presence_event_id}:{face_match_event_id}".encode("utf-8")
    ).hexdigest()[:16].upper()
    return f"CX-{digest}"


def _reasoning_trace(
    entity_label: str,
    presence_event: Event,
    face_match_event: Event,
    distance_km: float,
    elapsed_minutes: float,
    required_speed_kmh: float,
    threshold: float,
) -> str:
    return (
        f"Source presence: {entity_label} at {presence_event.location} on {presence_event.time.isoformat()}. "
        f"CCTV similarity indicator: {entity_label} at {face_match_event.location} on {face_match_event.time.isoformat()}. "
        f"Known-coordinate distance is {distance_km:.2f} km; elapsed time is {elapsed_minutes:.2f} minutes; "
        f"required travel speed is {required_speed_kmh:.2f} km/h versus the configured maximum feasible speed "
        f"of {threshold:.2f} km/h. This is an automatically generated review alert based on source records and an "
        f"unverified similarity indicator, not proof of identity, travel, or wrongdoing."
    )
