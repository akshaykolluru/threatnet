from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json

from .database import init_db


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEMO_FILE = DATA_DIR / "demo_state.json"


def _default_state():
    now = datetime(2026, 9, 4, 20, 0)
    case_id = "CASE-101"
    return {
        "cases": [
            {
                "id": case_id,
                "title": "Suspicious movement near Jubilee Hills",
                "status": "Open",
                "priority": "High",
                "summary": "Demo case with CCTV, witness statements, and vehicle sightings.",
            }
        ],
        "evidence": [
            {
                "id": "EV-1",
                "case_id": case_id,
                "title": "CCTV frame reference",
                "type": "cctv",
                "notes": "Frame sampled from shop camera; face match is an indicator only.",
                "confidence": 0.82,
                "source": "storage/cctv/frame-001.jpg",
            },
            {
                "id": "EV-2",
                "case_id": case_id,
                "title": "Witness statement",
                "type": "statement",
                "notes": "Witness saw a white hatchback around 20:15.",
                "confidence": 0.71,
                "source": "statement/W-11",
            },
        ],
        "events": [
            {"id": "EVT-1", "case_id": case_id, "time": (now).isoformat(), "label": "Camera sampled", "location": "Storefront"},
            {"id": "EVT-2", "case_id": case_id, "time": (now + timedelta(minutes=12)).isoformat(), "label": "Vehicle observed", "location": "Road junction"},
            {"id": "EVT-3", "case_id": case_id, "time": (now + timedelta(minutes=26)).isoformat(), "label": "Statement logged", "location": "Police station"},
        ],
        "relations": [
            {"source": "EV-1", "target": "EVT-1", "label": "supports"},
            {"source": "EV-2", "target": "EVT-2", "label": "correlates"},
        ],
        "timeline": [
            {"time": (now).isoformat(), "label": "CCTV ingestion started", "type": "system"},
            {"time": (now + timedelta(minutes=6)).isoformat(), "label": "Frame index created", "type": "system"},
            {"time": (now + timedelta(minutes=12)).isoformat(), "label": "Vehicle signal added", "type": "signal"},
        ],
        "audit": [
            {"at": (now + timedelta(minutes=2)).isoformat(), "actor": "demo-investigator", "action": "opened_case", "target": case_id}
        ],
        "ranking": [
            {"id": "PR-1", "label": "CCTV frame", "score": 0.92, "reason": "Strong visual similarity signal"},
            {"id": "PR-2", "label": "Vehicle sighting", "score": 0.74, "reason": "Matches route and time window"},
        ],
        "contradictions": [
            {"id": "CX-1", "severity": "medium", "summary": "Witness time differs from camera timestamp by 7 minutes."}
        ],
    }


def demo_state():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DEMO_FILE.exists():
        state = json.loads(DEMO_FILE.read_text())
        _expand_demo_state(state)
        DEMO_FILE.write_text(json.dumps(state, indent=2))
        init_db(state)
        # V2 data is generated from services so it remains idempotent and runnable
        # even when the JSON seed predates the intelligence-engine schema.
        from .seed_data import ensure_v2_demo_data

        ensure_v2_demo_data()
        return state
    state = _default_state()
    _expand_demo_state(state)
    DEMO_FILE.write_text(json.dumps(state, indent=2))
    init_db(state)
    from .seed_data import ensure_v2_demo_data

    ensure_v2_demo_data()
    return state


def _expand_demo_state(state):
    existing = {item["id"] for item in state["cases"]}
    additions = [
        ("CASE-202", "Warehouse access review", "Medium", "Access-card events, staff statements, and delivery records for a second demo investigation."),
        ("CASE-303", "Night vehicle pattern", "High", "Vehicle sightings and call-log signals across a late-night route."),
        (
            "CASE-RIYA-001",
            "Riya Sharma Homicide Investigation — Fictional Demo",
            "High",
            "Fictional presentation case. Source-linked records support review of Rohan Mehta as a possible organizer and Vikram Sethi as a possible direct perpetrator; all findings require human verification.",
        ),
    ]
    for case_id, title, priority, summary in additions:
        if case_id in existing:
            continue
        state["cases"].append({"id": case_id, "title": title, "status": "Open", "priority": priority, "summary": summary})
        state["evidence"].extend([{ "id": f"{case_id}-EV1", "case_id": case_id, "title": "Reference statement", "type": "statement", "notes": "Synthetic statement record for demo review.", "confidence": 0.64, "source": f"statement/{case_id}-W1" }, { "id": f"{case_id}-EV2", "case_id": case_id, "title": "Location signal", "type": "location", "notes": "Synthetic location signal requiring corroboration.", "confidence": 0.58, "source": f"location/{case_id}-L1" }])
        state["events"].extend([{ "id": f"{case_id}-EVT1", "case_id": case_id, "time": "2026-09-05T20:10:00", "label": "Source record added", "location": "Case desk" }, { "id": f"{case_id}-EVT2", "case_id": case_id, "time": "2026-09-05T20:28:00", "label": "Signal cross-check queued", "location": "Review queue" }])
        state["relations"].extend([{ "source": f"{case_id}-EV1", "target": f"{case_id}-EVT1", "label": "supports" }, { "source": f"{case_id}-EV2", "target": f"{case_id}-EVT2", "label": "correlates" }])
        state["ranking"].append({ "id": f"{case_id}-PR1", "label": "Reference statement", "score": 0.64, "reason": "Source record needs corroboration" })
        state["contradictions"].append({"id": f"{case_id}-CX1", "severity": "low", "summary": "Synthetic source records need cross-checking."})
        state["audit"].append({"at": "2026-09-05T20:02:00", "actor": "demo-investigator", "action": "opened_case", "target": case_id})


def build_graph_payload(state, case_id):
    nodes = []
    edges = []
    for item in state["evidence"]:
        if item["case_id"] == case_id:
            nodes.append({"id": item["id"], "label": item["title"], "group": item["type"]})
    for event in state["events"]:
        if event["case_id"] == case_id:
            nodes.append({"id": event["id"], "label": event["label"], "group": "event"})
    for rel in state["relations"]:
        edges.append(rel)
    case = next(item for item in state["cases"] if item["id"] == case_id)
    nodes.insert(0, {"id": f"CASE:{case_id}", "label": case["title"], "group": "case"})
    edges = [{"source": f"CASE:{case_id}", "target": node["id"], "label": "contains"} for node in nodes[1:]] + edges
    return {"nodes": nodes, "edges": edges}
