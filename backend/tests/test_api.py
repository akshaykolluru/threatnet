from fastapi.testclient import TestClient
from pathlib import Path

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_cases():
    response = client.get("/api/cases")
    assert response.status_code == 200
    assert len(response.json()["items"]) >= 1


def test_case_detail_has_provenance_sections():
    response = client.get("/api/case/CASE-101")
    assert response.status_code == 200
    body = response.json()
    assert body["evidence"][0]["source"]
    assert body["graph"]["edges"]


def test_create_case():
    response = client.post("/api/cases", json={"title": "Test case"})
    assert response.status_code == 200
    assert response.json()["title"] == "Test case"


def test_new_records_update_case_timeline_and_graph():
    case = client.post("/api/cases", json={"title": "Record update case"}).json()
    case_id = case["id"]
    evidence = client.post(
        f"/api/case/{case_id}/evidence",
        json={"title": "New source record", "type": "statement", "source": "Interview 08", "notes": "Added during review"},
    )
    event = client.post(
        f"/api/case/{case_id}/events",
        json={"label": "New review event", "location": "Control room", "time": "2026-09-04T10:30:00"},
    )
    assert evidence.status_code == 200
    assert event.status_code == 200

    detail = client.get(f"/api/case/{case_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert any(item["title"] == "New source record" for item in body["evidence"])
    assert any(item["label"] == "New review event" for item in body["events"])
    assert any(edge["target"] == evidence.json()["id"] for edge in body["graph"]["edges"])
    assert any(edge["target"] == event.json()["id"] for edge in body["graph"]["edges"])

    closed = client.patch(f"/api/case/{case_id}", json={"status": "Closed"})
    assert closed.status_code == 200
    assert closed.json()["status"] == "Closed"

    deleted_evidence = client.delete(f"/api/case/{case_id}/evidence/{evidence.json()['id']}")
    deleted_event = client.delete(f"/api/case/{case_id}/events/{event.json()['id']}")
    assert deleted_evidence.status_code == 200
    assert deleted_event.status_code == 200
    updated = client.get(f"/api/case/{case_id}").json()
    assert all(item["id"] != evidence.json()["id"] for item in updated["evidence"])
    assert all(item["id"] != event.json()["id"] for item in updated["events"])
    assert any(entry["action"].startswith("deleted_") for entry in updated["audit"])

    deleted_case = client.delete(f"/api/cases/{case_id}")
    assert deleted_case.status_code == 200
    assert client.get(f"/api/case/{case_id}").status_code == 404
    assert client.delete("/api/cases/CASE-101").status_code == 409


def test_image_evidence_upload_is_case_scoped_and_deletable():
    case = client.post("/api/cases", json={"title": "Image evidence case"}).json()
    case_id = case["id"]
    response = client.post(
        f"/api/case/{case_id}/evidence/image",
        data={"title": "Camera still", "source": "Camera 04 / frame 18", "notes": "Image requires investigator review"},
        files={"image": ("camera-still.png", b"not-a-real-png-but-a-valid-upload-payload", "image/png")},
    )
    assert response.status_code == 200
    uploaded = response.json()
    assert uploaded["type"] == "image"
    assert uploaded["source"].startswith("storage/")
    assert client.get(f"/{uploaded['source']}").status_code == 200

    deleted = client.delete(f"/api/case/{case_id}/evidence/{uploaded['id']}")
    assert deleted.status_code == 200
    assert client.get(f"/{uploaded['source']}").status_code == 404


def test_v2_demo_seed_exposes_match_and_spatiotemporal_alert():
    response = client.get("/api/case/CASE-101")
    assert response.status_code == 200
    body = response.json()
    assert any(match["label"] == "Match Indicator (Requires Verification)" for match in body["face_matches"])
    contradiction = next(item for item in body["contradictions"] if item["reasoning_trace"])
    assert "required travel speed" in contradiction["reasoning_trace"]
    assert any(alert["type"] == "face_match" for alert in body["alerts"])
    assert any(alert["type"] == "contradiction" for alert in body["alerts"])


def test_text_extraction_creates_case_entities_events_and_relations():
    case = client.post("/api/cases", json={"title": "Extraction workflow case"}).json()
    response = client.post(
        f"/api/case/{case['id']}/intelligence/extract",
        json={
            "source": "Witness note 44",
            "source_type": "statement",
            "text": "Maya Shah was seen at Banjara Hills at 2026-09-04T20:00:00. Maya Shah called Rohan Das at 2026-09-04T20:01:00 from Banjara Hills.",
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert {triple["relation"] for triple in result["triples"]} == {"SIGHTED_AT", "CALLED"}
    detail = client.get(f"/api/case/{case['id']}").json()
    assert any(entity["label"] == "Maya Shah" for entity in detail["entities"])
    assert any(event["kind"] == "presence" for event in detail["events"])
    assert any(edge["label"] == "SIGHTED_AT" for edge in detail["graph"]["edges"])


def test_face_match_returns_ranked_indicator_candidates():
    detail = client.get("/api/case/CASE-101").json()
    arjun = next(entity for entity in detail["entities"] if entity["label"] == "Arjun Rao")
    reference_path = Path(__file__).resolve().parents[1] / "storage" / "cctv" / "v2-demo" / "arjun-rao-reference.jpg"
    response = client.post(
        "/api/cctv/match-face",
        data={"case_id": "CASE-101", "reference_label": "Arjun Rao", "entity_id": arjun["id"], "limit": "3"},
        files={"reference_photo": ("arjun-rao-reference.jpg", reference_path.read_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["matches"]
    assert result["matches"][0]["label"] == "Match Indicator (Requires Verification)"
    assert "Indicator Only, Not Proof" in result["matches"][0]["interpretation"]
