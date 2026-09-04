-- ThreatNet V2 additions. Existing SQLite demo databases are migrated safely by
-- backend/api/database.py; this file documents the equivalent fresh-schema tables.
CREATE TABLE IF NOT EXISTS entities (
  id VARCHAR(64) PRIMARY KEY,
  case_id VARCHAR(64) NOT NULL REFERENCES cases(id),
  label VARCHAR(255) NOT NULL,
  canonical_name VARCHAR(255) NOT NULL,
  type VARCHAR(64) NOT NULL DEFAULT 'unknown',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS face_embeddings (
  id VARCHAR(64) PRIMARY KEY,
  case_id VARCHAR(64) NOT NULL REFERENCES cases(id),
  evidence_id VARCHAR(64) REFERENCES evidence(id),
  entity_id VARCHAR(64) REFERENCES entities(id),
  frame_path TEXT NOT NULL,
  timestamp_seconds REAL NOT NULL,
  captured_at TIMESTAMP,
  location VARCHAR(255) NOT NULL DEFAULT '',
  bbox_json TEXT NOT NULL DEFAULT '[]',
  embedding_backend VARCHAR(128) NOT NULL,
  created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS face_matches (
  id VARCHAR(64) PRIMARY KEY,
  case_id VARCHAR(64) NOT NULL REFERENCES cases(id),
  embedding_id VARCHAR(64) NOT NULL REFERENCES face_embeddings(id),
  entity_id VARCHAR(64) REFERENCES entities(id),
  reference_label VARCHAR(255) NOT NULL,
  similarity REAL NOT NULL,
  label VARCHAR(128) NOT NULL,
  interpretation TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL
);

-- Existing-table additions:
-- events.entity_id, events.kind, events.metadata_json
-- relations.event_id, relations.metadata_json
-- contradictions.entity_id, contradictions.source_event_id,
-- contradictions.conflicting_event_id, contradictions.reasoning_trace,
-- contradictions.created_at
