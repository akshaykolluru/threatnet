# ThreatNet V2 Intelligence Discovery Engine

Human-in-the-loop investigation workspace and intelligence-discovery demo built for local use.

## What it includes

- Case, entity, evidence, statement, relationship, and event records
- CCTV ingestion that samples reviewable frames and creates 512-dimensional descriptors
- Durable FAISS vector index when available, with a persistent NumPy cosine-search fallback
- Reference-photo matching endpoint returning ranked "Match Indicator (Requires Verification)" candidates
- Transparent text extraction from statements, notes, and phone logs into canonical graph triples
- Spatiotemporal review alerts with a distance, elapsed-time, speed, and threshold reasoning trace
- Explainable investigator relevance ranking
- Interactive timeline and relationship graph with an Intelligence Alerts feed
- Audit log and role-based access metadata
- Idempotent synthetic V2 demo data, including a mocked face-match indicator and travel-speed conflict

## What it does not do

- It does not declare guilt or identity
- It does not treat AI output, extracted relationships, or similarity scores as proof
- It does not store raw CCTV binaries in the relational database
- It retains source text, source evidence, frame paths, timestamps, and reasoning traces for review

## Stack

- Frontend: Next.js, React, TypeScript
- Backend: FastAPI, Python
- Database: SQLite for demo, PostgreSQL-compatible schema
- Vision: InsightFace when installed; explicit OpenCV visual-descriptor fallback for local demo mode
- Retrieval: FAISS when installed, otherwise a durable NumPy cosine index
- Graph UI: Cytoscape.js

## Run locally

1. Copy `.env.example` to `.env` if you need custom storage or database locations.
2. Create and activate a virtual environment.
3. Install backend dependencies: `python -m pip install -r backend/requirements.txt`
4. Install frontend dependencies: `npm ci`
5. Seed the complete V2 workflow: `python seed.py`
6. Start the API and web app: `npm run dev`

`insightface` and `faiss-cpu` are optional accelerators. When they are available,
ThreatNet uses InsightFace embeddings and FAISS automatically. The local demo stays
runnable without them and labels its OpenCV fallback output accordingly.

### Windows presentation setup

Use PowerShell in the project folder. The command below creates a usable virtual
environment, seeds the fictional **CASE-RIYA-001** presentation case, and opens
separate API and web-server terminals:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\start-demo.ps1
```

If a prior `.venv` references a removed Python installation, recreate it once:

```powershell
.\start-demo.ps1 -ResetEnvironment
```

If Python is missing, install Python 3.12, close and reopen PowerShell, then run
the command again:

```powershell
winget install Python.Python.3.12
```

Open [http://127.0.0.1:3000](http://127.0.0.1:3000) after both terminals are ready.
The fictional Riya case includes a local workbook at
`backend/storage/demo-assets/case-riya-suspect-screening.xlsx` for the **Suspect
screening** view, plus generated synthetic Riverside CAM-07 frames for **CCTV
intake**. All records are labeled as fictional leads requiring human verification.

## Tests

- `python -m pytest`
- `npm run build --workspace web`

## Notes

This is a prototype foundation. Production hardening, biometric-model validation,
compliance review, access control, retention policy, and operational security are
not claimed unless separately implemented and verified.
