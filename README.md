# Crime Investigation Prototype

Police-investigation case management and evidence-analysis demo built for local use.

## What it includes

- Case, suspect, person, location, vehicle, evidence, statement, and event records
- CCTV ingestion pipeline that samples frames, stores file references, and indexes derived embeddings
- FAISS-style vector search fallback implemented in pure Python for zero-cost demo mode
- Timeline reconstruction and contradiction detection
- Explainable investigator relevance ranking
- Interactive relationship graph data model
- Audit log and role-based access metadata
- Synthetic demo data and a sample CCTV workflow

## What it does not do

- It does not declare guilt
- It does not treat AI output as proof
- It does not store raw CCTV binaries in the relational database

## Stack

- Frontend: Next.js, React, TypeScript
- Backend: FastAPI, Python
- Database: SQLite for demo, PostgreSQL-compatible schema
- Vision: OpenCV, optional face embedding hooks, vector search fallback
- Graph UI: Cytoscape.js

## Run locally

1. Copy `.env.example` to `.env`
2. Start the API and web app:
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install -r backend/requirements.txt`
   - `npm install`
   - `npm run dev`

## Tests

- `npm test` (with the virtual environment activated)

## Notes

This is a prototype foundation. Production hardening, compliance review, operational security, and model validation are not claimed unless separately implemented and verified.
