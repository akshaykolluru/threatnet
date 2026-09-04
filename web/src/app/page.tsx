"use client";

import { ChangeEvent, DragEvent, FormEvent, RefObject, useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
type View = "overview" | "timeline" | "mindmap" | "cctv" | "screening";
type CaseItem = { id: string; title: string; status: string; priority: string; summary: string };
type CctvFrame = { path: string; frame_index: number; timestamp_seconds: number; face_candidates?: number; face_detected?: boolean };
type Evidence = { id: string; title: string; type: string; notes: string; confidence?: number; source: string; metadata?: { frames?: CctvFrame[]; duration_seconds?: number; fps?: number } };
type GraphNode = { id: string; label: string; group: string };
type GraphEdge = { source: string; target: string; label: string };
type TimelineEvent = { id: string; time: string; label: string; location: string; entity_id?: string | null; kind?: string };
type IntelligenceAlert = { id: string; type: string; severity: string; title: string; summary: string; reasoning_trace?: string; created_at?: string | null };
type Detail = CaseItem & { evidence: Evidence[]; events: TimelineEvent[]; ranking: any[]; contradictions: any[]; audit: any[]; alerts?: IntelligenceAlert[]; face_matches?: any[]; graph?: { nodes: GraphNode[]; edges: GraphEdge[] } };

function storageUrl(path?: string): string {
  if (!path) return "";
  const normalized = path.replace(/\\/g, "/");
  if (normalized.startsWith("storage/")) return `${API_BASE}/${normalized}`;
  if (normalized.startsWith("/storage/")) return `${API_BASE}${normalized}`;
  return "";
}

function evidenceImageUrl(item: Evidence): string {
  return item.type === "image" && item.source.startsWith("storage/") ? storageUrl(item.source) : "";
}

export default function Home() {
  const [cases, setCases] = useState<CaseItem[]>([]);
  const [active, setActive] = useState("");
  const [detail, setDetail] = useState<Detail | null>(null);
  const [view, setView] = useState<View>("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [pendingKind, setPendingKind] = useState<"video" | "sheet" | null>(null);
  const [screening, setScreening] = useState<any>(null);
  const [hovered, setHovered] = useState<Evidence | null>(null);
  const [showAddCase, setShowAddCase] = useState(false);
  const [newCaseTitle, setNewCaseTitle] = useState("");
  const [caseMessage, setCaseMessage] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const sheetRef = useRef<HTMLInputElement>(null);

  async function loadCases() {
    const response = await fetch(`${API_BASE}/api/cases`);
    if (!response.ok) throw Error("Cases could not be loaded");
    const data = await response.json();
    setCases(data.items ?? []);
    setActive((current) => current || data.items?.[0]?.id || "");
  }

  async function loadDetail(id: string) {
    const response = await fetch(`${API_BASE}/api/case/${id}`);
    if (!response.ok) throw Error("Case details could not be loaded");
    setDetail(await response.json());
  }

  async function addCase() {
    if (!newCaseTitle.trim()) return;
    const response = await fetch(`${API_BASE}/api/cases`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: newCaseTitle }) });
    const created = await response.json();
    if (!response.ok) { setError(created.detail || "Case could not be created"); return; }
    setCases((items) => [...items, created]); setActive(created.id); setNewCaseTitle(""); setShowAddCase(false); setView("overview");
  }

  async function toggleCaseStatus() {
    if (!active || !detail) return;
    const nextStatus = detail.status === "Closed" ? "Open" : "Closed";
    if (!window.confirm(`${nextStatus === "Closed" ? "Close" : "Reopen"} ${active}?`)) return;
    setCaseMessage("");
    try {
      const response = await fetch(`${API_BASE}/api/case/${active}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: nextStatus }) });
      const updated = await response.json();
      if (!response.ok) throw Error(updated.detail || "Case status could not be updated");
      setCases((items) => items.map((item) => item.id === active ? { ...item, status: updated.status } : item));
      setDetail((current) => current ? { ...current, status: updated.status } : current);
      setCaseMessage(nextStatus === "Closed" ? "Case closed. Reopen it to add new records." : "Case reopened.");
    } catch (reason: any) { setCaseMessage(reason.message); }
  }

  async function deleteRecord(kind: "evidence" | "event", id: string): Promise<boolean> {
    const label = kind === "event" ? "timeline event" : "evidence record";
    if (!window.confirm(`Delete this ${label}? This cannot be undone.`)) return false;
    const path = kind === "event" ? "events" : "evidence";
    const response = await fetch(`${API_BASE}/api/case/${active}/${path}/${id}`, { method: "DELETE" });
    const data = await response.json();
    if (!response.ok) throw Error(data.detail || `${label} could not be deleted`);
    await loadDetail(active);
    return true;
  }

  async function deleteCase() {
    if (!active || !window.confirm(`Permanently delete ${active}? All of its records will be removed.`)) return;
    setCaseMessage("");
    try {
      const response = await fetch(`${API_BASE}/api/cases/${active}`, { method: "DELETE" });
      const data = await response.json();
      if (!response.ok) throw Error(data.detail || "Case could not be deleted");
      const remaining = cases.filter((item) => item.id !== active);
      setCases(remaining); setDetail(null); setView("overview"); setActive(remaining[0]?.id || "");
    } catch (reason: any) { setCaseMessage(reason.message); }
  }

  useEffect(() => { loadCases().catch((reason) => setError(reason.message)); }, []);
  useEffect(() => { if (!active) return; setLoading(true); loadDetail(active).catch((reason) => setError(reason.message)).finally(() => setLoading(false)); }, [active]);

  async function uploadVideo(file?: File) {
    if (!file || !active) return;
    setUploading(true); setUploadMessage("Analyzing video and sampling frames...");
    const form = new FormData(); form.append("file", file);
    try {
      const response = await fetch(`${API_BASE}/api/case/${active}/cctv/inspect`, { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw Error(data.detail || "CCTV inspection failed");
      setUploadMessage(`Indexed ${data.metadata.frames.length} frames and attached ${data.evidence_id}.`);
      await loadDetail(active);
    } catch (reason: any) {
      setUploadMessage(reason?.message === "Failed to fetch" ? `Cannot reach the analysis API at ${API_BASE}. Start the API, then try Analyze video again.` : reason.message);
    } finally { setUploading(false); }
  }

  async function uploadSheet(file?: File) {
    if (!file || !active) return;
    setUploading(true); setUploadMessage("Analyzing suspect and call-log rows...");
    const form = new FormData(); form.append("file", file);
    try { const response = await fetch(`${API_BASE}/api/case/${active}/screening/upload`, { method: "POST", body: form }); const data = await response.json(); if (!response.ok) throw Error(data.detail || "Spreadsheet screening failed"); setScreening(data); setUploadMessage(`Narrowed ${data.input_rows} rows to ${data.shortlist_size} review candidates.`); } catch (reason: any) { setUploadMessage(reason.message); } finally { setUploading(false); }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>, kind: "video" | "sheet") { event.preventDefault(); const file = event.dataTransfer.files[0]; setPendingFile(file); setPendingKind(kind); }
  function chooseFile(event: ChangeEvent<HTMLInputElement>, kind: "video" | "sheet") { setPendingFile(event.target.files?.[0] ?? null); setPendingKind(kind); }

  return <main><div className="shell">
    <header className="topbar"><div className="brand"><span className="brand-mark">SI</span><span>CASELINE <small>evidence workspace</small></span></div><span className="secure-label">LOCAL DEMO / HUMAN REVIEW</span></header>
    <section className="hero"><div><div className="kicker">Investigation desk</div><h1>Turn scattered signals into a reviewable case story.</h1><p className="muted">A structured workspace for evidence, timelines, relationships, and investigator judgment.</p></div><div className="hero-status"><span className="pulse" />Demo mode active<br /><small>AI outputs are indicators, never proof.</small></div></section>
    <section className="workspace-grid"><aside className="case-rail panel"><div className="rail-heading"><div><span className="eyebrow">Workspace</span><h2>Cases</h2></div><button className="count" onClick={() => setShowAddCase((value) => !value)}>+ Add case</button></div>{showAddCase && <div className="add-case"><input autoFocus placeholder="Case title" value={newCaseTitle} onChange={(event) => setNewCaseTitle(event.target.value)} onKeyDown={(event) => event.key === "Enter" && addCase()} /><button onClick={addCase}>Create</button></div>}<div className="case-list">{cases.map((item) => <button className={`case-card ${active === item.id ? "active" : ""}`} key={item.id} onClick={() => { setActive(item.id); setHovered(null); }}><span className={`priority ${item.priority.toLowerCase()}`} /><span><strong>{item.id}</strong><b>{item.title}</b><small>{item.status} · {item.priority} priority</small></span></button>)}{!cases.length && !error && <div className="empty-state">Loading cases...</div>}{error && <div className="error-state">{error}<button onClick={() => loadCases()}>Retry</button></div>}</div></aside>
      <div className="content-column">{detail && <><div className="case-heading"><div><span className="eyebrow">Active investigation / {detail.id}</span><h2>{detail.title}</h2><p className="muted">{detail.summary}</p></div><div className="case-actions"><span className="status-chip">● {detail.status}</span><button className="status-action" onClick={toggleCaseStatus}>{detail.status === "Closed" ? "Reopen case" : "Close case"}</button><button className="delete-case" onClick={deleteCase}>Delete case</button>{caseMessage && <small className="case-message">{caseMessage}</small>}</div></div><nav className="view-nav" aria-label="Case views">{([['overview', 'Command view'], ['timeline', 'Timeline'], ['mindmap', 'Mind map'], ['cctv', 'CCTV intake'], ['screening', 'Suspect screening']] as [View, string][]).map(([key, label]) => <button key={key} className={view === key ? "selected" : ""} onClick={() => setView(key)}>{label}</button>)}</nav>{loading ? <div className="panel empty-state">Loading case intelligence...</div> : view === "overview" ? <Overview detail={detail} setView={setView} /> : view === "timeline" ? <Timeline detail={detail} onSaved={() => loadDetail(active)} onDelete={deleteRecord} /> : view === "mindmap" ? <MindMap detail={detail} hovered={hovered} setHovered={setHovered} onSaved={() => loadDetail(active)} onDelete={deleteRecord} /> : view === "cctv" ? <Cctv evidence={detail.evidence} file={pendingKind === "video" ? pendingFile : null} onDrop={(event: DragEvent<HTMLDivElement>) => handleDrop(event, "video")} onFile={(event: ChangeEvent<HTMLInputElement>) => chooseFile(event, "video")} onAnalyze={() => uploadVideo(pendingFile ?? undefined)} onBrowse={() => fileRef.current?.click()} inputRef={fileRef} uploading={uploading} message={uploadMessage} /> : <Screening result={screening} file={pendingKind === "sheet" ? pendingFile : null} onDrop={(event: DragEvent<HTMLDivElement>) => handleDrop(event, "sheet")} onFile={(event: ChangeEvent<HTMLInputElement>) => chooseFile(event, "sheet")} onAnalyze={() => uploadSheet(pendingFile ?? undefined)} onBrowse={() => sheetRef.current?.click()} inputRef={sheetRef} uploading={uploading} message={uploadMessage} />}</>}</div>
    </section>
  </div></main>;
}

function Overview({ detail, setView }: { detail: Detail; setView: (view: View) => void }) {
  const alerts = detail.alerts ?? [];
  return <><section className="metric-row"><Metric label="Evidence items" value={detail.evidence.length} /><Metric label="Timeline events" value={detail.events.length} /><Metric label="Intelligence alerts" value={alerts.length} /><Metric label="Audit entries" value={detail.audit.length} /></section><section className="dashboard-grid"><Panel title="Investigator relevance" action="Explainable signals"><div className="signal-list">{detail.ranking.map((item) => <div className="signal" key={item.id}><div><strong>{item.label}</strong><small>{item.reason}</small></div><b>{Math.round(item.score * 100)}%</b><span className="signal-bar"><i style={{ width: String(item.score * 100) + "%" }} /></span></div>)}</div></Panel><Panel title="Intelligence alerts" action={String(alerts.length) + " active"}><IntelligenceAlertFeed alerts={alerts} compact /></Panel></section><section className="dashboard-grid lower"><Panel title="Recent evidence" action="Source-linked records"><div className="evidence-list">{detail.evidence.map((item) => <div className="evidence-row" key={item.id}><span className="evidence-thumb">{item.type === "cctv" ? "O" : "[]"}</span><div><strong>{item.title}</strong><small>{item.notes}</small></div><span className="type-tag">{item.type}</span></div>)}</div></Panel><div className="quick-actions panel"><span className="eyebrow">Navigate</span><h3>See the story in context</h3><button onClick={() => setView("timeline")}>Open timeline <span>-&gt;</span></button><button onClick={() => setView("mindmap")}>Explore mind map <span>-&gt;</span></button><button onClick={() => setView("cctv")}>Drop in CCTV <span>-&gt;</span></button></div></section></>;
}

function IntelligenceAlertFeed({ alerts, compact = false }: { alerts: IntelligenceAlert[]; compact?: boolean }) {
  if (!alerts.length) return <div className="alert-empty">No automated alerts. New signals will remain reviewable source records.</div>;
  return <div className={compact ? "intelligence-alerts compact" : "intelligence-alerts"}>{alerts.map((alert) => <article className={"intelligence-alert " + alert.type} key={alert.id}><div className="alert-heading"><span className="alert-kind">{alert.type === "face_match" ? "MATCH INDICATOR" : "SPEED CONFLICT"}</span><span className="alert-severity">{alert.severity}</span></div><strong>{alert.title}</strong><p>{alert.summary}</p>{!compact && alert.reasoning_trace && <small>{alert.reasoning_trace}</small>}<em>Human verification required. Indicator only, not proof.</em></article>)}</div>;
}
function Timeline({ detail, onSaved, onDelete }: { detail: Detail; onSaved: () => Promise<void>; onDelete: (kind: "evidence" | "event", id: string) => Promise<boolean> }) {
  const [message, setMessage] = useState(""); const [removing, setRemoving] = useState("");
  async function remove(id: string) { setRemoving(id); setMessage(""); try { if (await onDelete("event", id)) setMessage("Timeline event deleted and the case map updated."); } catch (reason: any) { setMessage(reason.message); } finally { setRemoving(""); } }
  const closed = detail.status === "Closed";
  return <><section className="panel timeline-panel"><div className="panel-title"><div><span className="eyebrow">Chronology</span><h3>Case timeline</h3></div><span className="outline-chip">{detail.events.length} events</span></div>{closed && <div className="closed-notice">This case is closed. Reopen it before adding new records.</div>}<AddEventForm caseId={detail.id} onSaved={onSaved} disabled={closed} /><div className="timeline">{detail.events.length ? detail.events.map((event) => <div className="timeline-item" key={event.id}><span className="timeline-dot" /><div className="timeline-content"><time>{new Date(event.time).toLocaleString()}</time><h4>{event.label}</h4><p>{event.location || "Location not recorded"}</p>{event.kind === "face_match" && <small className="match-indicator">Similarity candidate - indicator only, not proof.</small>}</div><button className="delete-record" disabled={removing === event.id} onClick={() => remove(event.id)}>{removing === event.id ? "Deleting..." : "Delete"}</button></div>) : <div className="empty-state">No timeline events yet. Add the first event above.</div>}</div>{message && <div className="record-status">{message}</div>}</section><section className="panel intelligence-alert-panel"><div className="panel-title"><div><span className="eyebrow">Automated review</span><h3>Intelligence alerts</h3></div><span className="outline-chip">{(detail.alerts ?? []).length} alerts</span></div><IntelligenceAlertFeed alerts={detail.alerts ?? []} /></section></>;
}

function MindMap({ detail, hovered, setHovered, onSaved, onDelete }: { detail: Detail; hovered: Evidence | null; setHovered: (e: Evidence | null) => void; onSaved: () => Promise<void>; onDelete: (kind: "evidence" | "event", id: string) => Promise<boolean> }) {
  const previewImage = hovered ? evidenceImageUrl(hovered) : "";
  const alerts = detail.alerts ?? [];
  return <section className="mindmap-layout"><div className="panel mindmap-panel"><div className="panel-title"><div><span className="eyebrow">Relationship view / {detail.id}</span><h3>Evidence concept map</h3></div><span className="outline-chip">{alerts.length} intelligence alerts</span></div><ConceptMap detail={detail} setHovered={setHovered} />{detail.status === "Closed" && <div className="closed-notice">This case is closed. Reopen it before adding new records.</div>}<div className="mindmap-alerts"><div className="record-list-heading"><span>Intelligence alerts</span><small>Indicator only, not proof</small></div><IntelligenceAlertFeed alerts={alerts} compact /></div><AddEvidenceForm caseId={detail.id} onSaved={onSaved} disabled={detail.status === "Closed"} /><EvidenceRecords items={detail.evidence} onDelete={onDelete} /><p className="graph-note">New evidence records are connected to this case automatically. Relationships and alerts require source review before any decision.</p></div><aside className={"evidence-preview panel " + (hovered ? "visible" : "")}><span className="eyebrow">Evidence preview</span>{hovered ? <><div className="preview-image">{previewImage ? <img src={previewImage} alt={hovered.title} /> : <><span>{hovered.type === "cctv" ? "CCTV" : "SOURCE"}</span><b>{hovered.id}</b></>}</div><h3>{hovered.title}</h3><p>{hovered.notes}</p><small>Source: {hovered.source}</small></> : <div className="preview-empty">Hover over an evidence label to inspect its source context.</div>}</aside></section>;
}

function ConceptMap({ detail, setHovered }: { detail: Detail; setHovered: (e: Evidence | null) => void }) {
  const [zoom, setZoom] = useState(1);
  const graph = detail.graph ?? { nodes: [], edges: [] }; const center = { x: 50, y: 86 }; const outer = graph.nodes.filter((node) => node.group !== "case");
  const positions = new Map<string, { x: number; y: number }>(); positions.set(`CASE:${detail.id}`, center);
  outer.forEach((node, index) => { const columns = Math.min(4, Math.max(2, outer.length)); const column = index % columns; const row = Math.floor(index / columns); positions.set(node.id, { x: 12 + column * (76 / (columns - 1)), y: 20 + row * 25 }); });
  return <div className="concept-map"><div className="map-controls" aria-label="Mind map zoom controls"><button type="button" aria-label="Zoom out" title="Zoom out" onClick={() => setZoom((value) => Math.max(.7, Number((value - .1).toFixed(1))))}>-</button><span>{Math.round(zoom * 100)}%</span><button type="button" aria-label="Zoom in" title="Zoom in" onClick={() => setZoom((value) => Math.min(1.4, Number((value + .1).toFixed(1))))}>+</button><button type="button" aria-label="Reset zoom" title="Reset zoom" onClick={() => setZoom(1)}>Reset</button></div><div className="concept-stage" style={{ transform: `scale(${zoom})` }}><svg className="concept-lines" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">{graph.edges.map((edge, index) => { const from = positions.get(edge.source), to = positions.get(edge.target); return from && to ? <line key={`${edge.source}-${edge.target}-${index}`} x1={from.x} y1={from.y} x2={to.x} y2={to.y} /> : null; })}</svg><div className="concept-world">{graph.nodes.map((node) => { const position = positions.get(node.id); if (!position) return null; const evidence = detail.evidence.find((item) => item.id === node.id) ?? null; return <button className={`concept-node ${node.group === "case" ? "case-node" : ""}`} key={node.id} style={{ left: `${position.x}%`, top: `${position.y}%` }} onMouseEnter={() => setHovered(evidence)} onMouseLeave={() => setHovered(null)} onClick={() => setHovered(evidence)}>{node.label}</button>; })}</div></div></div>;
}

function AddEventForm({ caseId, onSaved, disabled }: { caseId: string; onSaved: () => Promise<void>; disabled: boolean }) {
  const [open, setOpen] = useState(false); const [label, setLabel] = useState(""); const [location, setLocation] = useState(""); const [time, setTime] = useState(""); const [message, setMessage] = useState(""); const [saving, setSaving] = useState(false);
  async function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); if (!label.trim()) return; setSaving(true); setMessage(""); try { const response = await fetch(`${API_BASE}/api/case/${caseId}/events`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ label, location, time: time || null }) }); const data = await response.json(); if (!response.ok) throw Error(data.detail || "Event could not be added"); setLabel(""); setLocation(""); setTime(""); setMessage("Event saved. Timeline and case map updated."); await onSaved(); } catch (reason: any) { setMessage(reason.message); } finally { setSaving(false); } }
  return <div className="record-editor"><button type="button" className="editor-toggle" disabled={disabled} onClick={() => setOpen((value) => !value)}>{disabled ? "Case closed" : open ? "Close event form" : "+ Add timeline event"}</button>{open && !disabled && <form className="record-form" onSubmit={submit}><div className="form-grid"><label>Event<input autoFocus required placeholder="e.g. Witness statement recorded" value={label} onChange={(event) => setLabel(event.target.value)} /></label><label>Location<input placeholder="e.g. Jubilee Hills" value={location} onChange={(event) => setLocation(event.target.value)} /></label><label>Time<input type="datetime-local" value={time} onChange={(event) => setTime(event.target.value)} /></label></div><button className="save-record" type="submit" disabled={saving}>{saving ? "Saving..." : "Save event"}</button>{message && <span className="form-message">{message}</span>}</form>}</div>;
}

function AddEvidenceForm({ caseId, onSaved, disabled }: { caseId: string; onSaved: () => Promise<void>; disabled: boolean }) {
  const [open, setOpen] = useState(false); const [title, setTitle] = useState(""); const [type, setType] = useState("source"); const [source, setSource] = useState(""); const [notes, setNotes] = useState(""); const [imageFile, setImageFile] = useState<File | null>(null); const [message, setMessage] = useState(""); const [saving, setSaving] = useState(false); const imageRef = useRef<HTMLInputElement>(null);
  function chooseImage(file?: File) { if (!file) return; if (!file.type.startsWith("image/")) { setMessage("Choose an image file such as JPG or PNG."); return; } setImageFile(file); setMessage(""); }
  async function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); if (!title.trim()) return; setSaving(true); setMessage(""); try { let response: Response; if (imageFile) { const form = new FormData(); form.append("title", title); form.append("type", "image"); form.append("source", source || "image upload"); form.append("notes", notes); form.append("image", imageFile); response = await fetch(`${API_BASE}/api/case/${caseId}/evidence/image`, { method: "POST", body: form }); } else { response = await fetch(`${API_BASE}/api/case/${caseId}/evidence`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title, type, source: source || "investigator entry", notes }) }); } const data = await response.json(); if (!response.ok) throw Error(data.detail || "Evidence could not be added"); setTitle(""); setSource(""); setNotes(""); setImageFile(null); if (imageRef.current) imageRef.current.value = ""; setMessage(imageFile ? "Image evidence saved. The map updated with a new case connection." : "Evidence saved. The map updated with a new case connection."); await onSaved(); } catch (reason: any) { setMessage(reason.message); } finally { setSaving(false); } }
  return <div className="record-editor"><button type="button" className="editor-toggle" disabled={disabled} onClick={() => setOpen((value) => !value)}>{disabled ? "Case closed" : open ? "Close evidence form" : "+ Add evidence record"}</button>{open && !disabled && <form className="record-form" onSubmit={submit}><div className="form-grid"><label>Evidence title<input autoFocus required placeholder="e.g. Warehouse camera still" value={title} onChange={(event) => setTitle(event.target.value)} /></label><label>Type<select value={type} onChange={(event) => setType(event.target.value)}><option value="source">Source</option><option value="image">Image</option><option value="statement">Statement</option><option value="call-log">Call log</option><option value="location">Location</option><option value="document">Document</option></select></label><label>Source reference<input placeholder="e.g. Camera 04 / frame 18" value={source} onChange={(event) => setSource(event.target.value)} /></label><label className="wide-field">Notes<textarea rows={2} placeholder="What should the investigator review?" value={notes} onChange={(event) => setNotes(event.target.value)} /></label></div><div className="image-drop" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); chooseImage(event.dataTransfer.files[0]); }}><span>{imageFile ? imageFile.name : "Drop an evidence image here"}</span><button type="button" onClick={() => imageRef.current?.click()}>Choose image</button><input ref={imageRef} type="file" accept="image/jpeg,image/png,image/webp,image/gif" hidden onChange={(event) => chooseImage(event.target.files?.[0])} /></div><button className="save-record" type="submit" disabled={saving}>{saving ? "Saving..." : imageFile ? "Save image evidence" : "Save evidence"}</button>{message && <span className="form-message">{message}</span>}</form>}</div>;
}

function EvidenceRecords({ items, onDelete }: { items: Evidence[]; onDelete: (kind: "evidence" | "event", id: string) => Promise<boolean> }) {
  const [message, setMessage] = useState(""); const [removing, setRemoving] = useState("");
  async function remove(id: string) { setRemoving(id); setMessage(""); try { if (await onDelete("evidence", id)) setMessage("Evidence record deleted and the case map updated."); } catch (reason: any) { setMessage(reason.message); } finally { setRemoving(""); } }
  return <div className="map-record-list"><div className="record-list-heading"><span>Evidence records</span><small>{items.length} linked to this case</small></div>{items.map((item) => { const imageUrl = evidenceImageUrl(item); return <div className="map-record" key={item.id}>{imageUrl ? <img className="record-thumb" src={imageUrl} alt="" /> : <span className="record-thumb record-thumb-fallback">{item.type.slice(0, 2).toUpperCase()}</span>}<div><strong>{item.title}</strong><small>{item.type} · {item.source}</small></div><button className="delete-record" disabled={removing === item.id} aria-label={`Delete ${item.title}`} onClick={() => remove(item.id)}>{removing === item.id ? "Deleting..." : "Delete"}</button></div>; })}{!items.length && <div className="empty-records">No evidence records yet.</div>}{message && <div className="record-status">{message}</div>}</div>;
}
function Cctv({ evidence, onDrop, onFile, onAnalyze, onBrowse, inputRef, uploading, message, file }: {
  evidence: Evidence[];
  onDrop: (event: DragEvent<HTMLDivElement>) => void;
  onFile: (event: ChangeEvent<HTMLInputElement>) => void;
  onAnalyze: () => void;
  onBrowse: () => void;
  inputRef: RefObject<HTMLInputElement>;
  uploading: boolean;
  message: string;
  file: File | null;
}) {
  const clips = evidence.filter((item) => item.type === "cctv" && (item.metadata?.frames?.length ?? 0) > 0);
  const latest = clips[clips.length - 1];
  const [activeId, setActiveId] = useState(latest?.id ?? "");
  const selected = clips.find((item) => item.id === activeId) ?? latest;
  const frames = selected?.metadata?.frames ?? [];
  const [preview, setPreview] = useState<CctvFrame | null>(null);
  const previewFrame = preview && frames.some((frame) => frame.path === preview.path) ? preview : frames[0] ?? null;

  useEffect(() => {
    if (latest?.id) {
      setActiveId(latest.id);
      setPreview(null);
    }
  }, [latest?.id]);

  return (
    <section className="panel intake-panel">
      <div className="panel-title">
        <div>
          <span className="eyebrow">Evidence intake</span>
          <h3>Inspect a CCTV clip</h3>
          <p className="muted">Choose a clip, then analyze it for sampled frames and metadata.</p>
        </div>
        <span className="outline-chip">OpenCV workflow</span>
      </div>
      <div className="drop-zone" onDragOver={(event) => event.preventDefault()} onDrop={onDrop}>
        <div className="upload-icon">^</div>
        <h3>{file ? file.name : "Drop CCTV video here"}</h3>
        <p>MP4, MOV, AVI - maximum 100 MB</p>
        <button onClick={onBrowse} disabled={uploading}>{file ? "Change video" : "Choose video"}</button>
        <input ref={inputRef} type="file" accept="video/*" onChange={onFile} hidden />
        {file && <button onClick={onAnalyze} disabled={uploading}>{uploading ? "Analyzing..." : "Analyze video"}</button>}
      </div>
      {message && <div className="upload-message">{message}</div>}
      {clips.length > 1 && (
        <div className="clip-switcher">
          {clips.map((clip) => (
            <button key={clip.id} className={clip.id === selected?.id ? "selected" : ""} onClick={() => { setActiveId(clip.id); setPreview(null); }}>
              {clip.id} · {clip.metadata?.frames?.length ?? 0} frames
            </button>
          ))}
        </div>
      )}
      {selected && frames.length > 0 && (
        <div className="frame-review">
          <div className="record-list-heading">
            <span>Sampled frames</span>
            <small>{selected.id} · {frames.length} stills for human review</small>
          </div>
          {previewFrame && (
            <figure className="frame-hero">
              <img src={storageUrl(previewFrame.path)} alt={`Sampled frame at ${previewFrame.timestamp_seconds}s`} />
              <figcaption>
                {previewFrame.timestamp_seconds.toFixed(1)}s
                {previewFrame.face_detected === true ? " · Face candidate" : previewFrame.face_detected === false ? " · No face detected" : ""}
              </figcaption>
            </figure>
          )}
          <div className="frame-grid">
            {frames.map((frame) => {
              const src = storageUrl(frame.path);
              if (!src) return null;
              return (
                <button type="button" className={`frame-card ${previewFrame?.path === frame.path ? "selected" : ""}`} key={frame.path} onClick={() => setPreview(frame)}>
                  <img src={src} alt={`Frame at ${frame.timestamp_seconds}s`} />
                  <span>{frame.timestamp_seconds.toFixed(1)}s</span>
                </button>
              );
            })}
          </div>
        </div>
      )}
      <div className="guardrail">
        <strong>Human review boundary</strong>
        <span>The original video stays as a file reference. Sampled frames and metadata are reviewable evidence, not an automated conclusion.</span>
      </div>
    </section>
  );
}
function Screening({ result, onDrop, onFile, onAnalyze, onBrowse, inputRef, uploading, message, file }: any) { return <section className="panel intake-panel"><div className="panel-title"><div><span className="eyebrow">Spreadsheet intake</span><h3>Narrow a suspect review queue</h3><p className="muted">Choose a workbook, then analyze the rows for this case.</p></div><span className="outline-chip">{result ? `${result.input_rows} -> ${result.shortlist_size}` : "Excel workflow"}</span></div><div className="drop-zone" onDragOver={(event) => event.preventDefault()} onDrop={onDrop}><div className="upload-icon">[]</div><h3>{file ? file.name : "Drop suspect workbook here"}</h3><p>.xlsx or .xlsm - use the sample workbook structure</p><button onClick={onBrowse} disabled={uploading}>{file ? "Change workbook" : "Choose workbook"}</button><input ref={inputRef} type="file" accept=".xlsx,.xlsm" onChange={onFile} hidden />{file && <button onClick={onAnalyze} disabled={uploading}>{uploading ? "Analyzing..." : "Analyze workbook"}</button>}</div>{message && <div className="upload-message">{message}</div>}{result && <div className="screening-result"><div className="result-summary"><strong>{result.shortlist_size} candidates queued</strong><span>from {result.input_rows} input rows</span></div>{result.shortlist.slice(0, 8).map((item: any) => <div className="screen-row" key={item.suspect_id}><strong>{item.suspect_id}</strong><span>{item.name}</span><small>{item.area} · {item.route_match} route · {item.call_link_count} linked calls</small><b>{Math.round(item.score * 100)}%</b></div>)}</div>}<div className="guardrail"><strong>Human review boundary</strong><span>This is a ranked review queue based on declared spreadsheet signals. It does not identify a person or establish guilt.</span></div></section>; }
function Panel({ title, action, children }: any) { return <section className="panel"><div className="panel-title"><h3>{title}</h3><span className="eyebrow">{action}</span></div>{children}</section>; }
function Metric({ label, value }: { label: string; value: number }) { return <div className="metric panel"><span>{label}</span><strong>{value.toString().padStart(2, "0")}</strong></div>; }
