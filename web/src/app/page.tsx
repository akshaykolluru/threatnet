"use client";

import { ChangeEvent, DragEvent, FormEvent, RefObject, useEffect, useRef, useState } from "react";

// Uvicorn's local dev server binds to IPv4 by default.  Using the explicit
// loopback address avoids Windows browsers resolving localhost to ::1.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
type View = "overview" | "timeline" | "mindmap" | "cctv" | "screening";
type CaseItem = { id: string; title: string; status: string; priority: string; summary: string };
type CctvFrame = { path: string; frame_index: number; timestamp_seconds: number; face_candidates?: number; face_detected?: boolean };
type Evidence = { id: string; case_id?: string; title: string; type: string; notes: string; confidence?: number; source: string; metadata?: { frames?: CctvFrame[]; duration_seconds?: number; fps?: number } };
type FrameSearchResult = { frame_path: string; timestamp_seconds: number; score: number; reason: string; evidence_id?: string };
type GraphNode = { id: string; label: string; group: string };
type GraphEdge = { source: string; target: string; label: string };
type TimelineEvent = { id: string; time: string; label: string; location: string; entity_id?: string | null; kind?: string };
type IntelligenceAlert = { id: string; type: string; severity: string; title: string; summary: string; reasoning_trace?: string; created_at?: string | null };
type Detail = CaseItem & { evidence: Evidence[]; events: TimelineEvent[]; ranking: any[]; contradictions: any[]; audit: any[]; alerts?: IntelligenceAlert[]; face_matches?: any[]; graph?: { nodes: GraphNode[]; edges: GraphEdge[] } };
type CaseIntake = { title: string; incidentType: string; priority: "Low" | "Medium" | "High"; incidentAt: string; location: string; description: string; reportedBy: string; witnesses: string; witnessContact: string; suspectInfo: string; notes: string };

const emptyCaseIntake = (): CaseIntake => ({ title: "", incidentType: "Suspicious activity", priority: "High", incidentAt: "", location: "", description: "", reportedBy: "", witnesses: "", witnessContact: "", suspectInfo: "", notes: "" });

function localDateTimeValue(date = new Date()) {
  const timezoneOffset = date.getTimezoneOffset() * 60_000;
  return new Date(date.getTime() - timezoneOffset).toISOString().slice(0, 16);
}

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
  const [caseIntake, setCaseIntake] = useState<CaseIntake>(emptyCaseIntake);
  const [intakeError, setIntakeError] = useState("");
  const [savingCase, setSavingCase] = useState(false);
  const [toastMessage, setToastMessage] = useState("");
  const [intakeLoggedAt, setIntakeLoggedAt] = useState("");
  const [caseMessage, setCaseMessage] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const sheetRef = useRef<HTMLInputElement>(null);
  const caseTitleRef = useRef<HTMLInputElement>(null);

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

  const nextCaseId = `CASE-${Math.max(100, ...cases.map((item) => Number(item.id.replace("CASE-", "")) || 0)) + 1}`;

  function openCaseIntake() {
    const openedAt = new Date();
    setCaseIntake({ ...emptyCaseIntake(), incidentAt: localDateTimeValue(openedAt) });
    setIntakeLoggedAt(openedAt.toLocaleString());
    setIntakeError("");
    setShowAddCase(true);
  }

  function closeCaseIntake() {
    if (savingCase) return;
    setShowAddCase(false);
    setIntakeError("");
  }

  function updateIntake<K extends keyof CaseIntake>(field: K, value: CaseIntake[K]) {
    setCaseIntake((current) => ({ ...current, [field]: value }));
  }

  async function addCase(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const requiredFields = [caseIntake.title, caseIntake.incidentType, caseIntake.incidentAt, caseIntake.location, caseIntake.description, caseIntake.reportedBy];
    if (requiredFields.some((value) => !value.trim())) {
      setIntakeError("Please fill in all mandatory intelligence fields marked with an asterisk (*).");
      return;
    }
    const supportingDetails = [
      `Incident type: ${caseIntake.incidentType}`,
      `Incident time: ${new Date(caseIntake.incidentAt).toLocaleString()}`,
      `Location: ${caseIntake.location.trim()}`,
      `Reported by: ${caseIntake.reportedBy.trim()}`,
      caseIntake.witnesses.trim() && `Witnesses: ${caseIntake.witnesses.trim()}`,
      caseIntake.witnessContact.trim() && `Secure contact: ${caseIntake.witnessContact.trim()}`,
      caseIntake.suspectInfo.trim() && `Descriptors: ${caseIntake.suspectInfo.trim()}`,
      caseIntake.notes.trim() && `Notes: ${caseIntake.notes.trim()}`,
    ].filter(Boolean).join("\n");
    setSavingCase(true);
    setIntakeError("");
    try {
      const response = await fetch(`${API_BASE}/api/cases`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: caseIntake.title.trim(), priority: caseIntake.priority, summary: `${caseIntake.description.trim()}\n\n${supportingDetails}` }) });
      const created = await response.json();
      if (!response.ok) throw Error(created.detail || "Case could not be created");
      setCases((items) => [...items, created]);
      setActive(created.id);
      setView("overview");
      setShowAddCase(false);
      setToastMessage(`${created.id} registered successfully to the workspace.`);
    } catch (reason: any) {
      setIntakeError(reason.message || "Case could not be created. Please try again.");
    } finally { setSavingCase(false); }
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
  useEffect(() => {
    if (!showAddCase) return;
    document.body.style.overflow = "hidden";
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === "Escape") closeCaseIntake(); };
    document.addEventListener("keydown", onKeyDown);
    window.setTimeout(() => caseTitleRef.current?.focus(), 0);
    return () => { document.body.style.overflow = ""; document.removeEventListener("keydown", onKeyDown); };
  }, [showAddCase, savingCase]);
  useEffect(() => {
    if (!toastMessage) return;
    const timer = window.setTimeout(() => setToastMessage(""), 3800);
    return () => window.clearTimeout(timer);
  }, [toastMessage]);

  async function uploadVideo(file?: File, query = "", referenceImage: File | null = null) {
    if (!file || !active) return;
    setUploading(true); setUploadMessage("Analyzing video and sampling frames...");
    const form = new FormData(); form.append("file", file); form.append("query", query); if (referenceImage) form.append("reference_image", referenceImage);
    try {
      const response = await fetch(`${API_BASE}/api/case/${active}/cctv/inspect`, { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw Error(data.detail || "CCTV inspection failed");
      setUploadMessage(`Indexed ${data.metadata.frames.length} frames and attached ${data.evidence_id}.${data.results?.length ? ` Found ${data.results.length} matching frame candidates.` : ""}`);
      await loadDetail(active);
      return data;
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
    <section className="workspace-grid"><aside className="case-rail panel"><div className="rail-heading"><div><span className="eyebrow">Workspace</span><h2>Cases</h2></div><button className="count" type="button" onClick={openCaseIntake}>+ Add case</button></div><div className="case-list">{cases.map((item) => <button className={`case-card ${active === item.id ? "active" : ""}`} key={item.id} onClick={() => { setActive(item.id); setHovered(null); }}><span className={`priority ${item.priority.toLowerCase()}`} /><span><strong>{item.id}</strong><b>{item.title}</b><small>{item.status} · {item.priority} priority</small></span></button>)}{!cases.length && !error && <div className="empty-state">Loading cases...</div>}{error && <div className="error-state">{error}<button onClick={() => loadCases()}>Retry</button></div>}</div></aside>
      <div className="content-column">{detail && <><div className="case-heading"><div><span className="eyebrow">Active investigation / {detail.id}</span><h2>{detail.title}</h2><p className="muted">{detail.summary}</p></div><div className="case-actions"><span className="status-chip">● {detail.status}</span><button className="status-action" onClick={toggleCaseStatus}>{detail.status === "Closed" ? "Reopen case" : "Close case"}</button><button className="delete-case" onClick={deleteCase}>Delete case</button>{caseMessage && <small className="case-message">{caseMessage}</small>}</div></div><nav className="view-nav" aria-label="Case views">{([['overview', 'Command view'], ['timeline', 'Timeline'], ['mindmap', 'Mind map'], ['cctv', 'CCTV intake'], ['screening', 'Suspect screening']] as [View, string][]).map(([key, label]) => <button key={key} className={view === key ? "selected" : ""} onClick={() => setView(key)}>{label}</button>)}</nav>{loading ? <div className="panel empty-state">Loading case intelligence...</div> : view === "overview" ? <Overview detail={detail} setView={setView} /> : view === "timeline" ? <Timeline detail={detail} onSaved={() => loadDetail(active)} onDelete={deleteRecord} /> : view === "mindmap" ? <MindMap detail={detail} hovered={hovered} setHovered={setHovered} onSaved={() => loadDetail(active)} onDelete={deleteRecord} /> : view === "cctv" ? <Cctv evidence={detail.evidence} file={pendingKind === "video" ? pendingFile : null} onDrop={(event: DragEvent<HTMLDivElement>) => handleDrop(event, "video")} onFile={(event: ChangeEvent<HTMLInputElement>) => chooseFile(event, "video")} onAnalyze={(query, referenceImage) => uploadVideo(pendingFile ?? undefined, query, referenceImage)} onBrowse={() => fileRef.current?.click()} inputRef={fileRef} uploading={uploading} message={uploadMessage} /> : <Screening result={screening} file={pendingKind === "sheet" ? pendingFile : null} onDrop={(event: DragEvent<HTMLDivElement>) => handleDrop(event, "sheet")} onFile={(event: ChangeEvent<HTMLInputElement>) => chooseFile(event, "sheet")} onAnalyze={() => uploadSheet(pendingFile ?? undefined)} onBrowse={() => sheetRef.current?.click()} inputRef={sheetRef} uploading={uploading} message={uploadMessage} />}</>}</div>
    </section>
    {showAddCase && <CaseIntakeDialog intake={caseIntake} loggedAt={intakeLoggedAt} nextCaseId={nextCaseId} error={intakeError} saving={savingCase} titleRef={caseTitleRef} onChange={updateIntake} onClose={closeCaseIntake} onSubmit={addCase} />}
    <div className={`case-toast ${toastMessage ? "visible" : ""}`} role="status" aria-live="polite"><span />{toastMessage}</div>
  </div></main>;
}

function CaseIntakeDialog({ intake, loggedAt, nextCaseId, error, saving, titleRef, onChange, onClose, onSubmit }: { intake: CaseIntake; loggedAt: string; nextCaseId: string; error: string; saving: boolean; titleRef: RefObject<HTMLInputElement>; onChange: <K extends keyof CaseIntake>(field: K, value: CaseIntake[K]) => void; onClose: () => void; onSubmit: (event: FormEvent<HTMLFormElement>) => void }) {
  return <div className="case-dialog-layer" role="presentation"><button className="case-dialog-backdrop" type="button" aria-label="Close case intake" onClick={onClose} /><section className="case-dialog" role="dialog" aria-modal="true" aria-labelledby="case-intake-title">
    <header className="case-dialog-header"><div><div className="dialog-kicker"><span>Case file generator</span><small>ThreatNet dossier intake</small></div><h2 id="case-intake-title">Create new investigation case</h2></div><button className="dialog-close" type="button" aria-label="Close dialog" onClick={onClose} disabled={saving}>×</button></header>
    {error && <div className="dialog-alert" role="alert">{error}</div>}
    <form className="case-intake-form" onSubmit={onSubmit}>
      <IntakeSection number="01" title="Case information"><div className="intake-grid"><label className="span-8">Case title <Required /><input ref={titleRef} required value={intake.title} placeholder="e.g. Unsanctioned perimeter breach at Sector 4" onChange={(event) => onChange("title", event.target.value)} /></label><label className="span-4">Case ID <small>System generated</small><input readOnly value={nextCaseId} aria-label="System generated case ID" /></label><label className="span-4">Incident type <Required /><select value={intake.incidentType} onChange={(event) => onChange("incidentType", event.target.value)}><option>Suspicious activity</option><option>Perimeter breach</option><option>Theft / larceny</option><option>Missing person</option><option>Cyber intrusion</option><option>Vehicular anomaly</option><option>Other intelligence</option></select></label><label className="span-4">Severity <Required /><select value={intake.priority} onChange={(event) => onChange("priority", event.target.value as CaseIntake["priority"])}><option value="High">High</option><option value="Medium">Medium</option><option value="Low">Low</option></select></label><label className="span-4">Initial status<select disabled value="Open"><option>Open</option></select><small>New cases open automatically.</small></label></div></IntakeSection>
      <IntakeSection number="02" title="Incident details"><div className="intake-grid"><label className="span-6">Incident date & time <Required /><input required type="datetime-local" value={intake.incidentAt} onChange={(event) => onChange("incidentAt", event.target.value)} /></label><label className="span-6">Incident location <Required /><input required value={intake.location} placeholder="e.g. Jubilee Hills Checkpost, Gate 3" onChange={(event) => onChange("location", event.target.value)} /></label><label className="span-12">Case description / initial narrative <Required /><textarea required rows={3} value={intake.description} placeholder="Outline initial observations, sensor anomalies, vehicle IDs, or immediate operational flags..." onChange={(event) => onChange("description", event.target.value)} /></label></div></IntakeSection>
      <IntakeSection number="03" title="People involved"><div className="intake-grid"><label className="span-4">Case reported by <Required /><input required value={intake.reportedBy} placeholder="Officer name & badge ID" onChange={(event) => onChange("reportedBy", event.target.value)} /></label><label className="span-4">Witness name(s)<input value={intake.witnesses} placeholder="Key witnesses, comma separated" onChange={(event) => onChange("witnesses", event.target.value)} /></label><label className="span-4">Witness contact / secure line<input value={intake.witnessContact} placeholder="Phone, frequency, or department" onChange={(event) => onChange("witnessContact", event.target.value)} /></label><label className="span-12">Suspect information & vehicle descriptors<textarea rows={2} value={intake.suspectInfo} placeholder="Aliases, clothing or features, known associates, license plates..." onChange={(event) => onChange("suspectInfo", event.target.value)} /></label></div></IntakeSection>
      <IntakeSection number="04" title="Additional information"><div className="intake-grid"><label className="span-6">Intake logged at<input readOnly value={loggedAt} aria-label="Current intake time" /></label><label className="span-12">Additional notes & chain of custody flags<textarea rows={2} value={intake.notes} placeholder="Optional security caveat, evidence vault locker ID, or jurisdiction handoff notes..." onChange={(event) => onChange("notes", event.target.value)} /></label></div></IntakeSection>
      <footer className="case-dialog-footer"><button type="button" className="dialog-cancel" onClick={onClose} disabled={saving}>Cancel</button><button type="submit" className="dialog-submit" disabled={saving}>{saving ? "Creating case file..." : "Create case file"}</button></footer>
    </form>
  </section></div>;
}

function IntakeSection({ number, title, children }: { number: string; title: string; children: React.ReactNode }) { return <section className="intake-section"><h3><span>{number}.</span>{title}</h3>{children}</section>; }
function Required() { return <b className="required" aria-label="required">*</b>; }

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
  onAnalyze: (query: string, referenceImage: File | null) => Promise<any>;
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
  const [results, setResults] = useState<FrameSearchResult[]>([]);
  const [faceResults, setFaceResults] = useState<FrameSearchResult[]>([]);
  const [searchMessage, setSearchMessage] = useState("");
  const [query, setQuery] = useState("");
  const [referenceImage, setReferenceImage] = useState<File | null>(null);
  const referenceImageRef = useRef<HTMLInputElement>(null);
  const previewFrame = preview && frames.some((frame) => frame.path === preview.path) ? preview : frames[0] ?? null;

  useEffect(() => {
    if (latest?.id) {
      setActiveId(latest.id);
      setPreview(null);
    }
  }, [latest?.id]);

  async function searchSelectedClip() {
    if (!selected?.case_id || (!query.trim() && !referenceImage)) return;
    setSearchMessage("Searching indexed frames...");
    const form = new FormData(); form.append("evidence_id", selected.id); form.append("query", query); if (referenceImage) form.append("reference_image", referenceImage);
    const response = await fetch(`${API_BASE}/api/case/${selected.case_id}/cctv/search`, { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw Error(data.detail || "Frame search failed");
    setResults(data.results ?? []);
    setFaceResults(data.face_results ?? []);
    setSearchMessage(data.results?.length ? `Found ${data.results.length} review candidates.` : "No relevant frames were found for this search.");
  }

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
        {file && <button onClick={async () => { const response = await onAnalyze(query, referenceImage); setResults(response?.results ?? []); setFaceResults(response?.face_results ?? []); }} disabled={uploading}>{uploading ? "Analyzing..." : "Analyze video"}</button>}
      </div>
      <div className="cctv-query">
        <label>What do you want to search for?
          <textarea value={query} onChange={(event) => setQuery(event.target.value)} placeholder="e.g. Find a person carrying a backpack" rows={3} />
        </label>
        <div className="image-drop">
          <span>{referenceImage ? referenceImage.name : "Optional: add a person, vehicle, object, or scene reference"}</span>
          <button type="button" onClick={() => referenceImageRef.current?.click()} disabled={uploading}>{referenceImage ? "Change image" : "Choose reference image"}</button>
          <input ref={referenceImageRef} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={(event) => setReferenceImage(event.target.files?.[0] ?? null)} />
        </div>
        {selected && <button type="button" className="search-clip" disabled={uploading || (!query.trim() && !referenceImage)} onClick={() => searchSelectedClip().catch((error) => setSearchMessage(error.message || "Frame search failed"))}>Search indexed clip</button>}
        <small>Text, image, and combined searches use the cached frame index. This visual search feature requires the optional OpenCLIP model.</small>
        {searchMessage && <div className="upload-message">{searchMessage}</div>}
      </div>
      {message && <div className="upload-message">{message}</div>}
      {results.length > 0 && <div className="frame-review"><div className="record-list-heading"><span>Strongest search candidates</span><small>Lower-ranked frames were suppressed</small></div><div className="frame-grid">{results.map((result) => <figure className="frame-card search-result" key={`${result.frame_path}-${result.timestamp_seconds}`}><img src={storageUrl(result.frame_path)} alt={`Search result at ${result.timestamp_seconds}s`} /><figcaption><b>{result.timestamp_seconds.toFixed(1)}s · rank {result.score.toFixed(3)}</b><span>{result.reason}</span></figcaption></figure>)}</div></div>}
      {faceResults.length > 0 && <div className="frame-review"><div className="record-list-heading"><span>Person-photo candidates</span><small>Compared against detected face regions</small></div><div className="frame-grid">{faceResults.map((result) => <figure className="frame-card search-result" key={`face-${result.frame_path}-${result.timestamp_seconds}`}><img src={storageUrl(result.frame_path)} alt={`Face candidate at ${result.timestamp_seconds}s`} /><figcaption><b>{result.timestamp_seconds.toFixed(1)}s · similarity {result.score.toFixed(3)}</b><span>{result.reason}</span></figcaption></figure>)}</div></div>}
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
