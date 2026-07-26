// src/pages/Tickets.jsx — with real escalation matrix from Employee_Hierarchy_RoleAccess.xlsx
import { useState } from "react";
import { useApi } from "../hooks/useApi";
import { ArrowUpRight, Plus, X, Users, Shield } from "lucide-react";
import { C } from "../utils/tokens";
import PageHeader from "../components/ui/PageHeader";
import Pill from "../components/ui/Pill";
import ProgressBar from "../components/ui/ProgressBar";
import DataPatterns from "../components/DataPatterns";

var priorityLabel = function(p) { return { P1: "Critical", P2: "High", P3: "Medium", P4: "Low" }[p] || p; };
var priorityTone = function(p) { return { P1: "red", P2: "amber", P3: "blue", P4: "slate" }[p] || "slate"; };
var formatTs = function(ts) { return ts ? ts.replace("T", " ").slice(0, 16) : ""; };
var minutesToHm = function(m) { if (m == null) return "—"; var h = Math.floor(m / 60); var mm = Math.round(m % 60); return h > 0 ? h + "h " + mm + "m" : mm + "m"; };
var btn = function(bg, color, border) { return { display: "inline-flex", alignItems: "center", gap: 6, padding: "8px 14px", borderRadius: 8, fontSize: 14, fontWeight: 500, cursor: "pointer", background: bg || C.white, color: color || C.t7, border: "1px solid " + (border || C.border) }; };
var thStyle = { textAlign: "left", fontWeight: 500, fontSize: 11, color: C.t6, textTransform: "uppercase", letterSpacing: 0.5, padding: "10px 16px", background: C.bg, borderBottom: "1px solid " + C.border };
var tdStyle = { padding: "12px 16px", borderBottom: "1px solid " + C.borderLight, fontSize: 14, color: C.t7 };
var inputStyle = { width: "100%", padding: "10px 12px", border: "1px solid " + C.border, borderRadius: 8, fontSize: 14, color: C.t7, background: C.bg, outline: "none", boxSizing: "border-box" };
var labelStyle = { fontSize: 13, fontWeight: 500, color: C.t6, marginBottom: 4, display: "block" };

// ════════════════════════════════════════════
// EMPLOYEE HIERARCHY (from Employee_Hierarchy_RoleAccess.xlsx)
// ════════════════════════════════════════════

var EMPLOYEES = {
  D001: { name: "Rajesh Nair", role: "IT Director", level: 0, dept: "IT Operations", spec: "General", exp: 15, email: "r.nair@innodatatics.com", reports_to: null },
  M001: { name: "Priya Sharma", role: "Senior Manager", level: 1, dept: "Network Ops", spec: "Network", exp: 12, email: "p.sharma@innodatatics.com", reports_to: "D001" },
  M002: { name: "Vikram Mehta", role: "Senior Manager", level: 1, dept: "Server Ops", spec: "Server", exp: 11, email: "v.mehta@innodatatics.com", reports_to: "D001" },
  TL001: { name: "Ananya Krishnan", role: "Team Lead", level: 2, dept: "Network Ops", spec: "Network", exp: 8, email: "a.krishnan@innodatatics.com", reports_to: "M001" },
  TL002: { name: "Siddharth Joshi", role: "Team Lead", level: 2, dept: "Network Ops", spec: "Network", exp: 7, email: "s.joshi@innodatatics.com", reports_to: "M001" },
  TL003: { name: "Deepika Rajan", role: "Team Lead", level: 2, dept: "Server Ops", spec: "Server", exp: 9, email: "d.rajan@innodatatics.com", reports_to: "M002" },
  TL004: { name: "Arjun Pillai", role: "Team Lead", level: 2, dept: "Server Ops", spec: "Server", exp: 8, email: "a.pillai@innodatatics.com", reports_to: "M002" },
  E001: { name: "Karthik Iyer", role: "Support Engineer", level: 3, dept: "Network Ops", spec: "Network", exp: 6, email: "k.iyer@innodatatics.com", reports_to: "TL001" },
  E002: { name: "Meera Pillai", role: "Support Engineer", level: 3, dept: "Network Ops", spec: "Network", exp: 7, email: "m.pillai@innodatatics.com", reports_to: "TL001" },
  E003: { name: "Rohan Gupta", role: "Junior Engineer", level: 3, dept: "Network Ops", spec: "Network", exp: 1, email: "r.gupta@innodatatics.com", reports_to: "TL001" },
  E004: { name: "Tushar Bhatt", role: "Support Engineer", level: 3, dept: "Server Ops", spec: "Server", exp: 6, email: "t.bhatt@innodatatics.com", reports_to: "TL003" },
  E005: { name: "Aisha Kapoor", role: "Support Engineer", level: 3, dept: "Network Ops", spec: "Network", exp: 6, email: "a.kapoor@innodatatics.com", reports_to: "TL001" },
  E006: { name: "Nikhil Verma", role: "Support Engineer", level: 3, dept: "Network Ops", spec: "Network", exp: 4, email: "n.verma@innodatatics.com", reports_to: "TL002" },
  E007: { name: "Pritha Das", role: "Support Engineer", level: 3, dept: "Network Ops", spec: "Network", exp: 6, email: "p.das@innodatatics.com", reports_to: "TL002" },
  E008: { name: "Sunita Rao", role: "Senior Engineer", level: 3, dept: "Server Ops", spec: "Server", exp: 10, email: "s.rao@innodatatics.com", reports_to: "TL003" },
  E009: { name: "Suresh Nambiar", role: "Support Engineer", level: 3, dept: "Network Ops", spec: "Network", exp: 4, email: "s.nambiar@innodatatics.com", reports_to: "TL002" },
  E010: { name: "Lavanya Menon", role: "Support Engineer", level: 3, dept: "Network Ops", spec: "Network", exp: 3, email: "l.menon@innodatatics.com", reports_to: "TL002" },
  E011: { name: "Mihir Shah", role: "Junior Engineer", level: 3, dept: "Server Ops", spec: "Server", exp: 1, email: "m.shah@innodatatics.com", reports_to: "TL003" },
  E012: { name: "Ravi Chandran", role: "Support Engineer", level: 3, dept: "Network Ops", spec: "Network", exp: 6, email: "r.chandran@innodatatics.com", reports_to: "TL001" },
  E013: { name: "Rekha Nair", role: "Support Engineer", level: 3, dept: "Server Ops", spec: "Server", exp: 4, email: "r.nair2@innodatatics.com", reports_to: "TL003" },
  E014: { name: "Varun Tiwari", role: "Support Engineer", level: 3, dept: "Server Ops", spec: "Server", exp: 3, email: "v.tiwari@innodatatics.com", reports_to: "TL004" },
  E015: { name: "Divya Subramanian", role: "Support Engineer", level: 3, dept: "Network Ops", spec: "Network", exp: 7, email: "d.subramanian@innodatatics.com", reports_to: "TL002" },
  E016: { name: "Chandan Mishra", role: "Senior Engineer", level: 3, dept: "Network Ops", spec: "Network", exp: 8, email: "c.mishra@innodatatics.com", reports_to: "TL001" },
  E017: { name: "Tanvi Kulkarni", role: "Support Engineer", level: 3, dept: "Network Ops", spec: "Network", exp: 4, email: "t.kulkarni@innodatatics.com", reports_to: "TL002" },
  E018: { name: "Sameer Desai", role: "Support Engineer", level: 3, dept: "Server Ops", spec: "Server", exp: 6, email: "s.desai@innodatatics.com", reports_to: "TL004" },
  E019: { name: "Pooja Agarwal", role: "Support Engineer", level: 3, dept: "Server Ops", spec: "Server", exp: 4, email: "p.agarwal@innodatatics.com", reports_to: "TL004" },
  E020: { name: "Ishaan Bhose", role: "Junior Engineer", level: 3, dept: "Network Ops", spec: "Network", exp: 1, email: "i.bhose@innodatatics.com", reports_to: "TL002" },
};

// SLA targets from SLA_RULES (Gold tier used as default)
var SLA_TARGETS = {
  P1: { response: 60, resolution: 240, escalate_pcts: [50, 75, 90, 100] },
  P2: { response: 120, resolution: 480, escalate_pcts: [50, 75, 90, 100] },
  P3: { response: 240, resolution: 1440, escalate_pcts: [75, 90, 100] },
  P4: { response: 480, resolution: 2880, escalate_pcts: [90, 100] },
};

// Escalation rules from Escalation_Matrix sheet
var ESCALATION_RULES = [
  { id: "ESC01", trigger: "Breach prob >= 0.70 at creation", type: "AI-Triggered", notify: ["Team Lead", "Senior Manager"], action: "Reassign to exp>=5yr" },
  { id: "ESC02", trigger: "50% SLA window elapsed, P1", type: "Time-Triggered", notify: ["Assigned Engineer", "Team Lead"], action: "Acknowledge + update status" },
  { id: "ESC03", trigger: "75% SLA window elapsed, P1-P2", type: "Time-Triggered", notify: ["Team Lead", "Senior Manager"], action: "Reassign or escalate" },
  { id: "ESC04", trigger: "90% SLA window elapsed, any", type: "Time-Triggered", notify: ["Senior Manager", "IT Director"], action: "Emergency override" },
  { id: "ESC05", trigger: "SLA breached — Gold client", type: "Breach-Triggered", notify: ["Senior Manager", "IT Director", "Client Acct Mgr"], action: "Post-mortem + client notify" },
  { id: "ESC06", trigger: "SLA breached — Silver/Bronze", type: "Breach-Triggered", notify: ["Team Lead", "Senior Manager"], action: "RCA within 24hr" },
  { id: "ESC07", trigger: "Ticket reopened after close", type: "Quality-Triggered", notify: ["Team Lead"], action: "Root cause analysis" },
  { id: "ESC08", trigger: "Engineer at capacity (3+ tickets)", type: "Workload-Triggered", notify: ["Team Lead"], action: "Load balance assignment" },
  { id: "ESC09", trigger: "Junior eng assigned P1 ticket", type: "AI-Triggered", notify: ["Team Lead"], action: "Auto-escalate to senior" },
  { id: "ESC10", trigger: "Breach prob >= 0.40 (Medium)", type: "AI-Triggered", notify: ["Team Lead"], action: "Flag for monitoring" },
  { id: "ESC11", trigger: "No response in 50% response window", type: "Time-Triggered", notify: ["Team Lead", "Senior Manager"], action: "Manual response required" },
  { id: "ESC12", trigger: "Multi-bounce (>=3 reassignments)", type: "Quality-Triggered", notify: ["Senior Manager", "IT Director"], action: "Manager override" },
];

// ════════════════════════════════════════════
// ESCALATION PATH BUILDER — uses real hierarchy
// ════════════════════════════════════════════

function getEscalationPath(priority, category) {
  var sla = SLA_TARGETS[priority] || SLA_TARGETS.P3;
  var isNetwork = category === "Network" || category === "Software" || category === "Security" || category === "Database";
  var path = [];

  // Pick best engineer by specialization and experience
  var candidates = [];
  Object.keys(EMPLOYEES).forEach(function(id) {
    var e = EMPLOYEES[id];
    if (e.level === 3 && (isNetwork ? e.dept === "Network Ops" : e.dept === "Server Ops")) {
      candidates.push({ id: id, name: e.name, role: e.role, exp: e.exp, reports_to: e.reports_to, email: e.email });
    }
  });
  candidates.sort(function(a, b) { return b.exp - a.exp; });

  // For P1: assign senior (exp>=5), skip juniors (ESC09)
  var assigned;
  if (priority === "P1") {
    assigned = candidates.find(function(c) { return c.exp >= 5 && c.role !== "Junior Engineer"; }) || candidates[0];
  } else if (priority === "P2") {
    assigned = candidates.find(function(c) { return c.exp >= 4; }) || candidates[0];
  } else {
    assigned = candidates[0] || { id: "E001", name: "Karthik Iyer", role: "Support Engineer", exp: 6, reports_to: "TL001", email: "" };
  }

  // Build chain: Engineer → Team Lead → Senior Manager → IT Director
  var tlId = assigned.reports_to;
  var tl = EMPLOYEES[tlId] || {};
  var mgrId = tl.reports_to;
  var mgr = EMPLOYEES[mgrId] || {};
  var dirId = mgr.reports_to || "D001";
  var dir = EMPLOYEES[dirId] || EMPLOYEES["D001"];

  // Step 1: Engineer assignment
  path.push({
    level: "L3 — Engineer Assignment",
    id: assigned.id, name: assigned.name, role: assigned.role,
    email: assigned.email, exp: assigned.exp,
    sla: sla.response + "min response",
    time: "0 min", active: true,
  });

  if (priority === "P1") {
    // P1: immediate TL notification + auto-escalate chain
    path.push({ level: "L2 — Team Lead Notification", id: tlId, name: tl.name || "Team Lead", role: tl.role || "Team Lead", email: tl.email || "", sla: "Immediate", time: "0 min (auto)", active: true, rule: "ESC01" });
    path.push({ level: "L1 — Senior Manager Alert", id: mgrId, name: mgr.name || "Sr Manager", role: mgr.role || "Senior Manager", email: mgr.email || "", sla: sla.resolution / 2 + "min", time: "+" + Math.round(sla.resolution * 0.5) + " min", active: true, rule: "ESC03" });
    path.push({ level: "L0 — IT Director Escalation", id: dirId, name: dir.name || "IT Director", role: dir.role || "IT Director", email: dir.email || "", sla: "Emergency", time: "+" + Math.round(sla.resolution * 0.9) + " min", active: true, rule: "ESC04" });
  } else if (priority === "P2") {
    path.push({ level: "L2 — Team Lead (if 50% elapsed)", id: tlId, name: tl.name || "Team Lead", role: tl.role || "Team Lead", email: tl.email || "", sla: sla.resolution / 2 + "min", time: "+" + Math.round(sla.resolution * 0.5) + " min", active: false, rule: "ESC02" });
    path.push({ level: "L1 — Senior Manager (if 75%)", id: mgrId, name: mgr.name || "Sr Manager", role: mgr.role || "Senior Manager", email: mgr.email || "", sla: Math.round(sla.resolution * 0.75) + "min", time: "+" + Math.round(sla.resolution * 0.75) + " min", active: false, rule: "ESC03" });
    path.push({ level: "L0 — IT Director (if 90%)", id: dirId, name: dir.name || "IT Director", role: dir.role || "IT Director", email: dir.email || "", sla: "Emergency", time: "+" + Math.round(sla.resolution * 0.9) + " min", active: false, rule: "ESC04" });
  } else if (priority === "P3") {
    path.push({ level: "L2 — Team Lead (if 75% elapsed)", id: tlId, name: tl.name || "Team Lead", role: tl.role || "Team Lead", email: tl.email || "", sla: Math.round(sla.resolution * 0.75) + "min", time: "+" + Math.round(sla.resolution * 0.75) + " min", active: false, rule: "ESC03" });
    path.push({ level: "L1 — Senior Manager (on breach)", id: mgrId, name: mgr.name || "Sr Manager", role: mgr.role || "Senior Manager", email: mgr.email || "", sla: "On breach", time: "+" + sla.resolution + " min", active: false, rule: "ESC06" });
  } else {
    path.push({ level: "L2 — Team Lead (on breach only)", id: tlId, name: tl.name || "Team Lead", role: tl.role || "Team Lead", email: tl.email || "", sla: "On breach", time: "+" + sla.resolution + " min", active: false, rule: "ESC06" });
  }

  return { path: path, sla: sla, assigned: assigned, teamLead: tl, manager: mgr, director: dir };
}

// ════════════════════════════════════════════
// ESCALATION PREVIEW (inside modal)
// ════════════════════════════════════════════

function EscalationPreview(props) {
  var esc = getEscalationPath(props.priority, props.category);
  var path = esc.path;
  var sla = esc.sla;

  function levelColor(lvl) {
    if (lvl.indexOf("L3") >= 0) return "#3B82F6";
    if (lvl.indexOf("L2") >= 0) return "#F59E0B";
    if (lvl.indexOf("L1") >= 0) return "#EF4444";
    if (lvl.indexOf("L0") >= 0) return "#DC2626";
    return C.t5;
  }

  return (
    <div style={{ border: "1px solid " + C.border, borderRadius: 10, overflow: "hidden", marginBottom: 16 }}>
      {/* Header */}
      <div style={{ background: props.priority === "P1" ? "#FEF2F2" : props.priority === "P2" ? "#FFFBEB" : C.bg, padding: "12px 16px", borderBottom: "1px solid " + C.border, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Shield size={14} color={props.priority === "P1" ? C.rose : props.priority === "P2" ? C.amber : C.brand} />
          <span style={{ fontSize: 13, fontWeight: 600, color: C.t9 }}>SLA Escalation Matrix</span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <Pill tone={priorityTone(props.priority)}>{priorityLabel(props.priority)}</Pill>
          <Pill tone="slate">{props.category}</Pill>
        </div>
      </div>

      {/* SLA targets */}
      <div style={{ padding: "10px 16px", background: C.bg, borderBottom: "1px solid " + C.borderLight, display: "flex", gap: 16 }}>
        <span style={{ fontSize: 11, color: C.t5 }}>Response: <strong style={{ color: C.t9 }}>{sla.response < 60 ? sla.response + "min" : sla.response / 60 + "h"}</strong></span>
        <span style={{ fontSize: 11, color: C.t5 }}>Resolution: <strong style={{ color: C.t9 }}>{sla.resolution < 60 ? sla.resolution + "min" : sla.resolution / 60 + "h"}</strong></span>
      </div>

      {/* Escalation path */}
      <div style={{ padding: "12px 16px" }}>
        {path.map(function(step, i) {
          return (
            <div key={i} style={{ display: "flex", gap: 12 }}>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: 24 }}>
                <div style={{ width: 10, height: 10, borderRadius: "50%", background: step.active ? levelColor(step.level) : C.borderLight, border: "2px solid " + (step.active ? levelColor(step.level) : C.border), flexShrink: 0, marginTop: 4 }} />
                {i < path.length - 1 && <div style={{ width: 2, flex: 1, background: C.borderLight, minHeight: 30 }} />}
              </div>
              <div style={{ flex: 1, paddingBottom: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: 13, fontWeight: 600, color: step.active ? C.t9 : C.t5 }}>{step.level}</span>
                  <span style={{ fontSize: 11, color: C.t4 }}>{step.time}</span>
                </div>
                <div style={{ fontSize: 12, color: C.t6, marginTop: 2 }}>
                  <span style={{ background: C.brandLight, padding: "1px 6px", borderRadius: 4, fontSize: 11, fontWeight: 500, color: C.brand, marginRight: 6 }}>{step.id}</span>
                  {step.name}
                  <span style={{ color: C.t4 }}>{" · " + step.role}</span>
                  {step.exp ? <span style={{ color: C.t4 }}>{" · " + step.exp + "yr exp"}</span> : null}
                </div>
                <div style={{ fontSize: 11, color: C.t4, marginTop: 2 }}>
                  {"SLA: " + step.sla}
                  {step.rule ? <span style={{ marginLeft: 8, color: C.brand, fontWeight: 500 }}>{step.rule}</span> : null}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Applicable rules */}
      <div style={{ padding: "10px 16px", background: C.brandLight, borderTop: "1px solid " + C.border }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: C.brand, marginBottom: 4 }}>Applicable Escalation Rules:</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {ESCALATION_RULES.filter(function(r) {
            if (props.priority === "P1") return true;
            if (props.priority === "P2") return ["ESC01", "ESC02", "ESC03", "ESC04", "ESC05", "ESC06", "ESC08", "ESC10", "ESC11", "ESC12"].indexOf(r.id) >= 0;
            return ["ESC03", "ESC04", "ESC06", "ESC07", "ESC08", "ESC10", "ESC12"].indexOf(r.id) >= 0;
          }).slice(0, 5).map(function(r) {
            return <span key={r.id} style={{ fontSize: 10, padding: "2px 6px", borderRadius: 4, background: C.white, border: "1px solid " + C.border, color: C.t6 }}>{r.id + ": " + r.trigger.substring(0, 35)}</span>;
          })}
        </div>
      </div>

      {/* Footer */}
      <div style={{ padding: "8px 16px", borderTop: "1px solid " + C.borderLight, display: "flex", alignItems: "center", gap: 6 }}>
        <Users size={12} color={C.brand} />
        <span style={{ fontSize: 12, color: C.brand, fontWeight: 500 }}>
          {path.length + " levels · " + path.filter(function(p) { return p.active; }).length + " auto-triggered for " + props.priority + " · " + (esc.assigned.name) + " assigned"}
        </span>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════
// CREATE TICKET MODAL
// ════════════════════════════════════════════

function CreateTicketModal(props) {
  var onClose = props.onClose;
  var onSubmit = props.onSubmit;

  var _f = useState({ description: "", priority: "P3", category: "Network", issue_type: "Failure" });
  var form = _f[0]; var setForm = _f[1];
  var _s = useState(false); var submitting = _s[0]; var setSubmitting = _s[1];
  var _r = useState(null); var result = _r[0]; var setResult = _r[1];

  function update(field, value) { var next = {}; for (var k in form) next[k] = form[k]; next[field] = value; setForm(next); }

  var escalation = getEscalationPath(form.priority, form.category);

  function handleSubmit() {
    if (!form.description.trim()) { alert("Please enter a ticket description."); return; }
    setSubmitting(true);
    fetch("http://localhost:8000/api/tickets/create", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        description: form.description, priority: form.priority, category: form.category,
        issue_type: form.issue_type, notify_email: escalation.assigned.email || "",
        assigned_engineer: escalation.assigned.id, assigned_name: escalation.assigned.name,
        escalation_path: escalation.path.map(function(s) { return { level: s.level, id: s.id, name: s.name, role: s.role, time: s.time, rule: s.rule || "" }; }),
      }),
    })
      .then(function(res) { return res.json(); })
      .then(function(data) { setResult(data); if (onSubmit) onSubmit(data); setSubmitting(false); })
      .catch(function(err) { setResult({ success: false, error: err.message }); setSubmitting(false); });
  }

  return (
    <div style={{ position: "fixed", top: 0, left: 0, right: 0, bottom: 0, background: "rgba(0,0,0,0.5)", zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={onClose}>
      <div style={{ background: C.white, borderRadius: 16, width: 640, maxHeight: "92vh", overflow: "auto", boxShadow: "0 20px 60px rgba(0,0,0,0.15)" }} onClick={function(e) { e.stopPropagation(); }}>

        <div style={{ padding: "20px 24px", borderBottom: "1px solid " + C.border, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, color: C.t9 }}>Create New Ticket</div>
            <div style={{ fontSize: 13, color: C.t5, marginTop: 2 }}>AI assigns and routes via SLA escalation matrix</div>
          </div>
          <button onClick={onClose} style={{ border: "none", background: C.bg, borderRadius: 8, width: 32, height: 32, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}><X size={16} color={C.t6} /></button>
        </div>

        <div style={{ padding: "20px 24px" }}>
          {result && (
            <div style={{ padding: 12, borderRadius: 8, marginBottom: 16, background: result.success ? "#F0FDF4" : "#FEF2F2", border: "1px solid " + (result.success ? "#BBF7D0" : "#FECACA"), fontSize: 13, color: result.success ? "#166534" : "#991B1B" }}>
              {result.success ? "Ticket " + result.ticket_id + " created! Assigned to " + escalation.assigned.id + " (" + escalation.assigned.name + "). " + (result.email_sent ? "Notification sent to " + escalation.assigned.email + "." : "") : "Error: " + (result.error || "Failed")}
            </div>
          )}

          <div style={{ marginBottom: 16 }}>
            <label style={labelStyle}>Description *</label>
            <textarea value={form.description} onChange={function(e) { update("description", e.target.value); }} placeholder="Describe the issue..." rows={3} style={{ ...inputStyle, resize: "vertical" }} />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 16 }}>
            <div>
              <label style={labelStyle}>Priority</label>
              <select value={form.priority} onChange={function(e) { update("priority", e.target.value); }} style={inputStyle}>
                <option value="P1">P1 — Critical (1h response / 4h resolution)</option>
                <option value="P2">P2 — High (2h response / 8h resolution)</option>
                <option value="P3">P3 — Medium (4h response / 24h resolution)</option>
                <option value="P4">P4 — Low (8h response / 48h resolution)</option>
              </select>
            </div>
            <div>
              <label style={labelStyle}>Category</label>
              <select value={form.category} onChange={function(e) { update("category", e.target.value); }} style={inputStyle}>
                <option value="Network">Network</option>
                <option value="Hardware">Hardware</option>
                <option value="Software">Software</option>
                <option value="Security">Security</option>
                <option value="Database">Database</option>
              </select>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 16 }}>
            <div>
              <label style={labelStyle}>Issue Type</label>
              <select value={form.issue_type} onChange={function(e) { update("issue_type", e.target.value); }} style={inputStyle}>
                <option value="Failure">Failure</option><option value="Slow">Slow / Performance</option><option value="Access">Access / Auth</option><option value="Configuration">Configuration</option><option value="Other">Other</option>
              </select>
            </div>
            <div>
              <label style={labelStyle}>Assigned Engineer Email</label>
              <input type="email" value={escalation.assigned.email || ""} readOnly style={{ ...inputStyle, background: "#F1F5F9", color: C.t9, fontWeight: 500, cursor: "default" }} />
            </div>
          </div>

          <EscalationPreview priority={form.priority} category={form.category} />
        </div>

        <div style={{ padding: "16px 24px", borderTop: "1px solid " + C.border, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ fontSize: 12, color: C.t5 }}>
            {"Assigning to: "}<strong style={{ color: C.brand }}>{escalation.assigned.id}</strong>{" (" + escalation.assigned.name + " · " + escalation.assigned.exp + "yr exp)"}
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button onClick={onClose} style={btn()}>Cancel</button>
            <button onClick={handleSubmit} disabled={submitting} style={{ ...btn(C.brand, "#fff", C.brand), opacity: submitting ? 0.6 : 1 }}>
              <Plus size={14} /> {submitting ? "Creating..." : "Create Ticket"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════
// TICKETS PAGE
// ════════════════════════════════════════════

export default function Tickets() {
  var _d = useApi("/api/tickets"); var data = _d.data; var loading = _d.loading; var error = _d.error;
  var _sel = useState(0); var selIdx = _sel[0]; var setSelIdx = _sel[1];
  var _m = useState(false); var showModal = _m[0]; var setShowModal = _m[1];

  if (loading) return <div style={{ padding: 40, textAlign: "center", color: C.t5 }}>Loading tickets...</div>;
  if (error) return <div style={{ padding: 40, textAlign: "center", color: C.red }}>{"Error: " + error.message}</div>;

  var tickets = Array.isArray(data) ? data : [];
  if (tickets.length === 0) return <div style={{ padding: 40, color: C.t5 }}>No ticket data available.</div>;
  var sel = tickets[selIdx] || tickets[0];

  return (
    <div>
      {showModal && <CreateTicketModal onClose={function() { setShowModal(false); }} onSubmit={function(r) { if (r.success) setTimeout(function() { setShowModal(false); }, 2500); }} />}

      <PageHeader title="Service Ticket Management" subtitle="Manage, classify, and resolve IT incidents using predictive AI insights."
        right={<>
          <button style={btn()} onClick={function() { window.open("http://localhost:8000/api/export/tickets", "_blank"); }}><ArrowUpRight size={14} /> Export Report</button>
          <button style={btn(C.brand, "#fff", C.brand)} onClick={function() { setShowModal(true); }}><Plus size={14} /> Create New Ticket</button>
        </>} />

      <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
        <select style={{ ...btn(), flex: 1 }}><option>Priority: All</option><option>P1</option><option>P2</option><option>P3</option><option>P4</option></select>
        <select style={{ ...btn(), flex: 1 }}><option>Status: All</option><option>Open</option><option>Resolved</option></select>
        <select style={{ ...btn(), flex: 1 }}><option>Team: All</option><option>Network Ops</option><option>Server Ops</option></select>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16 }}>
        <div style={{ background: C.white, borderRadius: 12, border: "1px solid " + C.border, overflow: "hidden" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead><tr><th style={thStyle}>Ticket ID</th><th style={thStyle}>Subject</th><th style={thStyle}>Priority</th><th style={thStyle}>Category</th><th style={thStyle}>Status</th><th style={thStyle}>Assigned</th></tr></thead>
            <tbody>
              {tickets.map(function(t, idx) {
                var isA = idx === selIdx;
                var eng = t.assigned_engineer || {};
                var engId = typeof eng === "object" ? (eng.engineer_id || "—") : (eng || "—");
                var engName = EMPLOYEES[engId] ? EMPLOYEES[engId].name : engId;
                return (
                  <tr key={t.year_ticket_id || idx} onClick={function() { setSelIdx(idx); }} style={{ cursor: "pointer", background: isA ? C.brandLight : "transparent" }}>
                    <td style={tdStyle}><span style={{ fontWeight: 600, color: C.brand, fontSize: 13 }}>{t.year_ticket_id}</span></td>
                    <td style={tdStyle}><div style={{ fontWeight: 500, color: C.t9 }}>{t.ticket_description}</div><div style={{ fontSize: 12, color: C.t5 }}>{formatTs(t.ticket_created_timestamp)}</div></td>
                    <td style={tdStyle}><Pill tone={priorityTone(t.predicted_priority || t.actual_priority)}>{priorityLabel(t.predicted_priority || t.actual_priority)}</Pill></td>
                    <td style={tdStyle}><Pill tone="slate">{t.predicted_category || t.actual_category}</Pill></td>
                    <td style={tdStyle}><Pill tone={t.ticket_status === "Open" ? "blue" : "green"}>{t.ticket_status}</Pill></td>
                    <td style={tdStyle}><span style={{ fontSize: 13, fontWeight: 500, color: C.t9 }}>{engName}</span></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div style={{ padding: "12px 16px", borderTop: "1px solid " + C.borderLight, fontSize: 13, color: C.t5 }}>{"Showing " + tickets.length + " tickets"}</div>
        </div>

        {/* Detail panel */}
        <div style={{ background: C.white, borderRadius: 12, border: "1px solid " + C.border, padding: 20 }}>
          <div style={{ fontSize: 12, color: C.t5, marginBottom: 4 }}>{sel.year_ticket_id}</div>
          <div style={{ fontSize: 16, fontWeight: 700, color: C.t9, marginBottom: 4 }}>{sel.ticket_description}</div>
          <div style={{ fontSize: 12, color: C.t5, marginBottom: 16 }}>{formatTs(sel.ticket_created_timestamp)}</div>
          <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
            <div><div style={{ fontSize: 11, color: C.t5, marginBottom: 4 }}>Priority</div><Pill tone={priorityTone(sel.predicted_priority || sel.actual_priority)}>{priorityLabel(sel.predicted_priority || sel.actual_priority)}</Pill></div>
            <div><div style={{ fontSize: 11, color: C.t5, marginBottom: 4 }}>SLA Risk</div><Pill tone={sel.sla_risk_band === "High" ? "red" : sel.sla_risk_band === "Watch" ? "amber" : "green"}>{sel.sla_risk_band || "OnTrack"}</Pill></div>
          </div>

          {/* Escalation path for selected ticket */}
          {(sel.predicted_priority || sel.actual_priority) && (sel.predicted_category || sel.actual_category) && (
            <div style={{ borderTop: "1px solid " + C.borderLight, paddingTop: 12, marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: C.t5, marginBottom: 6 }}>Escalation path</div>
              {(function() {
                var esc = getEscalationPath(sel.predicted_priority || sel.actual_priority, sel.predicted_category || sel.actual_category);
                var breached = sel.sla_risk_band === "High" || (sel.sla_breach_probability || 0) >= 0.5;
                return esc.path.map(function(step, i) {
                  var triggered = step.active || (breached && i <= 2);
                  return (
                    <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, padding: "3px 0" }}>
                      <span style={{ color: triggered ? C.t9 : C.t5, fontWeight: triggered ? 600 : 400 }}>
                        {step.level.split("—")[0].trim()}
                        {triggered && !step.active ? " TRIGGERED" : ""}
                      </span>
                      <span style={{ color: triggered && !step.active ? C.rose : C.t4 }}>{step.id + " · " + step.name}</span>
                    </div>
                  );
                });
              })()}
            </div>
          )}

          {/* Similar tickets */}
          {sel.recommendations && (sel.recommendations.similar_tickets || []).length > 0 && (
            <div style={{ borderTop: "1px solid " + C.borderLight, paddingTop: 12 }}>
              <div style={{ fontSize: 11, color: C.t5, marginBottom: 6 }}>Similar tickets</div>
              {sel.recommendations.similar_tickets.slice(0, 3).map(function(st, i) {
                return (
                  <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, padding: "3px 0" }}>
                    <span style={{ color: C.brand, fontWeight: 500 }}>{st.year_ticket_id}</span>
                    <span style={{ color: C.t5 }}>{"sim=" + (st.similarity_score ? st.similarity_score.toFixed(2) : "?") + " · " + minutesToHm(st.resolution_time_minutes)}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
      <DataPatterns tickets={tickets} />
    </div>
  );
}
