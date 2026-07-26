// src/components/DataPatterns.jsx
// Data Pattern Analysis — Tickets (category + engineer) AND Procurement (stock + device type)
// Usage: <DataPatterns tickets={[...]} /> or <DataPatterns procurement={[...]} /> or both

import { useState } from "react";
import { C } from "../utils/tokens";
import Pill from "./ui/Pill";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, PieChart, Pie } from "recharts";

var TH = { textAlign: "left", fontWeight: 500, fontSize: 11, color: C.t6, textTransform: "uppercase", padding: "8px 12px", background: C.bg, borderBottom: "1px solid " + C.border };
var TD = { padding: "10px 12px", borderBottom: "1px solid " + C.borderLight, fontSize: 13, color: C.t7 };

var PRIORITY_COLORS = { P1: "#EF4444", P2: "#F59E0B", P3: "#3B82F6", P4: "#94A3B8" };
var DEVICE_COLORS = { Server: "#3B82F6", Router: "#8B5CF6", Firewall: "#EF4444", Switch: "#14B8A6" };
var URGENCY_COLORS = { Urgent: "#EF4444", Planning: "#F59E0B", Optimal: "#10B981" };
var MFR_COLORS = { Dell: "#007DB8", Cisco: "#049FD9", HP: "#0096D6", Fortinet: "#EE3124" };

// ══════════════ TICKET ANALYSIS ══════════════
function analyzeTickets(tickets) {
  var cats = {};
  var priorities = {};
  tickets.forEach(function (t) {
    var cat = t.predicted_category || t.actual_category || "Other";
    var prio = t.predicted_priority || t.actual_priority || "P3";
    var st = String(t.ticket_status || "").toLowerCase();
    var isResolved = ["closed", "resolved", "complete", "done"].indexOf(st) >= 0;
    if (!cats[cat]) cats[cat] = { total: 0, resolved: 0, open: 0, P1: 0, P2: 0, P3: 0, P4: 0 };
    cats[cat].total++; cats[cat][prio] = (cats[cat][prio] || 0) + 1;
    if (isResolved) cats[cat].resolved++; else cats[cat].open++;
    priorities[prio] = (priorities[prio] || 0) + 1;
  });
  var catItems = Object.keys(cats).map(function (name) {
    var c = cats[name];
    return { name: name, total: c.total, resolved: c.resolved, open: c.open, rate: c.total > 0 ? Math.round(c.resolved / c.total * 100) : 0, P1: c.P1, P2: c.P2, P3: c.P3, P4: c.P4 };
  }).sort(function (a, b) { return b.total - a.total; });
  var prioData = ["P1", "P2", "P3", "P4"].map(function (p) { return { name: p, value: priorities[p] || 0 }; }).filter(function (d) { return d.value > 0; });

  var engs = {};
  tickets.forEach(function (t) {
    var eng = t.assigned_engineer; var id = typeof eng === "object" ? (eng.engineer_id || "—") : (eng || "—");
    var st = String(t.ticket_status || "").toLowerCase();
    var isResolved = ["closed", "resolved", "complete", "done"].indexOf(st) >= 0;
    var prio = t.predicted_priority || t.actual_priority || "P3";
    var cat = t.predicted_category || t.actual_category || "Other";
    if (!engs[id]) engs[id] = { tickets: 0, resolved: 0, cats: {}, P1: 0, P2: 0, P3: 0, P4: 0 };
    engs[id].tickets++; if (isResolved) engs[id].resolved++;
    engs[id].cats[cat] = (engs[id].cats[cat] || 0) + 1;
    engs[id][prio] = (engs[id][prio] || 0) + 1;
  });
  var engList = Object.keys(engs).map(function (id) {
    var e = engs[id];
    return { id: id, tickets: e.tickets, resolved: e.resolved, rate: e.tickets > 0 ? Math.round(e.resolved / e.tickets * 100) : 0, top_cat: Object.keys(e.cats).sort(function (a, b) { return e.cats[b] - e.cats[a]; })[0] || "—", P1: e.P1, P2: e.P2, P3: e.P3, P4: e.P4 };
  }).sort(function (a, b) { return b.tickets - a.tickets; });
  return { catItems: catItems, prioData: prioData, engList: engList };
}

// ══════════════ PROCUREMENT ANALYSIS ══════════════
function analyzeProcurement(plan) {
  var types = {}; var urgencies = {}; var mfrs = {};
  var totalCost = 0; var belowThreshold = 0;
  plan.forEach(function (p) {
    var dt = p.device_type || "Unknown";
    var urg = !p.to_order_flag ? "Optimal" : (p.lead_time_days || 0) >= 14 ? "Urgent" : "Planning";
    var mfr = p.manufacturer || "Unknown";
    var stock = p.current_stock_quantity || 0;
    var thresh = p.reorder_threshold_quantity || 1;
    if (stock < thresh) belowThreshold++;
    if (p.to_order_flag) totalCost += (p.unit_cost || 0) * (p.to_order_quantity || 0);

    if (!types[dt]) types[dt] = { count: 0, toOrder: 0, totalStock: 0, totalThresh: 0, totalCost: 0, urgent: 0 };
    types[dt].count++; types[dt].totalStock += stock; types[dt].totalThresh += thresh;
    if (p.to_order_flag) { types[dt].toOrder++; types[dt].totalCost += (p.unit_cost || 0) * (p.to_order_quantity || 0); }
    if (urg === "Urgent") types[dt].urgent++;

    urgencies[urg] = (urgencies[urg] || 0) + 1;
    mfrs[mfr] = (mfrs[mfr] || 0) + 1;
  });

  var typeItems = Object.keys(types).map(function (name) {
    var t = types[name]; var healthPct = t.totalThresh > 0 ? Math.round(t.totalStock / t.totalThresh * 100) : 0;
    return { name: name, count: t.count, toOrder: t.toOrder, healthPct: healthPct, totalCost: Math.round(t.totalCost), urgent: t.urgent, avgStock: Math.round(t.totalStock / t.count), avgThresh: Math.round(t.totalThresh / t.count) };
  }).sort(function (a, b) { return b.count - a.count; });

  var urgData = ["Urgent", "Planning", "Optimal"].map(function (u) { return { name: u, value: urgencies[u] || 0 }; }).filter(function (d) { return d.value > 0; });
  var mfrData = Object.keys(mfrs).map(function (m) { return { name: m, value: mfrs[m] }; }).sort(function (a, b) { return b.value - a.value; });

  return { typeItems: typeItems, urgData: urgData, mfrData: mfrData, totalCost: totalCost, belowThreshold: belowThreshold };
}

// ══════════════ MAIN COMPONENT ══════════════
export default function DataPatterns(props) {
  var tickets = props.tickets || [];
  var procurement = props.procurement || [];
  var hasTickets = tickets.length > 0;
  var hasProcurement = procurement.length > 0;

  var defaultTab = hasTickets ? "categories" : "stock";
  var _tab = useState(defaultTab);
  var tab = _tab[0]; var setTab = _tab[1];

  if (!hasTickets && !hasProcurement) return null;

  var tRes = hasTickets ? analyzeTickets(tickets) : null;
  var pRes = hasProcurement ? analyzeProcurement(procurement) : null;

  var totalResolved = tickets.filter(function (t) { var s = String(t.ticket_status || "").toLowerCase(); return ["closed", "resolved", "complete", "done"].indexOf(s) >= 0; }).length;
  var overallRate = tickets.length > 0 ? Math.round(totalResolved / tickets.length * 100) : 0;

  return (
    <div style={{ marginTop: 20 }}>
      {/* Header */}
      <div style={{ background: C.white, borderRadius: "12px 12px 0 0", border: "1px solid " + C.border, borderBottom: "none", padding: "16px 20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 600, color: C.t9 }}>Data Pattern Analysis</div>
          <div style={{ fontSize: 12, color: C.t5 }}>
            {hasTickets && hasProcurement ? "Ticket distribution and procurement insights" :
             hasTickets ? "Ticket distribution, priority breakdown, and engineer workload across " + tickets.length + " tickets" :
             "Stock health, device distribution, and procurement cost across " + procurement.length + " items"}
          </div>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {hasTickets && <Pill tone="blue">{tickets.length + " Tickets"}</Pill>}
          {hasTickets && <Pill tone={overallRate >= 50 ? "green" : "red"}>{overallRate + "% Resolved"}</Pill>}
          {hasProcurement && <Pill tone="green">{procurement.length + " Items"}</Pill>}
          {hasProcurement && pRes && <Pill tone="red">{pRes.belowThreshold + " Low Stock"}</Pill>}
        </div>
      </div>

      {/* Sub-tabs */}
      <div style={{ background: C.white, borderLeft: "1px solid " + C.border, borderRight: "1px solid " + C.border, padding: "0 20px 12px", display: "flex", gap: 4 }}>
        {hasTickets && <button onClick={function () { setTab("categories"); }} style={{ padding: "4px 10px", borderRadius: 4, fontSize: 11, fontWeight: 500, cursor: "pointer", border: "1px solid " + C.borderLight, background: tab === "categories" ? C.bg : "transparent", color: tab === "categories" ? C.t9 : C.t5 }}>By Category</button>}
        {hasTickets && <button onClick={function () { setTab("engineers"); }} style={{ padding: "4px 10px", borderRadius: 4, fontSize: 11, fontWeight: 500, cursor: "pointer", border: "1px solid " + C.borderLight, background: tab === "engineers" ? C.bg : "transparent", color: tab === "engineers" ? C.t9 : C.t5 }}>By Engineer</button>}
        {hasProcurement && <button onClick={function () { setTab("stock"); }} style={{ padding: "4px 10px", borderRadius: 4, fontSize: 11, fontWeight: 500, cursor: "pointer", border: "1px solid " + C.borderLight, background: tab === "stock" ? C.bg : "transparent", color: tab === "stock" ? C.t9 : C.t5 }}>Stock Overview</button>}
        {hasProcurement && <button onClick={function () { setTab("procurement"); }} style={{ padding: "4px 10px", borderRadius: 4, fontSize: 11, fontWeight: 500, cursor: "pointer", border: "1px solid " + C.borderLight, background: tab === "procurement" ? C.bg : "transparent", color: tab === "procurement" ? C.t9 : C.t5 }}>Procurement Cost</button>}
      </div>

      {/* ══════════════ TICKET: CATEGORY VIEW ══════════════ */}
      {tab === "categories" && tRes && (
        <div style={{ background: C.white, borderLeft: "1px solid " + C.border, borderRight: "1px solid " + C.border }}>
          <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: 0, borderBottom: "1px solid " + C.borderLight }}>
            <div style={{ padding: "12px 20px", borderRight: "1px solid " + C.borderLight }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: C.t6, marginBottom: 8 }}>Tickets by Category & Priority</div>
              <div style={{ height: 180 }}>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={tRes.catItems} layout="vertical" barSize={22}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E5E7EB" />
                    <XAxis type="number" tickLine={false} axisLine={false} tick={{ fill: "#6B7280", fontSize: 11 }} />
                    <YAxis type="category" dataKey="name" tickLine={false} axisLine={false} tick={{ fill: "#374151", fontSize: 12, fontWeight: 600 }} width={80} />
                    <Tooltip />
                    <Bar dataKey="P1" stackId="a" fill="#EF4444" name="P1 Critical" />
                    <Bar dataKey="P2" stackId="a" fill="#F59E0B" name="P2 High" />
                    <Bar dataKey="P3" stackId="a" fill="#3B82F6" name="P3 Medium" />
                    <Bar dataKey="P4" stackId="a" fill="#94A3B8" name="P4 Low" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div style={{ padding: "12px 20px" }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: C.t6, marginBottom: 8 }}>Priority Distribution</div>
              <div style={{ height: 180 }}>
                <ResponsiveContainer width="100%" height={180}>
                  <PieChart>
                    <Pie data={tRes.prioData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={35} outerRadius={65} paddingAngle={3} label={function (d) { return d.name + " (" + d.value + ")"; }} labelLine={false} style={{ fontSize: 11 }}>
                      {tRes.prioData.map(function (entry) { return <Cell key={entry.name} fill={PRIORITY_COLORS[entry.name] || "#94A3B8"} />; })}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(" + Math.min(tRes.catItems.length, 4) + ", 1fr)", gap: 12, padding: "16px 20px" }}>
            {tRes.catItems.map(function (cat) {
              return (
                <div key={cat.name} style={{ border: "1px solid " + C.border, borderRadius: 10, padding: 14 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                    <span style={{ fontSize: 14, fontWeight: 700, color: C.t9 }}>{cat.name}</span>
                    <Pill tone={cat.rate >= 50 ? "green" : cat.rate >= 25 ? "amber" : "red"}>{cat.rate + "% resolved"}</Pill>
                  </div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: C.brand, marginBottom: 4 }}>{cat.total}</div>
                  <div style={{ fontSize: 11, color: C.t5, marginBottom: 8 }}>{cat.open + " open · " + cat.resolved + " resolved"}</div>
                  <div style={{ width: "100%", height: 6, background: C.borderLight, borderRadius: 3, overflow: "hidden" }}>
                    <div style={{ width: cat.rate + "%", height: 6, background: cat.rate >= 50 ? "#10B981" : cat.rate >= 25 ? "#F59E0B" : "#EF4444", borderRadius: 3 }} />
                  </div>
                  <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
                    {["P1", "P2", "P3", "P4"].map(function (p) { var v = cat[p] || 0; if (v === 0) return null; return <span key={p} style={{ fontSize: 10, padding: "2px 5px", borderRadius: 3, background: PRIORITY_COLORS[p] + "18", color: PRIORITY_COLORS[p], fontWeight: 600 }}>{p + ": " + v}</span>; })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ══════════════ TICKET: ENGINEER VIEW ══════════════ */}
      {tab === "engineers" && tRes && (
        <div style={{ background: C.white, borderLeft: "1px solid " + C.border, borderRight: "1px solid " + C.border }}>
          <div style={{ padding: "12px 20px", borderBottom: "1px solid " + C.borderLight }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: C.t6, marginBottom: 8 }}>Engineer Workload & Resolution</div>
            <div style={{ height: Math.max(200, Math.min(tRes.engList.length, 12) * 32) }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={tRes.engList.slice(0, 12)} layout="vertical" barGap={2} barSize={14}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E5E7EB" />
                  <XAxis type="number" tickLine={false} axisLine={false} tick={{ fill: "#6B7280", fontSize: 11 }} />
                  <YAxis type="category" dataKey="id" tickLine={false} axisLine={false} tick={{ fill: "#374151", fontSize: 11, fontWeight: 600 }} width={50} />
                  <Tooltip />
                  <Bar dataKey="tickets" fill="#3B82F6" name="Total Tickets" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="resolved" fill="#10B981" name="Resolved" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead><tr><th style={TH}>Engineer</th><th style={TH}>Tickets</th><th style={TH}>Resolved</th><th style={TH}>Rate</th><th style={TH}>Priority Mix</th><th style={TH}>Top Category</th><th style={TH}>Performance</th></tr></thead>
            <tbody>{tRes.engList.map(function (eng) {
              return (
                <tr key={eng.id}>
                  <td style={{ ...TD, fontWeight: 600, color: C.brand }}>{eng.id}</td>
                  <td style={{ ...TD, fontWeight: 600 }}>{eng.tickets}</td>
                  <td style={TD}>{eng.resolved}</td>
                  <td style={TD}><div style={{ display: "flex", alignItems: "center", gap: 8 }}><div style={{ width: 60, height: 6, background: C.borderLight, borderRadius: 3, overflow: "hidden" }}><div style={{ width: eng.rate + "%", height: 6, background: eng.rate >= 70 ? "#10B981" : eng.rate >= 40 ? "#F59E0B" : "#EF4444", borderRadius: 3 }} /></div><span style={{ fontSize: 12, fontWeight: 600, color: eng.rate >= 70 ? "#10B981" : eng.rate >= 40 ? "#F59E0B" : "#EF4444" }}>{eng.rate + "%"}</span></div></td>
                  <td style={TD}><div style={{ display: "flex", gap: 3 }}>{["P1", "P2", "P3", "P4"].map(function (p) { var v = eng[p] || 0; if (v === 0) return null; return <span key={p} style={{ fontSize: 9, padding: "1px 4px", borderRadius: 3, background: PRIORITY_COLORS[p] + "18", color: PRIORITY_COLORS[p], fontWeight: 600 }}>{p + ":" + v}</span>; })}</div></td>
                  <td style={TD}><Pill tone="slate">{eng.top_cat}</Pill></td>
                  <td style={TD}><Pill tone={eng.rate >= 70 ? "green" : eng.rate >= 40 ? "amber" : "red"}>{eng.rate >= 70 ? "Strong" : eng.rate >= 40 ? "Average" : "Review"}</Pill></td>
                </tr>
              );
            })}</tbody>
          </table>
        </div>
      )}

      {/* ══════════════ PROCUREMENT: STOCK OVERVIEW ══════════════ */}
      {tab === "stock" && pRes && (
        <div style={{ background: C.white, borderLeft: "1px solid " + C.border, borderRight: "1px solid " + C.border }}>
          <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: 0, borderBottom: "1px solid " + C.borderLight }}>
            {/* Device type bar chart */}
            <div style={{ padding: "12px 20px", borderRight: "1px solid " + C.borderLight }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: C.t6, marginBottom: 8 }}>Items by Device Type — Stock Health</div>
              <div style={{ height: 200 }}>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={pRes.typeItems} barGap={4} barSize={20}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
                    <XAxis dataKey="name" tickLine={false} axisLine={false} tick={{ fill: "#374151", fontSize: 12, fontWeight: 600 }} />
                    <YAxis tickLine={false} axisLine={false} tick={{ fill: "#6B7280", fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="avgStock" fill="#3B82F6" name="Avg Stock" radius={[4, 4, 0, 0]}>
                      {pRes.typeItems.map(function (entry) { return <Cell key={entry.name} fill={DEVICE_COLORS[entry.name] || "#3B82F6"} />; })}
                    </Bar>
                    <Bar dataKey="avgThresh" fill="#E5E7EB" name="Avg Threshold" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            {/* Urgency pie */}
            <div style={{ padding: "12px 20px" }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: C.t6, marginBottom: 8 }}>Urgency Distribution</div>
              <div style={{ height: 200 }}>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie data={pRes.urgData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={35} outerRadius={65} paddingAngle={3} label={function (d) { return d.name + " (" + d.value + ")"; }} labelLine={false} style={{ fontSize: 11 }}>
                      {pRes.urgData.map(function (entry) { return <Cell key={entry.name} fill={URGENCY_COLORS[entry.name] || "#94A3B8"} />; })}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Device type summary cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(" + Math.min(pRes.typeItems.length, 4) + ", 1fr)", gap: 12, padding: "16px 20px" }}>
            {pRes.typeItems.map(function (dt) {
              var health = dt.healthPct;
              return (
                <div key={dt.name} style={{ border: "1px solid " + C.border, borderRadius: 10, padding: 14 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                    <span style={{ fontSize: 14, fontWeight: 700, color: C.t9 }}>{dt.name}</span>
                    <Pill tone={health >= 100 ? "green" : health >= 70 ? "amber" : "red"}>{health + "% health"}</Pill>
                  </div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: DEVICE_COLORS[dt.name] || C.brand, marginBottom: 4 }}>{dt.count}</div>
                  <div style={{ fontSize: 11, color: C.t5, marginBottom: 8 }}>{dt.toOrder + " to reorder · " + dt.urgent + " urgent"}</div>
                  <div style={{ width: "100%", height: 6, background: C.borderLight, borderRadius: 3, overflow: "hidden" }}>
                    <div style={{ width: Math.min(health, 100) + "%", height: 6, background: health >= 100 ? "#10B981" : health >= 70 ? "#F59E0B" : "#EF4444", borderRadius: 3 }} />
                  </div>
                  <div style={{ fontSize: 11, color: C.t5, marginTop: 6 }}>{"Avg stock: " + dt.avgStock + " / " + dt.avgThresh + " threshold"}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ══════════════ PROCUREMENT: COST VIEW ══════════════ */}
      {tab === "procurement" && pRes && (
        <div style={{ background: C.white, borderLeft: "1px solid " + C.border, borderRight: "1px solid " + C.border }}>
          <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: 0, borderBottom: "1px solid " + C.borderLight }}>
            {/* Cost by device type */}
            <div style={{ padding: "12px 20px", borderRight: "1px solid " + C.borderLight }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: C.t6, marginBottom: 8 }}>Procurement Cost by Device Type</div>
              <div style={{ height: 200 }}>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={pRes.typeItems} layout="vertical" barSize={22}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E5E7EB" />
                    <XAxis type="number" tickLine={false} axisLine={false} tick={{ fill: "#6B7280", fontSize: 11 }} tickFormatter={function (v) { return "$" + Math.round(v / 1000) + "K"; }} />
                    <YAxis type="category" dataKey="name" tickLine={false} axisLine={false} tick={{ fill: "#374151", fontSize: 12, fontWeight: 600 }} width={70} />
                    <Tooltip formatter={function (v) { return "$" + v.toLocaleString(); }} />
                    <Bar dataKey="totalCost" fill="#3B82F6" name="Total Cost" radius={[0, 4, 4, 0]}>
                      {pRes.typeItems.map(function (entry) { return <Cell key={entry.name} fill={DEVICE_COLORS[entry.name] || "#3B82F6"} />; })}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            {/* Manufacturer pie */}
            <div style={{ padding: "12px 20px" }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: C.t6, marginBottom: 8 }}>Items by Manufacturer</div>
              <div style={{ height: 200 }}>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie data={pRes.mfrData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={35} outerRadius={65} paddingAngle={3} label={function (d) { return d.name + " (" + d.value + ")"; }} labelLine={false} style={{ fontSize: 11 }}>
                      {pRes.mfrData.map(function (entry) { return <Cell key={entry.name} fill={MFR_COLORS[entry.name] || "#6B7280"} />; })}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Cost detail table */}
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead><tr><th style={TH}>Device Type</th><th style={TH}>Items</th><th style={TH}>To Reorder</th><th style={TH}>Urgent</th><th style={TH}>Est. Cost</th><th style={TH}>Stock Health</th><th style={TH}>Status</th></tr></thead>
            <tbody>{pRes.typeItems.map(function (dt) {
              var health = dt.healthPct;
              return (
                <tr key={dt.name}>
                  <td style={{ ...TD, fontWeight: 600, color: DEVICE_COLORS[dt.name] || C.brand }}>{dt.name}</td>
                  <td style={{ ...TD, fontWeight: 600 }}>{dt.count}</td>
                  <td style={TD}>{dt.toOrder}</td>
                  <td style={{ ...TD, fontWeight: 600, color: dt.urgent > 0 ? "#EF4444" : C.t7 }}>{dt.urgent}</td>
                  <td style={{ ...TD, fontWeight: 600 }}>{"$" + dt.totalCost.toLocaleString()}</td>
                  <td style={TD}><div style={{ display: "flex", alignItems: "center", gap: 8 }}><div style={{ width: 60, height: 6, background: C.borderLight, borderRadius: 3, overflow: "hidden" }}><div style={{ width: Math.min(health, 100) + "%", height: 6, background: health >= 100 ? "#10B981" : health >= 70 ? "#F59E0B" : "#EF4444", borderRadius: 3 }} /></div><span style={{ fontSize: 12 }}>{health + "%"}</span></div></td>
                  <td style={TD}><Pill tone={health >= 100 ? "green" : health >= 70 ? "amber" : "red"}>{health >= 100 ? "Healthy" : health >= 70 ? "Low" : "Critical"}</Pill></td>
                </tr>
              );
            })}</tbody>
          </table>
        </div>
      )}

      {/* Footer */}
      <div style={{ background: C.white, borderRadius: "0 0 12px 12px", border: "1px solid " + C.border, padding: "10px 20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 11, color: C.t4 }}>
          {hasTickets && tRes ? "Tickets: " + tickets.length + " · " + tRes.catItems.length + " categories · " + tRes.engList.length + " engineers" : ""}
          {hasTickets && hasProcurement ? " | " : ""}
          {hasProcurement && pRes ? "Inventory: " + procurement.length + " items · " + pRes.belowThreshold + " below threshold · $" + Math.round(pRes.totalCost).toLocaleString() + " pending" : ""}
        </span>
        <div style={{ display: "flex", gap: 8 }}>
          {(tab === "categories" || tab === "engineers") && ["P1", "P2", "P3", "P4"].map(function (p) {
            return <span key={p} style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: C.t5 }}><span style={{ width: 8, height: 8, borderRadius: 2, background: PRIORITY_COLORS[p] }} />{p}</span>;
          })}
          {(tab === "stock" || tab === "procurement") && Object.keys(DEVICE_COLORS).map(function (d) {
            return <span key={d} style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: C.t5 }}><span style={{ width: 8, height: 8, borderRadius: 2, background: DEVICE_COLORS[d] }} />{d}</span>;
          })}
        </div>
      </div>
    </div>
  );
}
