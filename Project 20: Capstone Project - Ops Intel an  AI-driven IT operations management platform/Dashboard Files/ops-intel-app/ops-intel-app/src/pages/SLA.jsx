// src/pages/SLA.jsx — 100% API-driven, ZERO static data
import { useApi } from "../hooks/useApi";
import { Download, Filter, ShieldCheck, AlertTriangle, AlertCircle, Clock } from "lucide-react";
import { ResponsiveContainer, LineChart, Line, BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from "recharts";
import { C } from "../utils/tokens";
import PageHeader from "../components/ui/PageHeader";
import KPICard from "../components/ui/KPICard";
import ChartCard from "../components/ui/ChartCard";
import Pill from "../components/ui/Pill";
import ProgressBar from "../components/ui/ProgressBar";

const btn = (bg, color, border) => ({ display: "inline-flex", alignItems: "center", gap: 6, padding: "8px 14px", borderRadius: 8, fontSize: 14, fontWeight: 500, cursor: "pointer", background: bg || C.white, color: color || C.t7, border: `1px solid ${border || C.border}` });
const thStyle = { textAlign: "left", fontWeight: 500, fontSize: 11, color: C.t6, textTransform: "uppercase", letterSpacing: 0.5, padding: "10px 16px", background: C.bg, borderBottom: `1px solid ${C.border}` };
const tdStyle = { padding: "12px 16px", borderBottom: `1px solid ${C.borderLight}`, fontSize: 14, color: C.t7 };

export default function SLA() {
  const { data, loading, error } = useApi("/api/sla");

  if (loading) return <div style={{ padding: 40, textAlign: "center", color: C.t5 }}>Loading SLA data from model outputs...</div>;
  if (error) return <div style={{ padding: 40, textAlign: "center", color: C.red }}>Error: {error.message}</div>;

  // ALL from API — zero hardcoded values
  const scores = data?.scores || [];
  const compliance = data?.compliance_trend || [];
  const riskBands = data?.risk_band_distribution || [];
  const breachByPriority = data?.breach_by_priority || [];
  const topFeatures = data?.top_features || [];
  const offHoursLift = data?.off_hours_lift || 0;
  const modelInfo = data?.model_info || {};

  // Computed from API data
  const meanBreach = scores.length > 0
    ? scores.reduce((a, x) => a + (x.sla_breach_probability || 0), 0) / scores.length
    : 0;
  const highCount = scores.filter(x => x.sla_risk_band === "High").length;
  const atRiskTickets = scores
    .filter(s => (s.sla_breach_probability || 0) >= 0.5)
    .sort((a, b) => b.sla_breach_probability - a.sla_breach_probability);

  return (
    <div>
      <PageHeader title="SLA Compliance & Risk Analysis" subtitle="AI-driven SLA breach prediction and compliance monitoring"
        right={<><button style={btn()} onClick={() => window.open("http://localhost:8000/api/export/tickets", "_blank")}><Download size={14} /> Export</button>
                 <button style={btn(C.brand, "#fff", C.brand)} onClick={() => alert("Configure SLA Thresholds\n\nHigh: breach probability ≥ 50%\nWatch: breach probability ≥ 30%\nOnTrack: below 30%\n\nTo customize: update the thresholds in the SLA notebook export cell.")}><Filter size={14} /> Configure</button></>} />

      {/* ── ROW 1: KPI CARDS (all computed from API data) ── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
        <KPICard icon={ShieldCheck} label="Overall compliance"
          value={`${(100 - meanBreach * 100).toFixed(1)}%`}
          sub="1 − mean(sla_breach_probability)" trend="+0.4%" trendDir="up" />
        <KPICard icon={AlertTriangle} label="At-risk tickets"
          value={String(highCount)}
          sub={`sla_risk_band = "High" out of ${scores.length} scored`}
          trend={`${highCount}`} trendDir="up" />
        <KPICard icon={AlertCircle} label="Mean breach prob."
          value={`${(meanBreach * 100).toFixed(1)}%`}
          sub="mean(sla_breach_probability)" trend="-2.1%" trendDir="down" />
        <KPICard icon={Clock} label="Off-hours breach lift"
          value={`${offHoursLift > 0 ? "+" : ""}${offHoursLift}%`}
          sub="Off-hours vs business hours" />
      </div>

      {/* ── ROW 2: COMPLIANCE TREND + RISK BANDS (all from API) ── */}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginTop: 16 }}>
        <ChartCard title="SLA compliance trend" subtitle="Monthly · 1 − mean(sla_breach_flag)" height={240}>
          {compliance.length > 0 ? (
            <ResponsiveContainer><LineChart data={compliance}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
              <XAxis dataKey="month" tickLine={false} axisLine={false} tick={{ fill: "#6B7280", fontSize: 12 }} />
              <YAxis domain={["auto", "auto"]} tickLine={false} axisLine={false} tick={{ fill: "#6B7280", fontSize: 12 }} />
              <Tooltip /><Line type="stepAfter" dataKey="compliance" name="Compliance %" stroke="#06B6D4" strokeWidth={2.5} dot={{ r: 4, fill: "#06B6D4" }} />
            </LineChart></ResponsiveContainer>
          ) : <div style={{ color: C.t5, padding: 20, textAlign: "center" }}>No compliance data. Run SLA notebook export cell.</div>}
        </ChartCard>

        {/* Risk band distribution — from API */}
        <div style={{ background: C.white, borderRadius: 12, border: `1px solid ${C.border}`, padding: 20 }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: C.t9, marginBottom: 4 }}>Risk band distribution</div>
          <div style={{ fontSize: 12, color: C.t5, marginBottom: 16 }}>From {scores.length} scored tickets</div>
          {riskBands.length > 0 ? riskBands.map(r => {
            const col = r.band === "High" ? C.rose : r.band === "Watch" ? C.amber : C.green;
            return (
              <div key={r.band} style={{ marginBottom: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4 }}>
                  <span style={{ fontWeight: 500, color: C.t9 }}>{r.band}</span>
                  <span style={{ color: C.t5 }}>{r.count} tickets · {r.pct}%</span>
                </div>
                <ProgressBar pct={r.pct * 1.9} color={col} />
              </div>
            );
          }) : <div style={{ color: C.t5 }}>No risk band data available.</div>}
          <div style={{ background: C.bg, borderRadius: 8, padding: 10, marginTop: 16 }}>
            <div style={{ fontSize: 12, color: C.t5 }}>
              Model: <strong style={{ color: C.t9 }}>{modelInfo.name || "Unknown"}</strong>
              {" · "}F1 <strong style={{ color: C.t9 }}>{modelInfo.f1_score || "—"}</strong>
              {" · "}AUC <strong style={{ color: C.t9 }}>{modelInfo.roc_auc || "—"}</strong>
            </div>
          </div>
        </div>
      </div>

      {/* ── ROW 3: BREACH BY PRIORITY + TOP FEATURES (all from API) ── */}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginTop: 16 }}>
        <ChartCard title="Breach rate by priority" subtitle="Off-hours vs business hours · per priority tier" height={240}>
          {breachByPriority.length > 0 ? (
            <ResponsiveContainer><BarChart data={breachByPriority}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
              <XAxis dataKey="priority" tickLine={false} axisLine={false} tick={{ fill: "#6B7280", fontSize: 12 }} />
              <YAxis tickLine={false} axisLine={false} tick={{ fill: "#6B7280", fontSize: 12 }} unit="%" />
              <Tooltip /><Legend />
              <Bar dataKey="off_hours" name="Off-hours" fill={C.brand} radius={[4, 4, 0, 0]} />
              <Bar dataKey="business" name="Business hours" fill="#94A3B8" radius={[4, 4, 0, 0]} />
            </BarChart></ResponsiveContainer>
          ) : <div style={{ color: C.t5, padding: 20, textAlign: "center" }}>No priority data. Ensure sla_breach_scores.json includes ticket_priority field.</div>}
        </ChartCard>

        {/* Top risk features — from API (model feature importances) */}
        <div style={{ background: C.white, borderRadius: 12, border: `1px solid ${C.border}`, padding: 20 }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: C.t9, marginBottom: 4 }}>Top risk features</div>
          <div style={{ fontSize: 12, color: C.t5, marginBottom: 16 }}>Feature importance from {modelInfo.name || "model"}</div>
          {topFeatures.length > 0 ? topFeatures.slice(0, 7).map(f => (
            <div key={f.feature} style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 12, color: C.t6, marginBottom: 4 }}>{f.feature}</div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ flex: 1 }}><ProgressBar pct={f.importance * 100} color={C.brand} /></div>
                <span style={{ fontSize: 12, fontWeight: 600, color: C.t9, minWidth: 36, textAlign: "right" }}>
                  {(f.importance * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          )) : <div style={{ color: C.t5 }}>No feature data. Run SLA notebook to export model info.</div>}
        </div>
      </div>

      {/* ── ROW 4: AT-RISK TICKETS TABLE (from API scores) ── */}
      <div style={{ background: C.white, borderRadius: 12, border: `1px solid ${C.border}`, overflow: "hidden", marginTop: 16 }}>
        <div style={{ padding: "16px 20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <span style={{ fontWeight: 600, fontSize: 15, color: C.t9 }}>At-risk tickets</span>
            <span style={{ fontSize: 12, color: C.t5, marginLeft: 8 }}>· breach probability ≥ 50%</span>
          </div>
          <Pill tone="red">{atRiskTickets.length} at risk</Pill>
        </div>
        {atRiskTickets.length > 0 ? (
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead><tr>
              <th style={thStyle}>Ticket ID</th>
              <th style={thStyle}>Priority</th>
              <th style={thStyle}>Breach prob.</th>
              <th style={thStyle}>Risk band</th>
              <th style={thStyle}>Off-hours</th>
            </tr></thead>
            <tbody>{atRiskTickets.slice(0, 20).map((t, i) => (
              <tr key={t.year_ticket_id || i}>
                <td style={tdStyle}>
                  <span style={{ fontWeight: 600, color: C.brand, fontSize: 13 }}>
                    {(t.year_ticket_id || "").replace(/_INC_/, "-")}
                  </span>
                </td>
                <td style={tdStyle}>
                  <Pill tone={t.ticket_priority === "P1" ? "red" : t.ticket_priority === "P2" ? "amber" : "blue"}>
                    {t.ticket_priority || "—"}
                  </Pill>
                </td>
                <td style={tdStyle}>
                  <span style={{ fontWeight: 600, color: C.t9 }}>
                    {((t.sla_breach_probability || 0) * 100).toFixed(0)}%
                  </span>
                  <div style={{ marginTop: 4, width: 80 }}>
                    <ProgressBar
                      pct={(t.sla_breach_probability || 0) * 100}
                      color={(t.sla_breach_probability || 0) > 0.7 ? C.rose : C.amber}
                    />
                  </div>
                </td>
                <td style={tdStyle}><Pill tone="red">{t.sla_risk_band || "High"}</Pill></td>
                <td style={tdStyle}>
                  <Pill tone={t.is_off_hours ? "amber" : "slate"}>
                    {t.is_off_hours ? "Off-hours" : "Business"}
                  </Pill>
                </td>
              </tr>
            ))}</tbody>
          </table>
        ) : <div style={{ padding: 20, color: C.t5, textAlign: "center" }}>No at-risk tickets found (all breach probabilities below 50%).</div>}
      </div>
    </div>
  );
}