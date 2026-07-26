// src/pages/Maintenance.jsx — REAL API VERSION
import { useApi } from "../hooks/useApi";
import { Cpu, Database, Thermometer, AlertTriangle, HardDrive, Sparkles, Sliders } from "lucide-react";
import { ResponsiveContainer, AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip } from "recharts";
import { C } from "../utils/tokens";
import PageHeader from "../components/ui/PageHeader";
import ChartCard from "../components/ui/ChartCard";
import Pill from "../components/ui/Pill";
import ProgressBar from "../components/ui/ProgressBar";

const btn = (bg, color, border) => ({ display: "inline-flex", alignItems: "center", gap: 6, padding: "8px 14px", borderRadius: 8, fontSize: 14, fontWeight: 500, cursor: "pointer", background: bg || C.white, color: color || C.t7, border: `1px solid ${border || C.border}` });
const thStyle = { textAlign: "left", fontWeight: 500, fontSize: 11, color: C.t6, textTransform: "uppercase", letterSpacing: 0.5, padding: "10px 16px", background: C.bg, borderBottom: `1px solid ${C.border}` };
const tdStyle = { padding: "12px 16px", borderBottom: `1px solid ${C.borderLight}`, fontSize: 14, color: C.t7 };

// Static alerts — keep locally until API is extended
const system_alerts = [
  { alert_type: "FAILURE RISK", asset_id: "AID_053", body: "PSU degradation detected. Estimated failure in 72 hours." },
  { alert_type: "THERMAL ANOMALY", asset_id: "AID_041", body: "Core temp exceeding threshold by 12%. Inspect cooling path." },
  { alert_type: "STORAGE WEAR", asset_id: "AID_099", body: "SSD lifespan at 92%. Schedule replacement in Q4." },
];

export default function Maintenance() {
  const { data, loading, error } = useApi("/api/maintenance");
  if (loading) return <div style={{ padding: 40, textAlign: "center", color: C.t5 }}>Loading maintenance data...</div>;
  if (error) return <div style={{ padding: 40, textAlign: "center", color: C.red }}>Error: {error.message}</div>;

  const assets = data?.asset_risk_scores || [];
  const tel = data?.telemetry_agg || {};
  const kpis = tel?.fleet_kpis || { avg_cpu_pct: 0, avg_memory_pct: 0, avg_temp_pct: 0, critical_alerts: 0 };
  const cpuData = tel?.cpu_load_trend || [];
  const ramData = tel?.ram_trend || [];
  const thermalData = tel?.thermal_trend || [];
  const ranked = [...assets].sort((a, b) => (b.predicted_risk_probability || 0) - (a.predicted_risk_probability || 0));

  const TelCard = ({ title, subtitle, chartData, color, gradId, icon }) => (
    <ChartCard title={title} subtitle={subtitle} height={160} right={icon}>
      <ResponsiveContainer><AreaChart data={chartData}>
        <defs><linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={color} stopOpacity={0.45} /><stop offset="100%" stopColor={color} stopOpacity={0.05} /></linearGradient></defs>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
        <XAxis dataKey="time" tickLine={false} axisLine={false} tick={{ fill: "#94A3B8", fontSize: 11 }} />
        <YAxis domain={[0, 100]} hide /><Tooltip />
        <Area type="monotone" dataKey="value" stroke={color} strokeWidth={2.5} fill={`url(#${gradId})`} />
      </AreaChart></ResponsiveContainer>
    </ChartCard>
  );

  return (
    <div>
      <PageHeader title="Predictive Maintenance" subtitle="Real-time infrastructure health monitoring and AI-driven failure prediction." />

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
        {[{ icon: Cpu, label: "Fleet CPU Health", value: `${kpis.avg_cpu_pct || kpis.avg_cpu || 0}%`, color: C.brand, sub: "Optimal operating state" },
          { icon: Database, label: "Memory Utilization", value: `${kpis.avg_memory_pct || kpis.avg_mem || 0}%`, color: C.teal, sub: "Above normal levels" },
          { icon: Thermometer, label: "Chassis Temperature", value: `${kpis.avg_temp_pct || kpis.avg_temp || 0}°`, color: C.amber, sub: "Optimal operating state" }
        ].map(k => (
          <div key={k.label} style={{ background: C.white, borderRadius: 12, border: `1px solid ${C.border}`, padding: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}><div style={{ width: 40, height: 40, borderRadius: 8, background: C.brandLight, display: "flex", alignItems: "center", justifyContent: "center" }}><k.icon size={18} color={k.color} /></div><span style={{ fontSize: 20, fontWeight: 700, color: C.t9 }}>{k.value}</span></div>
            <div style={{ fontSize: 14, fontWeight: 500, color: C.t9 }}>{k.label}</div>
            <div style={{ marginTop: 8 }}><ProgressBar pct={parseInt(k.value) || 0} color={k.color} /></div>
            <div style={{ fontSize: 12, color: C.t5, marginTop: 4 }}>{k.sub}</div>
          </div>
        ))}
        <div style={{ background: C.white, borderRadius: 12, border: `2px solid ${C.roseBg}`, padding: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}><div style={{ width: 40, height: 40, borderRadius: 8, background: C.roseBg, display: "flex", alignItems: "center", justifyContent: "center" }}><AlertTriangle size={18} color={C.rose} /></div><Pill tone="red">{kpis.critical_alerts || kpis.critical || 0} Critical</Pill></div>
          <div style={{ fontSize: 14, fontWeight: 500, color: C.t9 }}>Active Predictive Alerts</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: C.t9, marginTop: 4 }}>High Risk</div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginTop: 16 }}>
        <TelCard title="CPU Load Trends" subtitle="Avg compute across clusters" chartData={cpuData} color="#3B82F6" gradId="grad-cpu" icon={<Cpu size={14} color="#3B82F6" />} />
        <TelCard title="RAM Consumption" subtitle="System-wide memory pressure" chartData={ramData} color="#14B8A6" gradId="grad-ram" icon={<Database size={14} color="#14B8A6" />} />
        <TelCard title="Thermal Telemetry" subtitle="Server rack environment heat" chartData={thermalData} color="#F59E0B" gradId="grad-thermal" icon={<Thermometer size={14} color="#F59E0B" />} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginTop: 16 }}>
        <div style={{ background: C.white, borderRadius: 12, border: `1px solid ${C.border}`, overflow: "hidden" }}>
          <div style={{ padding: "16px 20px", fontWeight: 600, fontSize: 15, color: C.t9 }}>Asset Health Status</div>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead><tr><th style={thStyle}>Device ID</th><th style={thStyle}>Model</th><th style={thStyle}>Risk Score</th><th style={thStyle}>Days Left</th><th style={thStyle}>Replace By</th><th style={thStyle}>Lead Time</th></tr></thead>
            <tbody>{ranked.map((a, i) => { const pct = Math.round((a.predicted_risk_probability || 0) * 100); const tone = pct > 70 ? "red" : pct > 40 ? "amber" : pct > 15 ? "blue" : "green"; const urgent = (a.predicted_remaining_days_to_failure || 999) <= (a.i_lead_time_days || 7); return (
              <tr key={a.asset_id || i}><td style={{ ...tdStyle, fontWeight: 600, color: C.t9 }}><div style={{ display: "flex", alignItems: "center", gap: 6 }}><HardDrive size={14} color={C.t4} />{a.asset_id}</div></td><td style={tdStyle}>{a.model_number || a.device_type}</td><td style={tdStyle}><Pill tone={tone}>{pct}</Pill></td><td style={{ ...tdStyle, fontWeight: 600, color: urgent ? C.rose : C.t7 }}>{a.predicted_remaining_days_to_failure || "—"}d</td><td style={{ ...tdStyle, fontSize: 12 }}>{a.replacement_needed_by_date || "—"}</td><td style={tdStyle}><span style={{ fontSize: 12 }}>{a.i_lead_time_days || "—"}d</span>{urgent && <> <Pill tone="red">At risk</Pill></>}</td></tr>
            ); })}</tbody>
          </table>
        </div>
        <div>
          <div style={{ background: C.white, borderRadius: 12, border: `1px solid ${C.border}`, padding: 20 }}>
            <div style={{ fontSize: 15, fontWeight: 600, color: C.t9, marginBottom: 12 }}>System Alerts</div>
            {system_alerts.map((al, i) => (<div key={i} style={{ borderLeft: `3px solid ${C.rose}`, paddingLeft: 10, marginBottom: 10 }}><div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}><span style={{ fontWeight: 600, color: C.t9, textTransform: "uppercase", letterSpacing: 0.5 }}>{al.alert_type}</span><span style={{ color: C.t4 }}>{al.asset_id}</span></div><div style={{ fontSize: 13, color: C.t6, marginTop: 2 }}>{al.body}</div></div>))}
          </div>
          {ranked.length > 0 && (
            <div style={{ background: `linear-gradient(135deg, ${C.brandLight}, #fff)`, borderRadius: 12, border: `1px solid ${C.border}`, padding: 20, marginTop: 16 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}><Sparkles size={14} color={C.brand} /><span style={{ fontSize: 14, fontWeight: 600, color: C.t9 }}>AI Maintenance Scheduler</span></div>
              <div style={{ background: C.white, borderRadius: 8, padding: 10, border: `1px solid ${C.border}`, marginBottom: 10 }}>
                <div style={{ fontSize: 12, color: C.t5 }}>Next Recommended Action</div>
                <div style={{ fontSize: 14, fontWeight: 600, color: C.t9, marginTop: 4 }}>Cluster A Reboot · {ranked[0].asset_id}</div>
                <div style={{ display: "flex", gap: 6, marginTop: 6 }}><Pill tone="blue">Risk: {Math.round((ranked[0].predicted_risk_probability || 0) * 100)}%</Pill><Pill tone="amber">{ranked[0].predicted_remaining_days_to_failure}d left</Pill></div>
              </div>
              <button style={{ ...btn(C.brand, "#fff", C.brand), width: "100%", justifyContent: "center" }}
                onClick={async () => {
                  const assetId = ranked[0]?.asset_id;
                  if (!assetId) { alert("No assets to schedule."); return; }
                  try {
                    const res = await fetch("http://localhost:8000/api/maintenance/schedule", {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ asset_id: assetId, action: "reboot_and_patch" }),
                    });
                    const result = await res.json();
                    if (result.success) {
                      alert(`✅ Maintenance scheduled!\n\nAsset: ${result.asset_id}\nAction: ${result.action}\nWindow: ${result.scheduled_window}\nRisk: ${result.risk_pct}%\n\nLogged to maintenance_schedule.json`);
                      } else {
                        alert(`❌ Failed: ${result.error}`);
                        }
                      } catch (err) { alert(`❌ Network error: ${err.message}`); }
                      }}>
                <Sliders size={14} /> Auto-Schedule Maintenance
                </button>
                </div>
                )}
                </div>
                </div>
                </div>
                );
                }
