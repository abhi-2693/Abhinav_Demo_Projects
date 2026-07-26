// src/pages/Dashboard.jsx — with working Export button
import { useApi } from "../hooks/useApi";
import PageHeader from "../components/ui/PageHeader";
import KPICard from "../components/ui/KPICard";
import ChartCard from "../components/ui/ChartCard";
import Pill from "../components/ui/Pill";
import { C } from "../utils/tokens";
import { Ticket, Activity, AlertCircle, Clock, Download, AlertTriangle, CheckCircle2, ChevronRight } from "lucide-react";
import { ResponsiveContainer, ComposedChart, Area, Line, BarChart, Bar, LineChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from "recharts";

const btn = (bg, color, border) => ({ display: "inline-flex", alignItems: "center", gap: 6, padding: "8px 14px", borderRadius: 8, fontSize: 14, fontWeight: 500, cursor: "pointer", background: bg || C.white, color: color || C.t7, border: `1px solid ${border || C.border}` });

// ── Export handler — downloads a CSV from the API ──
const handleExport = () => {
  window.open("http://localhost:8000/api/export/tickets", "_blank");
};

export default function Dashboard() {
  const { data, loading, error } = useApi("/api/dashboard");
  if (loading) return <div style={{ padding: 40, textAlign: "center", color: C.t5 }}>Loading dashboard data...</div>;
  if (error) return <div style={{ padding: 40, textAlign: "center", color: C.red }}>Error: {error.message}. Make sure FastAPI is running on port 8000.</div>;

  const volume = data?.ticket_volume_trend || [];
  const compliance = data?.sla_compliance_trend || [];
  const downtime = data?.system_downtime || [];
  const insights = data?.active_insights || [];
  const activeCount = data?.kpis?.total_active || 0;
  const meanBreach = data?.kpis?.mean_breach || 0;

  return (
    <div>
      <PageHeader title="Operational Overview" subtitle="Real-time health monitoring and predictive maintenance analytics."
        right={<>
          <button style={btn()} onClick={() => alert("Time filter: 24h selected")}><Clock size={14} /> 24h</button>
          <button style={btn(C.brand, "#fff", C.brand)} onClick={handleExport}><Download size={14} /> Export Report</button>
        </>} />

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
        <KPICard icon={Ticket} label="Total Active Tickets" value={activeCount.toLocaleString()} sub="Live system load" trend="+12.5%" trendDir="up" />
        <KPICard icon={Activity} label="Resolution Velocity" value={data?.kpis?.resolution_velocity || "—"} sub="Open vs. Closed ratio" trend="+4.2%" trendDir="up" />
        <KPICard icon={AlertCircle} label="SLA Breach Risk" value={`${(meanBreach * 100).toFixed(1)}%`} sub="Predicted compliance" trend="-0.8%" trendDir="down" />
        <KPICard icon={Clock} label="Avg. Resolution Time" value={data?.kpis?.avg_resolution_time || "—"} sub="From model predictions" trend="-15m" trendDir="down" />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginTop: 16 }}>
        <ChartCard title="Ticket Volume Trends" subtitle="Daily intake vs resolution" right={<Pill tone="blue">Real-time</Pill>}>
          {volume.length > 0 ? (
            <ResponsiveContainer><ComposedChart data={volume}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
              <XAxis dataKey="day" tickLine={false} axisLine={false} tick={{ fill: "#6B7280", fontSize: 12 }} />
              <YAxis tickLine={false} axisLine={false} tick={{ fill: "#6B7280", fontSize: 12 }} />
              <Tooltip /><Legend />
              <Area type="monotone" dataKey="total" name="Total Tickets" stroke={C.brand} fill={C.brandLight} strokeWidth={2} />
              <Line type="monotone" dataKey="resolved" name="Resolved" stroke={C.t9} strokeWidth={2} dot={{ r: 3 }} />
            </ComposedChart></ResponsiveContainer>
          ) : <div style={{ color: C.t5, padding: 20, textAlign: "center" }}>No ticket volume data available. Run Ticket_03 notebook.</div>}
        </ChartCard>
        <ChartCard title="System Downtime" subtitle="Minutes per core service (Today)">
          {downtime.length > 0 ? (
            <ResponsiveContainer><BarChart data={downtime} layout="vertical" margin={{ left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E5E7EB" />
              <XAxis type="number" tickLine={false} axisLine={false} tick={{ fill: "#6B7280", fontSize: 12 }} />
              <YAxis dataKey="service" type="category" tickLine={false} axisLine={false} tick={{ fill: "#6B7280", fontSize: 12 }} />
              <Tooltip /><Bar dataKey="minutes" fill={C.brand} radius={[0, 4, 4, 0]} />
            </BarChart></ResponsiveContainer>
          ) : <div style={{ color: C.t5, padding: 20, textAlign: "center" }}>No downtime data available.</div>}
        </ChartCard>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginTop: 16 }}>
        <ChartCard title="SLA Compliance Trends" subtitle="Historical compliance across last 6 reporting periods">
          {compliance.length > 0 ? (
            <ResponsiveContainer><LineChart data={compliance}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
              <XAxis dataKey="month" tickLine={false} axisLine={false} tick={{ fill: "#6B7280", fontSize: 12 }} />
              <YAxis domain={["auto", "auto"]} tickLine={false} axisLine={false} tick={{ fill: "#6B7280", fontSize: 12 }} />
              <Tooltip /><Line type="stepAfter" dataKey="compliance" stroke="#06B6D4" strokeWidth={2.5} dot={{ r: 4, fill: "#06B6D4" }} />
            </LineChart></ResponsiveContainer>
          ) : <div style={{ color: C.t5, padding: 20, textAlign: "center" }}>No SLA compliance data. Run SLA notebook.</div>}
        </ChartCard>
        <div style={{ background: C.white, borderRadius: 12, border: `1px solid ${C.border}`, padding: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <div style={{ fontSize: 15, fontWeight: 600, color: C.t9 }}>Active AI Insights</div>
            <Pill tone="blue">{insights.length} New</Pill>
          </div>
          <div style={{ maxHeight: 260, overflowY: "auto" }}>
            {insights.length > 0 ? insights.map((ins, i) => (
              <div key={ins.id || i} style={{ border: `1px solid ${C.border}`, borderRadius: 8, padding: 10, marginBottom: 8, cursor: "pointer" }}>
                <div style={{ display: "flex", gap: 8 }}>
                  <div style={{ marginTop: 2 }}>{ins.severity === "critical" ? <AlertCircle size={15} color={C.rose} /> : ins.severity === "warning" ? <AlertTriangle size={15} color={C.amber} /> : <CheckCircle2 size={15} color={C.green} />}</div>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", justifyContent: "space-between" }}><span style={{ fontSize: 13, fontWeight: 600, color: C.t9 }}>{ins.title}</span><span style={{ fontSize: 11, color: C.t4 }}>{ins.time}</span></div>
                    <div style={{ fontSize: 12, color: C.t6, marginTop: 4, lineHeight: 1.4 }}>{ins.body}</div>
                    <div style={{ fontSize: 11, color: C.t4, marginTop: 4, fontStyle: "italic" }}>{ins.source}</div>
                    <div style={{ fontSize: 12, color: C.brand, fontWeight: 500, marginTop: 6, display: "flex", alignItems: "center", gap: 2, cursor: "pointer" }}>Take Action <ChevronRight size={12} /></div>
                  </div>
                </div>
              </div>
            )) : <div style={{ color: C.t5, padding: 16, textAlign: "center" }}>No insights available. Run model notebooks to generate predictions.</div>}
          </div>
        </div>
      </div>
    </div>
  );
}
