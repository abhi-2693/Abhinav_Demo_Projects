// src/pages/Inventory.jsx — REAL API VERSION
import { useApi } from "../hooks/useApi";
import { Package, CheckCircle2, DollarSign, AlertTriangle, Download, Filter, Plus, Clock, Sparkles } from "lucide-react";
import { ResponsiveContainer, BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from "recharts";
import { C } from "../utils/tokens";
import PageHeader from "../components/ui/PageHeader";
import KPICard from "../components/ui/KPICard";
import ChartCard from "../components/ui/ChartCard";
import Pill from "../components/ui/Pill";
import ProgressBar from "../components/ui/ProgressBar";
import DataPatterns from "../components/DataPatterns";

const btn = (bg, color, border) => ({ display: "inline-flex", alignItems: "center", gap: 6, padding: "8px 14px", borderRadius: 8, fontSize: 14, fontWeight: 500, cursor: "pointer", background: bg || C.white, color: color || C.t7, border: `1px solid ${border || C.border}` });
const thStyle = { textAlign: "left", fontWeight: 500, fontSize: 11, color: C.t6, textTransform: "uppercase", letterSpacing: 0.5, padding: "10px 16px", background: C.bg, borderBottom: `1px solid ${C.border}` };
const tdStyle = { padding: "12px 16px", borderBottom: `1px solid ${C.borderLight}`, fontSize: 14, color: C.t7 };
const deriveUrgency = (r) => !r.to_order_flag ? "Optimal" : (r.lead_time_days || 0) >= 14 ? "Urgent" : "Planning";

// Category stock — keep locally until API is extended
const category_stock = [{ category: "Laptops", current: 48, threshold: 75 }, { category: "Monitors", current: 82, threshold: 35 }, { category: "Network", current: 18, threshold: 45 }, { category: "Accessories", current: 40, threshold: 62 }];

export default function Inventory() {
  const { data, loading, error } = useApi("/api/inventory");
  if (loading) return <div style={{ padding: 40, textAlign: "center", color: C.t5 }}>Loading inventory data...</div>;
  if (error) return <div style={{ padding: 40, textAlign: "center", color: C.red }}>Error: {error.message}</div>;

  const plan = data?.procurement_plan || [];
  const kpis = data?.inventory_kpis || { total_assets: 0, stock_health_pct: 0, procurement_cost_mtd: 0, critical_lows: 0 };

  return (
    <div>
      <PageHeader title="Inventory Optimization" subtitle="AI-driven stock management and procurement recommendations."
        right={<><button style={btn()}><Download size={14} /> Export</button><button style={btn()}><Filter size={14} /> Filter</button><button style={btn(C.brand, "#fff", C.brand)}><Plus size={14} /> New Asset</button></>} />

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
        <KPICard icon={Package} label="Total IT Assets" value={(kpis.total_assets || 0).toLocaleString()} sub="+4.2% across 12 categories" />
        <KPICard icon={CheckCircle2} label="Stock Health" value={`${kpis.stock_health_pct || 0}%`} sub="Items above threshold" trend="-1.5%" trendDir="down" />
        <KPICard icon={DollarSign} label="Procurement Cost" value={`$${((kpis.procurement_cost_mtd || 0) / 1000).toFixed(2)}K`} sub="Month-to-Date" trend="+12.8%" trendDir="up" />
        <KPICard icon={AlertTriangle} label="Critical Lows" value={String(kpis.critical_lows || 0)} sub="Urgent reorders required" />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginTop: 16 }}>
        <ChartCard title="Category Stock Levels" subtitle="Current stock vs. defined safety thresholds" height={240}>
          <ResponsiveContainer><BarChart data={category_stock} layout="vertical" margin={{ left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E5E7EB" />
            <XAxis type="number" tickLine={false} axisLine={false} tick={{ fill: "#6B7280", fontSize: 12 }} />
            <YAxis dataKey="category" type="category" tickLine={false} axisLine={false} tick={{ fill: "#6B7280", fontSize: 12 }} />
            <Tooltip /><Legend />
            <Bar dataKey="current" name="Current Stock" fill={C.brand} radius={[0, 4, 4, 0]} />
            <Bar dataKey="threshold" name="Reorder Threshold" fill={C.t7} radius={[0, 4, 4, 0]} />
          </BarChart></ResponsiveContainer>
        </ChartCard>
        <div style={{ background: C.white, borderRadius: 12, border: `1px solid ${C.border}`, padding: 20 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}><Sparkles size={16} color={C.brand} /><span style={{ fontSize: 15, fontWeight: 600, color: C.t9 }}>AI Reorder Insights</span></div>
          {plan.filter(p => p.to_order_flag).slice(0, 2).map((r, i) => { const urg = deriveUrgency(r); return (
            <div key={r.asset_id || i} style={{ border: `1px solid ${C.border}`, borderRadius: 8, padding: 10, marginBottom: 8 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}><Pill tone={urg === "Urgent" ? "red" : "blue"}>{urg}</Pill><span style={{ fontSize: 11, color: C.t5 }}>Lead: {r.lead_time_days}d</span></div>
              <div style={{ fontSize: 14, fontWeight: 600, color: C.t9, marginBottom: 4 }}>{r.model_number || r.device_type}</div>
              <div style={{ fontSize: 12, color: C.t6, lineHeight: 1.5 }}>Stock: <strong>{r.current_stock_quantity}</strong>, threshold: <strong>{r.reorder_threshold_quantity}</strong>. Order <strong>{r.to_order_quantity}</strong> units.</div>
              <div style={{ fontSize: 12, color: C.brand, fontWeight: 500, marginTop: 6, cursor: "pointer" }}
        onClick={async () => {
        const res = await fetch("http://localhost:8000/api/inventory/approve", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ asset_id: r.asset_id, model_number: r.model_number, quantity: r.to_order_quantity }),
        });
        const result = await res.json();
        if (result.success) alert(`✅ Approved!\n\n${result.model_number || result.asset_id}\nQuantity: ${result.quantity} units\nApproval ID: ${result.approval_id}`);
        else alert(`❌ Failed: ${result.error}`);
        }}>
        Approve Reorder →
        </div>
            </div>
          ); })}
        </div>
      </div>

      <div style={{ background: C.white, borderRadius: 12, border: `1px solid ${C.border}`, overflow: "hidden", marginTop: 16 }}>
        <div style={{ padding: "16px 20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}><div><span style={{ fontWeight: 600, fontSize: 15, color: C.t9 }}>Inventory Management</span></div><div style={{ display: "flex", gap: 6 }}><button style={{ ...btn(), fontSize: 12, padding: "6px 12px" }}>All Items</button><button style={{ ...btn(), fontSize: 12, padding: "6px 12px" }}>Low Stock</button></div></div>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead><tr><th style={thStyle}>Item & Category</th><th style={thStyle}>Current Level</th><th style={thStyle}>Usage Rate</th><th style={thStyle}>Lead Time</th><th style={thStyle}>AI Recommendation</th><th style={thStyle}>Actions</th></tr></thead>
          <tbody>{plan.map((it, i) => { const urg = deriveUrgency(it); const critical = (it.current_stock_quantity || 0) < (it.reorder_threshold_quantity || 1) * 0.5; const low = (it.current_stock_quantity || 0) < (it.reorder_threshold_quantity || 1); const pct = Math.min(100, ((it.current_stock_quantity || 0) / Math.max((it.reorder_threshold_quantity || 1) * 2, 1)) * 100); const usage = (it.asset_needed_within_next_leadtime || 0) / Math.max(it.current_stock_quantity || 1, 1); return (
            <tr key={it.asset_id || i}>
              <td style={tdStyle}><div style={{ fontWeight: 500, color: C.t9 }}>{it.model_number || it.device_type}</div><div style={{ fontSize: 12, color: C.t5 }}>{it.device_type}</div></td>
              <td style={tdStyle}><div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}><span style={{ fontWeight: 500, fontSize: 13 }}>{it.current_stock_quantity} / {it.reorder_threshold_quantity}</span><Pill tone={critical ? "red" : low ? "amber" : "green"}>{critical ? "Critical" : low ? "Low" : "Healthy"}</Pill></div><div style={{ width: 120 }}><ProgressBar pct={pct} color={critical ? C.rose : low ? C.amber : C.green} /></div></td>
              <td style={tdStyle}><Pill tone={usage > 0.3 ? "red" : usage > 0.1 ? "amber" : "slate"}>{usage > 0.3 ? "High" : usage > 0.1 ? "Moderate" : "Low"}</Pill></td>
              <td style={tdStyle}><div style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 13 }}><Clock size={12} color={C.t5} /> {it.lead_time_days} days</div></td>
              <td style={tdStyle}>{it.to_order_flag ? <Pill tone={urg === "Urgent" ? "red" : "blue"}>Reorder +{it.to_order_quantity}</Pill> : <Pill tone="green">Optimal</Pill>}</td>
              <td style={tdStyle}><div style={{ display: "flex", gap: 6 }}><button style={{ ...btn(), fontSize: 12, padding: "4px 10px" }}>Details</button>{it.to_order_flag && <button style={{ ...btn(C.brand, "#fff", C.brand), fontSize: 12, padding: "4px 10px" }}
        onClick={async () => {
        const res = await fetch("http://localhost:8000/api/inventory/approve", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ asset_id: it.asset_id, model_number: it.model_number, quantity: it.to_order_quantity }),
        });
        const result = await res.json();
          if (result.success) alert(`✅ Reorder approved!\n\n${result.model_number || result.asset_id}\nQuantity: ${result.quantity} units\nApproval ID: ${result.approval_id}\n\nLogged to approved_reorders.json`);
          else alert(`❌ Failed: ${result.error}`);
        }}>Approve</button>}</div></td>
            </tr>
          ); })}</tbody>
        </table>
      </div>
       <DataPatterns procurement={plan} />
    </div>
  );
}
