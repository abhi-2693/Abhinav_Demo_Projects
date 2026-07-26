import { C } from "../../utils/tokens";
 
export default function KPICard({
  icon: Icon,     // a lucide-react icon component
  label,          // "Total active tickets"
  value,          // "1,284"
  sub,            // "Increased volume in Cloud Infrastructure"
  trend,          // "+12.5%"
  trendDir,       // "up" or "down"
}) {
  const trendColor =
    trendDir === "up"   ? C.green :
    trendDir === "down" ? C.red : C.t5;
 
  return (
    <div style={{
      background: C.white, borderRadius: 12,
      border: `1px solid ${C.border}`, padding: 20,
    }}>
      <div style={{
        display: "flex", justifyContent: "space-between",
        alignItems: "center", marginBottom: 12,
      }}>
        <div style={{
          width: 40, height: 40, borderRadius: 8,
          background: C.borderLight, display: "flex",
          alignItems: "center", justifyContent: "center",
        }}>
          <Icon size={18} color={C.t7} />
        </div>
        {trend && (
          <span style={{ fontSize: 12, fontWeight: 600, color: trendColor }}>
            {trend}
          </span>
        )}
      </div>
      <div style={{ fontSize: 26, fontWeight: 700, color: C.t9 }}>{value}</div>
      <div style={{ fontSize: 14, fontWeight: 500, color: C.t6, marginTop: 4 }}>{label}</div>
      {sub && <div style={{ fontSize: 12, color: C.t5, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}