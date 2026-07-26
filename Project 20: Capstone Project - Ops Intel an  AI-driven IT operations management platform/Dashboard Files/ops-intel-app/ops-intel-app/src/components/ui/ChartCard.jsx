import { C } from "../../utils/tokens";
 
export default function ChartCard({
  title, subtitle, children,
  height = 220, right,
}) {
  return (
    <div style={{
      background: C.white, borderRadius: 12,
      border: `1px solid ${C.border}`, padding: 20,
    }}>
      <div style={{
        display: "flex", justifyContent: "space-between",
        alignItems: "center", marginBottom: 16,
      }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 600, color: C.t9 }}>{title}</div>
          {subtitle && <div style={{ fontSize: 12, color: C.t5, marginTop: 2 }}>{subtitle}</div>}
        </div>
        {right}
      </div>
      <div style={{ height }}>{children}</div>
    </div>
  );
}
