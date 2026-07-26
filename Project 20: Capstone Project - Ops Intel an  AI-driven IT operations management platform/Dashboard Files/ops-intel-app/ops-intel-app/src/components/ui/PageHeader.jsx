import { C } from "../../utils/tokens";
 
export default function PageHeader({ title, subtitle, right }) {
  return (
    <div style={{
      display: "flex", justifyContent: "space-between",
      alignItems: "center", marginBottom: 24,
    }}>
      <div>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: C.t9, margin: 0 }}>{title}</h1>
        {subtitle && <p style={{ fontSize: 14, color: C.t5, margin: "4px 0 0" }}>{subtitle}</p>}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>{right}</div>
    </div>
  );
}