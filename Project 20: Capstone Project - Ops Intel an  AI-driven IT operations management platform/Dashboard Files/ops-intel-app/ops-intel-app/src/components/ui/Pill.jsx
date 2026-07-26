import { C } from "../../utils/tokens";
 
const tones = {
  red:   { bg: C.redBg,       color: C.redText },
  amber: { bg: C.amberBg,     color: C.amberText },
  green: { bg: C.greenBg,     color: C.greenText },
  blue:  { bg: C.blueBg,      color: C.blueText },
  slate: { bg: C.borderLight, color: C.t6 },
};
 
export default function Pill({ tone = "slate", children }) {
  const t = tones[tone] || tones.slate;
  return (
    <span style={{
      display: "inline-flex", alignItems: "center",
      padding: "2px 10px", borderRadius: 9999,
      fontSize: 12, fontWeight: 500,
      background: t.bg, color: t.color,
    }}>
      {children}
    </span>
  );
}