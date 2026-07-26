import { C } from "../../utils/tokens";
 
export default function ProgressBar({
  pct,                    // 0-100: how much to fill
  color = C.brand,        // fill colour (default: brand blue)
  height = 6,             // bar thickness in pixels
}) {
  const width = Math.min(100, Math.max(0, pct));
  return (
    <div style={{
      width: "100%", height, borderRadius: height,
      background: C.borderLight,
    }}>
      <div style={{
        width: `${width}%`, height, borderRadius: height,
        background: color,
      }} />
    </div>
  );
}