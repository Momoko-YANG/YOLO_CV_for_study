/**
 * StatsChart — mirrors the count-per-class bar in Recognition_UI.ui
 * Uses a simple CSS-based bar chart (no extra dependencies).
 */
interface Props {
  classCounts: Record<string, number>;
}

export default function StatsChart({ classCounts }: Props) {
  const entries = Object.entries(classCounts);
  const max = Math.max(...entries.map(([, v]) => v), 1);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {entries.map(([name, count]) => (
        <div key={name} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 70, fontSize: 12, color: "#555", textAlign: "right" }}>{name}</span>
          <div style={{
            flex: 1, height: 18, background: "#eee", borderRadius: 4, overflow: "hidden",
          }}>
            <div style={{
              width: `${(count / max) * 100}%`,
              height: "100%",
              background: "#4796f0",
              transition: "width .3s",
            }} />
          </div>
          <span style={{ width: 24, fontSize: 12, color: "#333" }}>{count}</span>
        </div>
      ))}
      {entries.length === 0 && (
        <p style={{ color: "#aaa", fontSize: 13 }}>No data</p>
      )}
    </div>
  );
}
