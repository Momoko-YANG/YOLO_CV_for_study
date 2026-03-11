/**
 * Sidebar — mirrors the collapsible Sidebar widget in Recognition_UI.ui
 */
import { useState } from "react";
import "./Sidebar.css";

interface NavItem { icon: string; label: string; key: string; }

const NAV_ITEMS: NavItem[] = [
  { icon: "🖼️", label: "Image",  key: "image"  },
  { icon: "📷", label: "Camera", key: "camera" },
  { icon: "🎬", label: "Video",  key: "video"  },
  { icon: "📊", label: "Stats",  key: "stats"  },
  { icon: "⚙️", label: "Settings", key: "settings" },
];

interface Props {
  activeKey: string;
  onSelect: (key: string) => void;
}

export default function Sidebar({ activeKey, onSelect }: Props) {
  const [expanded, setExpanded] = useState(false);

  return (
    <aside className={`sidebar ${expanded ? "expanded" : ""}`}>
      <button className="toggle-btn" onClick={() => setExpanded(v => !v)}>
        {expanded ? "◀" : "▶"}
      </button>
      <nav>
        {NAV_ITEMS.map(item => (
          <button
            key={item.key}
            className={`nav-item ${activeKey === item.key ? "active" : ""}`}
            onClick={() => onSelect(item.key)}
            title={item.label}
          >
            <span className="nav-icon">{item.icon}</span>
            {expanded && <span className="nav-label">{item.label}</span>}
          </button>
        ))}
      </nav>
    </aside>
  );
}
