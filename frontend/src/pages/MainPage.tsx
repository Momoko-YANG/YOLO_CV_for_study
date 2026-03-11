/**
 * MainPage — mirrors System_noLogin.py / System_login.py main window
 * Layout: Sidebar | DetectionPanel | ResultsPanel
 */
import { useState } from "react";
import Sidebar from "../components/Layout/Sidebar";
import DetectionPanel from "../components/Detection/DetectionPanel";
import ResultsTable from "../components/Results/ResultsTable";
import StatsChart from "../components/Results/StatsChart";
import type { DetectionResult, UserInfo } from "../types";
import "./MainPage.css";

interface Props {
  user: UserInfo;
  onLogout: () => void;
}

export default function MainPage({ user, onLogout }: Props) {
  const [navKey, setNavKey] = useState("image");
  const [result, setResult] = useState<DetectionResult | null>(null);

  return (
    <div className="main-layout">
      <Sidebar activeKey={navKey} onSelect={setNavKey} />

      <div className="main-content">
        {/* Header */}
        <header className="main-header">
          <h1>YOLO Gesture Recognition</h1>
          <div className="user-info">
            {user.avatar_path && (
              <img src={user.avatar_path} alt="avatar" className="avatar" />
            )}
            <span>{user.username}</span>
            <button className="logout-btn" onClick={onLogout}>Logout</button>
          </div>
        </header>

        <div className="main-body">
          {/* Detection panel */}
          <section className="panel-detection">
            <DetectionPanel onResult={setResult} />
          </section>

          {/* Results panel */}
          <aside className="panel-results">
            <h3>Detections</h3>
            <ResultsTable
              detections={result?.detections ?? []}
              inferenceMs={result?.inference_ms}
            />
            <h3 style={{ marginTop: 20 }}>Class counts</h3>
            <StatsChart classCounts={result?.class_counts ?? {}} />
          </aside>
        </div>
      </div>
    </div>
  );
}
