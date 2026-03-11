/**
 * ResultsTable — mirrors the QTableWidget in Recognition_UI.ui
 * Shows per-detection rows: class, bbox, confidence
 */
import type { DetectionBox } from "../../types";
import "./ResultsTable.css";

interface Props {
  detections: DetectionBox[];
  inferenceMs?: number;
}

export default function ResultsTable({ detections, inferenceMs }: Props) {
  return (
    <div className="results-table-wrap">
      {inferenceMs !== undefined && (
        <p className="infer-time">Inference: {inferenceMs.toFixed(1)} ms</p>
      )}
      <table className="results-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Class</th>
            <th>BBox (x1,y1,x2,y2)</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          {detections.length === 0 && (
            <tr><td colSpan={4} className="empty">No detections</td></tr>
          )}
          {detections.map((d, i) => (
            <tr key={i}>
              <td>{i + 1}</td>
              <td>{d.class_name}</td>
              <td>[{d.bbox.join(", ")}]</td>
              <td>{(d.score * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
