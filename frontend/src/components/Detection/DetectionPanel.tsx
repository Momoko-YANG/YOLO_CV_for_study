/**
 * DetectionPanel — mirrors the central detection area of Recognition_UI.ui
 * Supports: image upload | camera stream | video stream
 */
import { useRef, useState, useCallback } from "react";
import { detectionApi } from "../../services/api";
import type { DetectionResult, DetectorParams } from "../../types";
import "./DetectionPanel.css";

interface Props {
  onResult: (result: DetectionResult) => void;
}

type Mode = "image" | "camera" | "video";

export default function DetectionPanel({ onResult }: Props) {
  const [mode, setMode] = useState<Mode>("image");
  const [preview, setPreview] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [params, setParams] = useState<DetectorParams>({ conf: 0.25, iou: 0.5 });

  const wsRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // ── image upload ────────────────────────────────────────────────
  async function handleImageUpload(file: File) {
    setLoading(true);
    try {
      const result = await detectionApi.detectImage(file);
      if (result.image_b64) setPreview(`data:image/jpeg;base64,${result.image_b64}`);
      onResult(result);
    } finally {
      setLoading(false);
    }
  }

  // ── camera stream ───────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    streamRef.current = stream;
    if (videoRef.current) videoRef.current.srcObject = stream;

    const ws = detectionApi.openStream((result) => {
      if (result.image_b64) setPreview(`data:image/jpeg;base64,${result.image_b64}`);
      onResult(result);
    });
    wsRef.current = ws;

    ws.onopen = () => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video) return;
      canvas.width = 640; canvas.height = 480;
      const ctx = canvas.getContext("2d")!;

      const send = () => {
        if (ws.readyState !== WebSocket.OPEN) return;
        ctx.drawImage(video, 0, 0, 640, 480);
        canvas.toBlob((blob) => {
          if (blob) ws.send(blob);
          requestAnimationFrame(send);
        }, "image/jpeg", 0.8);
      };
      send();
    };
  }, [onResult]);

  function stopCamera() {
    wsRef.current?.close();
    streamRef.current?.getTracks().forEach(t => t.stop());
    wsRef.current = null;
    streamRef.current = null;
    setPreview("");
  }

  // ── param sliders ───────────────────────────────────────────────
  async function applyParams(next: DetectorParams) {
    setParams(next);
    await detectionApi.updateParams(next);
  }

  // ── render ──────────────────────────────────────────────────────
  return (
    <div className="detection-panel">
      {/* Mode selector */}
      <div className="mode-tabs">
        {(["image", "camera", "video"] as Mode[]).map(m => (
          <button key={m} className={mode === m ? "active" : ""}
            onClick={() => { stopCamera(); setMode(m); setPreview(""); }}>
            {m.charAt(0).toUpperCase() + m.slice(1)}
          </button>
        ))}
      </div>

      {/* Preview area */}
      <div className="preview-area">
        {preview
          ? <img src={preview} alt="detection result" />
          : <span className="placeholder">No input</span>
        }
      </div>

      {/* Hidden video + canvas for camera */}
      <video ref={videoRef} autoPlay muted hidden />
      <canvas ref={canvasRef} hidden />

      {/* Controls */}
      {mode === "image" && (
        <label className="upload-btn">
          {loading ? "Processing…" : "Select Image"}
          <input type="file" accept="image/*" hidden
            onChange={e => e.target.files?.[0] && handleImageUpload(e.target.files[0])} />
        </label>
      )}

      {mode === "camera" && (
        <div className="cam-controls">
          <button onClick={startCamera}>Start Camera</button>
          <button onClick={stopCamera}>Stop</button>
        </div>
      )}

      {/* Param sliders — mirrors the sliders in Recognition_UI.ui */}
      <div className="param-sliders">
        <label>
          Confidence: {params.conf.toFixed(2)}
          <input type="range" min={0.1} max={0.9} step={0.05} value={params.conf}
            onChange={e => applyParams({ ...params, conf: +e.target.value })} />
        </label>
        <label>
          IOU: {params.iou.toFixed(2)}
          <input type="range" min={0.1} max={0.9} step={0.05} value={params.iou}
            onChange={e => applyParams({ ...params, iou: +e.target.value })} />
        </label>
      </div>
    </div>
  );
}
