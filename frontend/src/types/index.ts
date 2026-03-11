// ── Auth ──────────────────────────────────────────────────────────────────
export interface LoginRequest {
  username: string;
  password: string;
}

export interface UserInfo {
  username: string;
  avatar_path: string;
}

// ── Detection ─────────────────────────────────────────────────────────────
export interface DetectionBox {
  class_name: string;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  score: number;
  class_id: number;
}

export interface DetectionResult {
  detections: DetectionBox[];
  class_counts: Record<string, number>;
  inference_ms: number;
  image_b64?: string; // annotated JPEG (base64) from /detect/image/annotated
}

// ── Detector params ───────────────────────────────────────────────────────
export interface DetectorParams {
  conf: number;
  iou: number;
  classes?: number[] | null;
}
