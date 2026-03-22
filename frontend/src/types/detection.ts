export interface DetectionResult {
  class_name: string;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  score: number;
  class_id: number;
}

export interface WebSocketFrame {
  detections: DetectionResult[];
  inference_time: number;
  frame_id: number;
}

export interface ImageDetectionResponse {
  detections: DetectionResult[];
  inference_time: number;
  image_url: string;
  heatmap_url?: string;
}

export interface VideoTask {
  task_id: string;
  status: 'processing' | 'completed' | 'failed';
  progress: number;
  total_frames: number;
  processed_frames: number;
}

export interface VideoResults {
  task_id: string;
  total_frames: number;
  frames: DetectionResult[][];
}
