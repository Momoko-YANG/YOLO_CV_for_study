# -*- coding: utf-8 -*-
"""
Detection business logic — mirrors System_noLogin.py media-handling methods.
Owns a singleton YOLOv8v5Detector; exposes frame-level and file-level APIs
consumed by the detection router.
"""
import time
import cv2
import numpy as np
from pathlib import Path

from algorithm.detector import YOLOv8v5Detector, count_classes
from algorithm.label_names import LABEL_LIST

# ---------------------------------------------------------------------------
# Singleton detector — loaded once at startup
# ---------------------------------------------------------------------------
_detector: YOLOv8v5Detector | None = None


def load_detector(model_path: str) -> None:
    global _detector
    _detector = YOLOv8v5Detector()
    _detector.load_model(model_path)


def get_detector() -> YOLOv8v5Detector:
    if _detector is None:
        raise RuntimeError("Detector not initialised — call load_detector() first.")
    return _detector


# ---------------------------------------------------------------------------
# Core inference helper
# ---------------------------------------------------------------------------
def _run_inference(img: np.ndarray) -> dict:
    det = get_detector()
    pre = det.preprocess(img)
    t0 = time.time()
    pred, heatmap = det.predict(pre)
    elapsed_ms = (time.time() - t0) * 1000

    det_info = det.postprocess(pred)
    counts = dict(zip(LABEL_LIST, count_classes(det_info, LABEL_LIST)))

    return {
        "detections": det_info,
        "class_counts": counts,
        "inference_ms": round(elapsed_ms, 2),
        "heatmap": heatmap,   # ndarray or None — router decides what to do
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect_image(image_bytes: bytes) -> dict:
    """Detect objects in a single uploaded image."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes.")
    result = _run_inference(img)

    # Encode annotated image back to JPEG bytes
    annotated = _annotate(img, result["detections"])
    _, buf = cv2.imencode(".jpg", annotated)
    result["annotated_image_bytes"] = buf.tobytes()
    return result


def detect_video_frame(frame: np.ndarray) -> dict:
    """Detect objects in a single video frame (used for streaming)."""
    return _run_inference(frame)


# ---------------------------------------------------------------------------
# Annotation helper (draws bounding boxes — no Qt needed)
# ---------------------------------------------------------------------------
def _annotate(img: np.ndarray, detections: list[dict]) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} {det['score']:.0%}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 100), 2)
        cv2.putText(img, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 2)
    return img
