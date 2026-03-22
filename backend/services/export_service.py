import csv
import os
import time
from typing import List

import cv2
import numpy as np

from core.config import settings
from models.schemas import DetectionResult


def save_annotated_image(image: np.ndarray, detections: List[DetectionResult]) -> str:
    """Draw detections on image and save to disk. Returns file path."""
    annotated = draw_detections(image, detections)
    os.makedirs(settings.export_dir, exist_ok=True)
    filename = f"pic_{int(time.time())}.png"
    filepath = os.path.join(settings.export_dir, filename)
    cv2.imwrite(filepath, annotated)
    return filepath


def save_results_csv(
    detections_list: List[List[DetectionResult]],
    source_path: str = "",
) -> str:
    """Save detection results as CSV. Returns file path."""
    os.makedirs(settings.export_dir, exist_ok=True)
    filename = f"results_{int(time.time())}.csv"
    filepath = os.path.join(settings.export_dir, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Source", "Class", "BBox", "Confidence"])
        for frame_idx, dets in enumerate(detections_list):
            for det in dets:
                writer.writerow([
                    frame_idx,
                    source_path,
                    det.class_name,
                    str(det.bbox),
                    f"{det.score:.4f}",
                ])
    return filepath


def draw_detections(image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
    """Draw bounding boxes on image (OpenCV)."""
    img = image.copy()
    colors = [
        (255, 76, 76),   # Red
        (76, 175, 80),   # Green
        (33, 150, 243),  # Blue
    ]
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = colors[det.class_id % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.score:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img
