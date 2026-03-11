# -*- coding: utf-8 -*-
"""Pydantic schemas for detection endpoints."""
from pydantic import BaseModel


class DetectionBox(BaseModel):
    class_name: str
    bbox: list[int]   # [x1, y1, x2, y2]
    score: float
    class_id: int


class DetectionResponse(BaseModel):
    detections: list[DetectionBox]
    class_counts: dict[str, int]
    inference_ms: float


class DetectorParams(BaseModel):
    conf: float = 0.25
    iou: float = 0.5
    classes: list[int] | None = None
