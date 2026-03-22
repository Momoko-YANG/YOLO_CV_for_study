import asyncio
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core import settings
from models import DetectionResult

# Gesture label mapping
GESTURE_LABELS = {
    "Stop": {"en": "Stop", "zh": "停止"},
    "Understand": {"en": "Understand", "zh": "了解"},
    "NumberTwo": {"en": "Number Two", "zh": "数字2"},
}


class YOLOService:
    _instance: Optional["YOLOService"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.model = None
        self.device = None
        self.names: List[str] = []
        self.model_path: str = ""

    @classmethod
    def get_instance(cls) -> "YOLOService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, model_path: str, device: Optional[str] = None):
        import torch
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import select_device

        device = device or settings.device
        self.device = select_device(device)
        self.model = YOLO(model_path)
        self.model_path = model_path

        # Build class name list with Chinese mapping
        names_dict = self.model.names
        self.names = []
        for v in names_dict.values():
            if v in GESTURE_LABELS:
                self.names.append(GESTURE_LABELS[v]["zh"])
            else:
                self.names.append(v)

        # Warmup
        self.model(
            torch.zeros(1, 3, settings.image_size, settings.image_size)
            .to(self.device)
            .type_as(next(self.model.model.parameters()))
        )

    def detect(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.5,
    ) -> Tuple[List[DetectionResult], float]:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start = time.time()
        results = self.model(
            image,
            device=self.device,
            conf=conf,
            iou=iou,
            classes=None,
            verbose=False,
        )
        inference_time = time.time() - start

        detections = []
        for res in results[0].boxes:
            for box in res:
                class_id = int(box.cls.cpu())
                bbox = [int(c) for c in box.xyxy.cpu().squeeze().tolist()]
                score = box.conf.cpu().squeeze().item()
                detections.append(
                    DetectionResult(
                        class_name=self.names[class_id],
                        bbox=bbox,
                        score=round(score, 4),
                        class_id=class_id,
                    )
                )

        return detections, inference_time

    def get_class_names(self) -> List[str]:
        return self.names.copy()

    def get_current_model_name(self) -> str:
        return os.path.basename(self.model_path) if self.model_path else ""
