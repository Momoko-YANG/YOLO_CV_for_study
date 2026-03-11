# -*- coding: utf-8 -*-
"""
YOLO Detector — extracted from src/YOLOv8v5Model.py
Decoupled from Qt; can be used standalone by the FastAPI backend.
"""
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device

# ---------------------------------------------------------------------------
# Default inference parameters
# ---------------------------------------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"

DEFAULT_PARAMS = {
    "device": device,
    "conf": 0.25,    # confidence threshold
    "iou": 0.5,      # NMS IOU threshold
    "classes": None, # None = all classes
    "verbose": False,
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def count_classes(det_info: list, class_names: list) -> list:
    """Return per-class detection counts in the same order as class_names."""
    count_dict = {name: 0 for name in class_names}
    for info in det_info:
        name = info["class_name"]
        if name in count_dict:
            count_dict[name] += 1
    return [count_dict[name] for name in class_names]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class YOLOv8v5Detector:
    """
    Thin wrapper around Ultralytics YOLO.
    Replaces the Qt-dependent QtFusion Detector base class.
    """

    imgsz: int = 640

    def __init__(self, params: dict | None = None):
        self.model = None
        self.img = None
        self.names: list[str] = []
        self.params = params if params else DEFAULT_PARAMS.copy()

    # ------------------------------------------------------------------
    def load_model(self, model_path: str) -> None:
        self.device = select_device(self.params["device"])
        self.model = YOLO(model_path)

        # Map model label IDs to display names
        # (Chinese_name lookup kept in label_names.py)
        from algorithm.label_names import CHINESE_NAME
        names_dict = self.model.names
        self.names = [
            CHINESE_NAME.get(v, v) for v in names_dict.values()
        ]

        # Warm-up passes
        dummy = torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device)
        dummy = dummy.type_as(next(self.model.model.parameters()))
        self.model(dummy)
        self.model(torch.rand_like(dummy))

    # ------------------------------------------------------------------
    def preprocess(self, img):
        """Store original frame and return it unchanged (pre-processing hook)."""
        self.img = img
        return img

    # ------------------------------------------------------------------
    def predict(self, img):
        """Run inference. Returns (results, heatmap_img)."""
        results = self.model(img, **self.params)
        # Optional heatmap — returns None when no hook is registered
        heatmap_img = None
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            heatmap_img = self._heatmap.get_heatmap(rgb)
        except AttributeError:
            pass
        return results, heatmap_img

    # ------------------------------------------------------------------
    def postprocess(self, pred) -> list[dict]:
        """Convert raw YOLO results to a list of detection dicts."""
        results = []
        for res in pred[0].boxes:
            for box in res:
                class_id = int(box.cls.cpu())
                bbox = [int(c) for c in box.xyxy.cpu().squeeze().tolist()]
                results.append({
                    "class_name": self.names[class_id],
                    "bbox": bbox,
                    "score": float(box.conf.cpu().squeeze()),
                    "class_id": class_id,
                })
        return results

    # ------------------------------------------------------------------
    def set_param(self, params: dict) -> None:
        self.params.update(params)
