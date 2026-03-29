"""
Model quality gate tests.

These tests only run when a trained model weight file is present.
They validate that the model meets minimum deployment requirements.

Run explicitly:  pytest -m model_quality
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_PATH = Path(__file__).resolve().parent.parent / "weights" / "best-yolov8n.pt"
EXPECTED_CLASSES = {"Stop", "Understand", "NumberTwo"}

skip_no_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"Model weight not found: {MODEL_PATH}",
)


@pytest.mark.model_quality
@skip_no_model
class TestModelQualityGate:
    @pytest.fixture(autouse=True, scope="class")
    def load_model(self):
        from ultralytics import YOLO
        self.__class__._model = YOLO(str(MODEL_PATH))

    @property
    def model(self):
        return self.__class__._model

    def test_model_loads_with_correct_class_count(self):
        assert len(self.model.names) == 3

    def test_model_class_names_match(self):
        names = set(self.model.names.values())
        assert names == EXPECTED_CLASSES

    def test_inference_returns_valid_boxes(self):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = self.model(img, verbose=False)
        for box in results[0].boxes:
            coords = box.xyxy.cpu().squeeze().tolist()
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                assert x1 < x2, f"Invalid box: x1={x1} >= x2={x2}"
                assert y1 < y2, f"Invalid box: y1={y1} >= y2={y2}"
            score = box.conf.cpu().squeeze().item()
            assert 0.0 <= score <= 1.0

    def test_inference_latency_under_threshold(self):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        # Warmup
        self.model(img, verbose=False)

        times = []
        for _ in range(5):
            start = time.time()
            self.model(img, verbose=False)
            times.append(time.time() - start)

        mean_ms = sum(times) / len(times) * 1000
        assert mean_ms < 2000, f"Mean inference time {mean_ms:.0f}ms exceeds 2000ms threshold"
