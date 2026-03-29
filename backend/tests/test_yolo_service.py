import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.yolo_service import GESTURE_LABELS, YOLOService  # noqa: E402


class TestYOLOServiceSingleton:
    def test_returns_same_instance(self):
        YOLOService._instance = None
        a = YOLOService.get_instance()
        b = YOLOService.get_instance()
        assert a is b
        YOLOService._instance = None

    def test_fresh_instance_has_no_model(self):
        YOLOService._instance = None
        svc = YOLOService.get_instance()
        assert svc.model is None
        assert svc.names == []
        YOLOService._instance = None


class TestYOLOServiceDetect:
    def test_detect_raises_without_model(self):
        svc = YOLOService()
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            svc.detect(img)

    def test_detect_returns_detections(self):
        svc = YOLOService()
        svc.names = ["停止", "了解", "数字2"]
        svc.device = "cpu"

        # Build a fake box
        fake_box = MagicMock()
        fake_box.cls.cpu.return_value = MagicMock(__int__=lambda self: 0)
        fake_box.xyxy.cpu.return_value.squeeze.return_value.tolist.return_value = [10, 20, 100, 200]
        fake_box.conf.cpu.return_value.squeeze.return_value.item.return_value = 0.91

        fake_result = MagicMock()
        fake_result.boxes = [[fake_box]]

        svc.model = MagicMock(return_value=[fake_result])

        img = np.zeros((640, 640, 3), dtype=np.uint8)
        detections, elapsed = svc.detect(img)

        assert len(detections) == 1
        assert detections[0].class_name == "停止"
        assert detections[0].class_id == 0
        assert detections[0].score == 0.91
        assert detections[0].bbox == [10, 20, 100, 200]
        assert elapsed >= 0


class TestGestureLabels:
    def test_all_classes_have_bilingual_labels(self):
        for name, labels in GESTURE_LABELS.items():
            assert "en" in labels
            assert "zh" in labels

    def test_expected_classes_present(self):
        assert set(GESTURE_LABELS.keys()) == {"Stop", "Understand", "NumberTwo"}


class TestGetClassNames:
    def test_returns_copy(self):
        svc = YOLOService()
        svc.names = ["停止", "了解", "数字2"]
        result = svc.get_class_names()
        assert result == svc.names
        assert result is not svc.names

    def test_returns_empty_when_no_model(self):
        svc = YOLOService()
        assert svc.get_class_names() == []
