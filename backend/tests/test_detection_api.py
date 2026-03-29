import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import AsyncClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import DetectionResult  # noqa: E402
from tests.conftest import auth_headers  # noqa: E402


# ---------------------------------------------------------------------------
# Health endpoint (no auth required)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


# ---------------------------------------------------------------------------
# Image detection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_detect_image_no_auth(client: AsyncClient, test_image_bytes: bytes):
    resp = await client.post(
        "/api/detect/image",
        files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_detect_image_success(
    client: AsyncClient,
    auth_token: str,
    test_image_bytes: bytes,
    mock_yolo_service,
):
    sample = [DetectionResult(class_name="停止", bbox=[10, 10, 50, 50], score=0.95, class_id=0)]
    mock_yolo_service.detect = lambda img, conf=0.25, iou=0.5: (sample, 0.032)

    with patch("routers.detection.save_annotated_image", return_value="static/uploads/annotated.jpg"):
        resp = await client.post(
            "/api/detect/image",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
            params={"conf": 0.3},
            headers=auth_headers(auth_token),
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["detections"]) == 1
    assert data["detections"][0]["class_name"] == "停止"
    assert "inference_time" in data


@pytest.mark.asyncio
async def test_detect_image_invalid_file(client: AsyncClient, auth_token: str):
    resp = await client.post(
        "/api/detect/image",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_model_classes(client: AsyncClient, mock_yolo_service):
    resp = await client.get("/api/detect/model/classes")
    assert resp.status_code == 200
    assert resp.json()["classes"] == ["停止", "了解", "数字2"]


@pytest.mark.asyncio
async def test_model_list_requires_auth(client: AsyncClient):
    resp = await client.get("/api/detect/model/list")
    assert resp.status_code == 401
