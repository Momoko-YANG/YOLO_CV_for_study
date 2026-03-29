import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Ensure backend packages are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.database import Base, get_db  # noqa: E402
from models import DetectionResult  # noqa: E402
from services.yolo_service import YOLOService  # noqa: E402


# ---------------------------------------------------------------------------
# Database: in-memory SQLite for test isolation
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture()
async def test_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _override_get_db():
        async with session_factory() as session:
            yield session

    yield _override_get_db

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ---------------------------------------------------------------------------
# Mock YOLO service
# ---------------------------------------------------------------------------

SAMPLE_DETECTIONS = [
    DetectionResult(class_name="Stop", bbox=[100, 100, 200, 200], score=0.92, class_id=0),
]


@pytest.fixture()
def mock_yolo_service():
    """Patch YOLOService to avoid loading a real YOLO model."""
    service = YOLOService()
    service.model = MagicMock()
    service.model_path = "weights/test-model.pt"
    service.names = ["停止", "了解", "数字2"]
    service.device = "cpu"

    original = YOLOService._instance
    YOLOService._instance = service

    yield service

    YOLOService._instance = original


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture()
async def client(test_db, mock_yolo_service):
    """Async HTTP client backed by the real FastAPI app (with mocked deps)."""
    # Patch model loading in lifespan so it does not attempt real YOLO init
    with patch("main.YOLOService.get_instance", return_value=mock_yolo_service):
        from main import app
        app.dependency_overrides[get_db] = test_db

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture()
async def auth_token(client: AsyncClient) -> str:
    """Register a test user and return the JWT token."""
    resp = await client.post("/api/auth/register", json={"username": "testuser", "password": "testpass"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Test image helper
# ---------------------------------------------------------------------------

@pytest.fixture()
def test_image_bytes() -> bytes:
    """640x640 black JPEG image for upload tests."""
    import cv2
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()
