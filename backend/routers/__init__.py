from .auth import router as auth_router
from .detection import router as detection_router
from .websocket import router as ws_router

__all__ = [
    "auth_router",
    "detection_router",
    "ws_router",
]
