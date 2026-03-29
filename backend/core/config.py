from typing import Literal

import torch
from pydantic import model_validator
from pydantic_settings import BaseSettings

_DEV_SECRET = "gesture-recognition-secret-key-change-in-production"


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Settings(BaseSettings):
    # Environment
    app_env: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = False

    # JWT — stable default for dev; MUST be overridden via .env in staging/prod
    secret_key: str = _DEV_SECRET
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    @model_validator(mode="after")
    def _check_secret_in_prod(self):
        if self.app_env in ("staging", "prod") and self.secret_key == _DEV_SECRET:
            raise ValueError(
                "SECRET_KEY must be set via .env for staging/prod environments. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )
        return self

    # CORS
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Model
    model_path: str = "weights/best-yolov8n.pt"
    device: str = _default_device()
    default_conf: float = 0.25
    default_iou: float = 0.5
    image_size: int = 640

    # Database
    database_url: str = "sqlite+aiosqlite:///./app.db"

    # Uploads
    upload_dir: str = "./static/uploads"
    export_dir: str = "./static/exports"
    avatar_dir: str = "./static/avatars"
    max_upload_size: int = 50 * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
