import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # JWT
    secret_key: str = "gesture-recognition-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours

    # Model
    model_path: str = "weights/best-yolov8n.pt"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    default_conf: float = 0.25
    default_iou: float = 0.5
    image_size: int = 640

    # Database
    database_url: str = "sqlite+aiosqlite:///./app.db"

    # Uploads
    upload_dir: str = "./static/uploads"
    export_dir: str = "./static/exports"
    avatar_dir: str = "./static/avatars"
    max_upload_size: int = 50 * 1024 * 1024  # 50MB

    class Config:
        env_file = ".env"


settings = Settings()
