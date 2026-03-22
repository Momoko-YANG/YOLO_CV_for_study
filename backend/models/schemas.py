from typing import List, Optional
from pydantic import BaseModel


# --- Auth ---

class UserRegister(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    avatar_url: Optional[str] = None

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


# --- Detection ---

class DetectionResult(BaseModel):
    class_name: str
    bbox: List[int]  # [x1, y1, x2, y2]
    score: float
    class_id: int


class ImageDetectionResponse(BaseModel):
    detections: List[DetectionResult]
    inference_time: float
    image_url: str
    heatmap_url: Optional[str] = None


class VideoTaskResponse(BaseModel):
    task_id: str
    status: str = "processing"


class VideoStatusResponse(BaseModel):
    task_id: str
    status: str  # processing, completed, failed
    progress: float  # 0-100
    total_frames: int
    processed_frames: int


class VideoResultsResponse(BaseModel):
    task_id: str
    frames: List[List[DetectionResult]]
    total_frames: int


class WebSocketMessage(BaseModel):
    detections: List[DetectionResult]
    inference_time: float
    frame_id: int


class ModelInfo(BaseModel):
    name: str
    path: str
    is_current: bool = False


class ModelListResponse(BaseModel):
    current_model: Optional[str] = None
    models: List[ModelInfo]


class ModelSelectRequest(BaseModel):
    name: str
