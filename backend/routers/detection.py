import asyncio
import os
import time
import uuid
from typing import Dict, List

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from core.config import settings
from core.dependencies import get_current_user
from models.schemas import (
    DetectionResult,
    ImageDetectionResponse,
    VideoStatusResponse,
    VideoTaskResponse,
)
from models.user import User
from services.export_service import save_annotated_image, save_results_csv
from services.yolo_service import YOLOService

router = APIRouter(prefix="/api/detect", tags=["detection"])

# In-memory video task storage
video_tasks: Dict[str, dict] = {}


@router.post("/image", response_model=ImageDetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.5,
    _user: User = Depends(get_current_user),
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image = cv2.resize(image, (settings.image_size, settings.image_size))

    yolo = YOLOService.get_instance()
    loop = asyncio.get_event_loop()
    detections, inference_time = await loop.run_in_executor(
        None, lambda: yolo.detect(image, conf=conf, iou=iou)
    )

    # Save annotated image
    image_path = await loop.run_in_executor(
        None, lambda: save_annotated_image(image, detections)
    )
    image_url = "/" + image_path.replace("\\", "/")

    return ImageDetectionResponse(
        detections=detections,
        inference_time=round(inference_time, 4),
        image_url=image_url,
    )


@router.post("/video", response_model=VideoTaskResponse)
async def detect_video(
    file: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.5,
    _user: User = Depends(get_current_user),
):
    task_id = uuid.uuid4().hex[:12]
    os.makedirs(settings.upload_dir, exist_ok=True)
    video_path = os.path.join(settings.upload_dir, f"{task_id}.mp4")

    contents = await file.read()
    with open(video_path, "wb") as f:
        f.write(contents)

    video_tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "total_frames": 0,
        "processed_frames": 0,
        "results": [],
        "video_path": video_path,
        "conf": conf,
        "iou": iou,
    }

    asyncio.get_event_loop().run_in_executor(
        None, _process_video_sync, task_id
    )

    return VideoTaskResponse(task_id=task_id)


def _process_video_sync(task_id: str):
    task = video_tasks[task_id]
    cap = cv2.VideoCapture(task["video_path"])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    task["total_frames"] = total_frames

    yolo = YOLOService.get_instance()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (settings.image_size, settings.image_size))
        detections, _ = yolo.detect(frame, conf=task["conf"], iou=task["iou"])
        task["results"].append(detections)
        frame_idx += 1
        task["processed_frames"] = frame_idx
        task["progress"] = round(frame_idx / max(total_frames, 1) * 100, 1)

    cap.release()
    task["status"] = "completed"
    task["progress"] = 100


@router.get("/video/{task_id}/status", response_model=VideoStatusResponse)
async def get_video_status(task_id: str):
    task = video_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return VideoStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        total_frames=task["total_frames"],
        processed_frames=task["processed_frames"],
    )


@router.get("/video/{task_id}/results")
async def get_video_results(task_id: str):
    task = video_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task_id,
        "total_frames": task["total_frames"],
        "frames": [[d.model_dump() for d in frame] for frame in task["results"]],
    }


@router.get("/model/classes")
async def get_model_classes():
    yolo = YOLOService.get_instance()
    return {"classes": yolo.get_class_names()}


@router.post("/model/upload")
async def upload_model(
    file: UploadFile = File(...),
    _user: User = Depends(get_current_user),
):
    if not file.filename or not file.filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="Only .pt model files are accepted")

    os.makedirs("weights", exist_ok=True)
    model_path = os.path.join("weights", file.filename)
    contents = await file.read()
    with open(model_path, "wb") as f:
        f.write(contents)

    yolo = YOLOService.get_instance()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: yolo.load_model(model_path))

    return {"message": "Model loaded", "classes": yolo.get_class_names()}


# --- Export ---

@router.get("/export/csv/{task_id}")
async def export_csv(task_id: str):
    task = video_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    loop = asyncio.get_event_loop()
    filepath = await loop.run_in_executor(
        None,
        lambda: save_results_csv(task["results"], task.get("video_path", "")),
    )
    return FileResponse(filepath, filename=os.path.basename(filepath), media_type="text/csv")
