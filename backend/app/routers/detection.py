# -*- coding: utf-8 -*-
"""
Detection router — /detect/*
Mirrors System_noLogin.py: image / video-frame / camera endpoints.
"""
import base64
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket
from fastapi.responses import JSONResponse
import cv2
import numpy as np

from backend.app.models.detection import DetectionResponse, DetectionBox, DetectorParams
from backend.app.services import detection_service as svc

router = APIRouter(prefix="/detect", tags=["detection"])


# ------------------------------------------------------------------
# Single image
# ------------------------------------------------------------------
@router.post("/image", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...)):
    data = await file.read()
    try:
        result = svc.detect_image(data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    detections = [DetectionBox(**d) for d in result["detections"]]
    return DetectionResponse(
        detections=detections,
        class_counts=result["class_counts"],
        inference_ms=result["inference_ms"],
    )


@router.post("/image/annotated")
async def detect_image_annotated(file: UploadFile = File(...)):
    """Returns annotated JPEG as base64 + detection metadata."""
    data = await file.read()
    result = svc.detect_image(data)
    b64 = base64.b64encode(result["annotated_image_bytes"]).decode()
    return {
        "image_b64": b64,
        "detections": result["detections"],
        "class_counts": result["class_counts"],
        "inference_ms": result["inference_ms"],
    }


# ------------------------------------------------------------------
# WebSocket camera / video stream
# ------------------------------------------------------------------
@router.websocket("/stream")
async def detection_stream(websocket: WebSocket):
    """
    Client sends raw JPEG frames as binary messages.
    Server responds with JSON {detections, class_counts, inference_ms, image_b64}.
    """
    await websocket.accept()
    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            result = svc.detect_video_frame(frame)
            annotated = svc._annotate(frame.copy(), result["detections"])
            _, buf = cv2.imencode(".jpg", annotated)
            b64 = base64.b64encode(buf.tobytes()).decode()

            await websocket.send_json({
                "image_b64": b64,
                "detections": result["detections"],
                "class_counts": result["class_counts"],
                "inference_ms": result["inference_ms"],
            })
    except Exception:
        await websocket.close()


# ------------------------------------------------------------------
# Detector params update
# ------------------------------------------------------------------
@router.post("/params")
def update_params(params: DetectorParams):
    svc.get_detector().set_param(params.model_dump(exclude_none=True))
    return {"message": "params updated"}
