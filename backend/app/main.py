# -*- coding: utf-8 -*-
"""
FastAPI entry point.
Run with:  uvicorn backend.app.main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.routers import auth, detection
from backend.app.services.detection_service import load_detector
from QtFusion.path import abs_path   # keep same path helper as original

app = FastAPI(title="YOLO Gesture Recognition API", version="1.0.0")

# Allow React dev server on port 5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(detection.router)

# Serve uploaded avatars as static files
app.mount("/static", StaticFiles(directory="backend/static"), name="static")


@app.on_event("startup")
async def startup():
    # Load YOLO weights once at startup (same path as original scripts)
    model_path = abs_path("weights/best-yolov8n.pt", path_type="current")
    load_detector(model_path)


@app.get("/health")
def health():
    return {"status": "ok"}
