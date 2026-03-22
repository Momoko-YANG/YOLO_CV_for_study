import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core import init_db, settings
from routers import auth_router, detection_router, ws_router
from services import YOLOService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: init DB and load YOLO model
    await init_db()
    yolo = YOLOService.get_instance()
    if os.path.exists(settings.model_path):
        try:
            yolo.load_model(settings.model_path, settings.device)
        except Exception as e:
            print(f"Warning: Failed to load YOLO model: {e}")
    else:
        print(f"Warning: Model not found at {settings.model_path}, starting without model")
    yield


app = FastAPI(title="Gesture Recognition API", version="1.0.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
for d in [settings.upload_dir, settings.export_dir, settings.avatar_dir]:
    os.makedirs(d, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(auth_router)
app.include_router(detection_router)
app.include_router(ws_router)


@app.get("/api/health")
async def health():
    yolo = YOLOService.get_instance()
    return {
        "status": "ok",
        "model_loaded": yolo.model is not None,
        "classes": yolo.get_class_names() if yolo.model else [],
    }
