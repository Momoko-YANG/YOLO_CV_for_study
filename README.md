# Deep Learning Based Gesture Recognition System

> **Disclaimer**: This is a **personal learning project**, intended only for learning and researching the application of YOLO object detection algorithms in gesture recognition.

A real-time gesture recognition system built on YOLOv8/v5 models. The system recognizes gestures from images, videos, and real-time camera input through a web-based interface.

## Architecture

The project uses a **frontend-backend separation** architecture:

- **Backend**: FastAPI (Python) — YOLO model inference, user auth, file processing
- **Frontend**: React + TypeScript + Tailwind CSS — detection visualization, camera capture
- **Real-time**: WebSocket — camera frames streamed as binary JPEG, detection results returned as JSON
- **Database**: SQLite (via SQLAlchemy async)
- **Auth**: JWT (bcrypt password hashing)
- **i18n**: Japanese / English bilingual interface

## Features

- **Multi-model Support**: YOLOv5n and YOLOv8 with runtime model switching
- **Multiple Input Sources**: Camera (real-time WebSocket), image upload, video upload, folder batch
- **3 Gesture Classes**: Stop (停止), Understand (了解), Number Two (数字2)
- **Interactive UI**: Collapsible sidebar, confidence/IOU sliders, detection canvas with bounding box overlays, results table
- **User Management**: Registration, login, avatar upload (JWT-based)
- **Export**: Download annotated images, video results as CSV

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Model weights in `backend/weights/best-yolov8n.pt`

### One-command Start

```bash
./start.sh
```

This starts both the backend (port 8000) and frontend (port 5173), then open `http://localhost:5173`.

### Manual Start

**Terminal 1 — Backend:**
```bash
cd backend
PYTHONPATH="$(pwd)" ../venv/bin/python3 -m uvicorn main:app --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173`.

### First-time Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt
pip install ultralytics torch torchvision

# Install frontend dependencies
cd frontend && npm install
```

## Project Structure

```
├── backend/                     # FastAPI backend
│   ├── main.py                  # App entry, CORS, lifespan (auto-loads YOLO model)
│   ├── core/
│   │   ├── config.py            # Pydantic Settings
│   │   ├── database.py          # SQLAlchemy + SQLite async
│   │   ├── security.py          # JWT + bcrypt
│   │   └── dependencies.py      # get_current_user dependency
│   ├── models/
│   │   ├── user.py              # User model
│   │   └── schemas.py           # Request/response schemas
│   ├── routers/
│   │   ├── auth.py              # POST /register, /login, /change-password, /avatar
│   │   ├── detection.py         # POST /detect/image, /detect/video, model upload, export
│   │   └── websocket.py         # WS /ws/detect (real-time camera)
│   ├── services/
│   │   ├── yolo_service.py      # YOLO singleton service
│   │   └── export_service.py    # CSV/image export
│   ├── weights/                 # Model weights (.pt files)
│   └── requirements.txt
│
├── frontend/                    # React + TypeScript frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── LoginPage.tsx    # Login/Register page
│   │   │   └── MainPage.tsx     # Main detection page
│   │   ├── components/
│   │   │   ├── layout/          # Sidebar, TitleBar
│   │   │   ├── detection/       # DetectionCanvas, ResultsTable, DetectionInfo
│   │   │   └── controls/        # Confidence/IOU sliders
│   │   ├── hooks/
│   │   │   ├── useAuth.ts       # Zustand auth store
│   │   │   ├── useWebSocket.ts  # WebSocket connection management
│   │   │   ├── useCamera.ts     # getUserMedia + frame capture
│   │   │   └── useDetection.ts  # Detection state store
│   │   ├── i18n/                # en.json, ja.json
│   │   ├── services/api.ts      # Axios + JWT interceptor
│   │   ├── types/               # TypeScript type definitions
│   │   └── utils/canvas.ts      # Bounding box drawing
│   ├── package.json
│   └── vite.config.ts           # Proxy /api → 8000, /ws → ws://8000
│
├── src/                         # Original PySide6 desktop app (legacy)
├── scripts/                     # Legacy run/test scripts
├── datasets/                    # Training dataset
├── test_media/                  # Test images and videos
├── start.sh                     # One-command startup script
└── README.md
```

## API Endpoints

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login, returns JWT |
| GET | `/api/auth/me` | Get current user |
| PUT | `/api/auth/change-password` | Change password |
| POST | `/api/auth/avatar` | Upload avatar |
| DELETE | `/api/auth/account` | Delete account |

### Detection
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/detect/image` | Upload image for detection |
| POST | `/api/detect/video` | Upload video for processing |
| GET | `/api/detect/video/{id}/status` | Get video processing progress |
| GET | `/api/detect/video/{id}/results` | Get video detection results |
| POST | `/api/detect/model/upload` | Upload custom .pt model |
| GET | `/api/detect/export/csv/{id}` | Export results as CSV |

### WebSocket
| Endpoint | Description |
|----------|-------------|
| `WS /ws/detect?token=JWT` | Real-time camera detection |

**Protocol:**
- Client → Server: Binary (JPEG frame) or Text (`{"type":"config","conf":0.35,"iou":0.45}`)
- Server → Client: `{"detections":[{class_name, bbox, score, class_id}], "inference_time":0.034, "frame_id":42}`

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check + model status |

Full interactive API docs available at `http://localhost:8000/docs`.

## Model Performance

| Model | Precision | Recall | mAP@0.5 | Speed (CPU) |
|-------|-----------|--------|---------|-------------|
| YOLOv8n | 95%+ | 95%+ | 0.963 | ~78ms |
| YOLOv5n | 94%+ | 93%+ | 0.940 | ~80ms |

## Dataset

- **Source**: [HaGRID](https://github.com/hukenovs/hagrid) open-source gesture dataset
- **Total**: 11,886 images (train: 10,953 / val: 604 / test: 329)
- **Resolution**: 640x640
- **Classes**: Stop, Understand, NumberTwo

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI, SQLAlchemy, Uvicorn |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS |
| State | Zustand |
| Auth | JWT (python-jose), bcrypt (passlib) |
| ML | Ultralytics YOLOv8, PyTorch, OpenCV |
| Database | SQLite + aiosqlite |
| i18n | react-i18next (EN/JA) |
| Real-time | WebSocket (binary JPEG frames) |

## License

This project is for personal learning use only.

## Acknowledgments

- [HaGRID](https://github.com/hukenovs/hagrid) — Open-source gesture dataset
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — Detection framework
- [FastAPI](https://fastapi.tiangolo.com/) — Backend framework
