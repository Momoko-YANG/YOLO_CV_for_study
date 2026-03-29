# YOLO Hand Gesture Recognition

A web-based hand gesture recognition system built with **YOLOv8 / YOLOv5 + FastAPI + React**.

## Features

- User registration / login (JWT auth), password change, avatar upload, account deletion
- Image upload detection
- Async video upload processing
- Real-time webcam detection (WebSocket)
- Model listing, switching, and uploading
- Video detection results export to CSV
- Frontend i18n support (Chinese / English / Japanese)
- Multi-environment configuration (dev / staging / prod)
- One-command Docker Compose deployment
- CI automation (Ruff lint + ESLint + pytest + Docker build)

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19 · TypeScript · Vite · Tailwind CSS · Zustand · react-i18next |
| Backend | FastAPI · Uvicorn · SQLAlchemy (async) · aiosqlite · python-jose · passlib |
| Model | Ultralytics YOLO · OpenCV · PyTorch (CUDA / MPS / CPU) |
| Real-time | WebSocket |
| Database | SQLite |
| CI/CD | GitHub Actions · Docker Compose · Ruff · ESLint · pytest |
| Experiment Tracking | MLflow (optional) |

## Project Structure

```
.
├── backend/                  # FastAPI backend
│   ├── core/                 # Config, database, auth, dependency injection
│   ├── models/               # Pydantic schemas + SQLAlchemy entities
│   ├── routers/              # auth · detection · websocket routes
│   ├── services/             # YOLO inference service · export service
│   ├── tests/                # pytest tests
│   ├── weights/              # Model weights (.pt)
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── components/       # Sidebar · TitleBar · DetectionCanvas etc.
│   │   ├── hooks/            # useAuth · useCamera · useWebSocket · useDetection
│   │   ├── i18n/             # zh.json · en.json · ja.json
│   │   ├── pages/            # LoginPage · MainPage
│   │   ├── services/         # Axios API wrapper
│   │   ├── types/            # TypeScript types
│   │   └── utils/            # Canvas drawing utilities
│   ├── Dockerfile
│   └── vite.config.ts
├── training/                 # Model training pipeline
│   ├── train.py              # Fine-tuning
│   ├── evaluate.py           # Model evaluation
│   ├── benchmark.py          # Inference benchmarking
│   ├── export_onnx.py        # ONNX export
│   ├── validate_dataset.py   # Dataset quality validation
│   ├── compare.py            # Experiment comparison
│   ├── dataset.yaml          # Dataset configuration
│   ├── configs/              # Experiment hyperparameter configs
│   └── notebooks/            # EDA notebook
├── .env.example              # Environment variable template
├── .github/workflows/ci.yml  # GitHub Actions CI
├── docker-compose.yml        # Docker orchestration
├── Makefile                  # Common commands
├── start.sh                  # One-command startup script
└── pyproject.toml            # Project metadata + Ruff/pytest config
```

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- At least one `.pt` model weight file in `backend/weights/`

### Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

cd frontend && npm install && cd ..
```

Or use the Makefile:

```bash
make setup
```

### Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your values (defaults work fine for development)
```

### Start Services

```bash
# Start both frontend and backend
./start.sh

# Or start separately
make backend   # Backend at http://localhost:8000
make frontend  # Frontend at http://localhost:5173
```

Once running:

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs (disabled in prod) |

### Docker Deployment

```bash
docker compose up --build
```

Includes three services: `backend` (FastAPI), `frontend` (Vite), `mlflow` (experiment tracking).

## Configuration

The backend uses `pydantic-settings` for configuration management. All parameters can be overridden via `.env` file or environment variables.

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `dev` | Environment: `dev` / `staging` / `prod` |
| `DEBUG` | `false` | Debug mode |
| `SECRET_KEY` | Dev-only key | JWT signing key; **must** be overridden in staging/prod |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Allowed frontend origins (comma-separated) |
| `MODEL_PATH` | `weights/best-yolov8n.pt` | Default model path |
| `DEVICE` | Auto-detected | Inference device: `cuda:0` / `mps` / `cpu` |
| `DEFAULT_CONF` | `0.25` | Default confidence threshold |
| `DEFAULT_IOU` | `0.5` | Default IoU threshold |
| `DATABASE_URL` | `sqlite+aiosqlite:///./app.db` | Database connection |

> The backend will refuse to start in staging/prod if `SECRET_KEY` is not set. Generate one with:
> ```bash
> python -c "import secrets; print(secrets.token_urlsafe(32))"
> ```

## API Reference

### Authentication

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/auth/register` | Register a new user |
| `POST` | `/api/auth/login` | Login, returns JWT |
| `GET` | `/api/auth/me` | Get current user info |
| `PUT` | `/api/auth/change-password` | Change password |
| `POST` | `/api/auth/avatar` | Upload avatar |
| `DELETE` | `/api/auth/account` | Delete account |

### Detection

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/detect/image` | Detect from an image |
| `POST` | `/api/detect/video` | Upload and process video asynchronously |
| `GET` | `/api/detect/video/{task_id}/status` | Video processing progress |
| `GET` | `/api/detect/video/{task_id}/results` | Video detection results |
| `GET` | `/api/detect/export/csv/{task_id}` | Export results as CSV |

### Model Management

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/detect/model/list` | List available models |
| `GET` | `/api/detect/model/classes` | Get current model class names |
| `POST` | `/api/detect/model/select` | Switch active model |
| `POST` | `/api/detect/model/upload` | Upload and load a new model |

### WebSocket Real-time Detection

| Protocol | Path | Description |
|---|---|---|
| `WS` | `/ws/detect?token=<JWT>` | Real-time webcam detection |

**Message protocol:**

```
Client → Server: binary JPEG frame / {"type":"config","conf":0.3,"iou":0.4}
Server → Client: {"detections":[...],"inference_time":0.034,"frame_id":42}
```

### Health Check

```
GET /api/health → {"status":"ok","model_loaded":true,"classes":[...]}
```

## Model Training

The project includes a complete YOLO fine-tuning, evaluation, and comparison pipeline in the `training/` directory.

### Prepare Dataset

1. Annotate images in YOLO format (one `.txt` per image, each line: `class_id cx cy w h`)
2. Split into train/val directories
3. Update `training/dataset.yaml`

### Train

```bash
# Default configuration
python training/train.py

# Use a preset experiment config
python training/train.py --config training/configs/finetune_v8s.yaml

# Custom parameters
python training/train.py --model yolov8m.pt --epochs 200 --freeze 0 --lr0 0.0005
```

Available presets:

| Config File | Strategy | Use Case |
|---|---|---|
| `finetune_v8s.yaml` | YOLOv8s + frozen backbone | Recommended starting point |
| `finetune_v8m.yaml` | YOLOv8m + frozen backbone | Higher accuracy |
| `finetune_v8s_fullunfreeze.yaml` | YOLOv8s + all layers unfrozen | When data is abundant |

### Evaluate and Compare

```bash
# Evaluate a single model
python training/evaluate.py --model runs/gesture/exp1/weights/best.pt

# Compare against a baseline
python training/evaluate.py --model best.pt --baseline yolov8s.pt

# Summarize all experiments
python training/compare.py --output results.csv
```

### Additional Tools

```bash
# Dataset quality validation
python training/validate_dataset.py --data-root datasets/RPS

# Inference benchmarking
python training/benchmark.py --models best-yolov8n.pt best-yolov5nu.pt

# ONNX export
python training/export_onnx.py --model best.pt
```

### Deploy a Trained Model

After training, `best.pt` is automatically copied to `backend/weights/`. Switch to it via the frontend model management panel or by updating `MODEL_PATH` in `.env`.

## Development Commands

| Command | Description |
|---|---|
| `make setup` | Install all frontend and backend dependencies |
| `make backend` | Start backend with hot reload |
| `make frontend` | Start frontend dev server |
| `make dev` | Start both frontend and backend (`start.sh`) |
| `make build` | Build frontend production bundle |
| `make lint` | Run Ruff + ESLint |
| `make test` | Run backend unit/integration tests |
| `make test-quality` | Run model quality gate tests |
| `make mlflow` | Start MLflow tracking server |
| `make validate-data` | Run dataset quality checks |
| `make verify` | Lightweight local verification |
| `make clean` | Remove build artifacts |

## CI/CD

GitHub Actions runs automatically (`.github/workflows/ci.yml`):

1. **Lint** — `ruff check backend/ training/` + `npm run lint`
2. **Test** — `pytest backend/tests/` (excluding model quality gates)
3. **Docker Build** — `docker compose build`

## Known Limitations

- `Folder` currently only processes the first selected image, not full batch processing
- Video task state is stored in memory and does not survive server restarts
- CSV export only supports video detection results

## License

This project is for personal learning and research purposes only.
