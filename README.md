# Deep Learning Based Gesture Recognition System

> Personal learning project for gesture recognition with YOLO models.

当前仓库只保留 Web 版手势识别系统。

## Current Focus

Web 版采用前后端分离架构：

- Backend: FastAPI，负责鉴权、YOLO 推理、导出与 WebSocket
- Frontend: React + TypeScript + Vite，负责登录、检测界面和实时相机流
- Realtime: WebSocket 传输 JPEG 帧与检测结果
- Database: SQLite + SQLAlchemy async

## Repository Layout

```text
.
├── backend/        FastAPI backend
├── frontend/       React + TypeScript frontend
├── scripts/        Small helper scripts
├── datasets/       Local training data, ignored by git
├── test_media/     Test images and videos
├── weights/        Local model weights, ignored by git
├── ultralytics/    Vendored YOLO-related code kept in-repo
├── start.sh        Start backend + frontend together
└── Makefile        Common local development commands
```

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- A local virtualenv at `./venv`
- Model weights under `backend/weights/`

### Initial Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r backend/requirements.txt
cd frontend && npm install
```

### Start Everything

```bash
./start.sh
```

或者：

```bash
make dev
```

启动后访问：

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

## Common Commands

```bash
make setup           # install backend + frontend dependencies
make backend         # run FastAPI with reload
make frontend        # run Vite dev server
make dev             # run backend + frontend together
make clean           # remove common local build artifacts
```

## Backend Notes

- 入口文件：`backend/main.py`
- 路由目录：`backend/routers/`
- 业务服务：`backend/services/`
- 配置和数据库：`backend/core/`

健康检查接口：

```text
GET /api/health
```

常用业务接口：

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/detect/image`
- `POST /api/detect/video`
- `GET /api/detect/export/csv/{id}`
- `WS /ws/detect?token=JWT`

## Frontend Notes

- 页面：`frontend/src/pages/`
- 布局组件：`frontend/src/components/layout/`
- 检测组件：`frontend/src/components/detection/`
- 状态与连接：`frontend/src/hooks/`
- API 调用：`frontend/src/services/api.ts`

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI, SQLAlchemy, Uvicorn |
| Frontend | React 19, TypeScript, Vite |
| State | Zustand |
| Auth | JWT, bcrypt |
| ML | Ultralytics YOLO, PyTorch, OpenCV |
| Database | SQLite |
| Realtime | WebSocket |

## License

This project is for personal learning use only.
