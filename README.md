# 基于深度学习的手势识别 Web 系统

本仓库现已收敛为单一的 Web 架构，只保留：

- `backend/`：FastAPI 后端，负责认证、文件处理、YOLO 推理、WebSocket 实时检测
- `frontend/`：React + TypeScript 前端，负责登录、上传、摄像头检测和结果展示
- `scripts/`：模型训练和少量开发辅助脚本

旧桌面端界面、重复资源目录和仓库内 vendored `ultralytics/` 副本已移除，不再作为当前系统的一部分维护。

## 架构

```text
Browser
  ├─ HTTP /api/*
  └─ WebSocket /ws/detect
        ↓
FastAPI backend
  ├─ auth
  ├─ detection
  ├─ websocket
  ├─ SQLite
  └─ Ultralytics YOLO
        ↓
React frontend
```

## 目录

```text
.
├── backend/
│   ├── core/
│   ├── models/
│   ├── routers/
│   ├── services/
│   ├── static/
│   ├── weights/
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── scripts/
│   ├── run_train_model.py
│   └── test.py
├── test_media/
├── start.sh
└── README.md
```

## 运行

### 环境要求

- Python 3.9+
- Node.js 18+
- `backend/weights/best-yolov8n.pt`

### 首次安装

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

cd frontend
npm install
cd ..
```

### 一键启动

```bash
chmod +x start.sh
./start.sh
```

启动后访问：

- 前端：`http://localhost:5173`
- 后端：`http://localhost:8000`
- API 文档：`http://localhost:8000/docs`

### 手动启动

终端 1：

```bash
cd backend
PYTHONPATH="$(pwd)" ../venv/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

终端 2：

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

## 功能

- 用户注册、登录、改密、头像上传、注销
- 图片检测、视频检测、摄像头实时检测
- YOLOv5n / YOLOv8n 切换
- WebSocket 实时返回检测框和推理时间
- 结果导出和基础国际化

## 训练

训练脚本保留在 [`scripts/run_train_model.py`](/Users/momokoyang/GitHub/YOLO_hand_gesture-main/scripts/run_train_model.py)：

```bash
source venv/bin/activate
python scripts/run_train_model.py
```

默认读取 `datasets/RPS/`，输出到 `runs/detect/`。
