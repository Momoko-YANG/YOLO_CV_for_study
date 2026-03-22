# YOLO Hand Gesture Web System

> A web-based hand gesture recognition system built with FastAPI, React, and YOLO.

中文说明见 [中文说明 / Chinese Guide](#中文说明)  
English documentation starts at [English Guide](#english-guide)

---

## 中文说明

### 1. 项目简介

这是一个基于 **YOLO + FastAPI + React** 的手势识别 Web 项目。

项目支持：

- 用户注册、登录、鉴权
- 图片识别
- 视频识别
- 摄像头实时识别
- 模型上传与切换
- 视频结果导出 CSV
- 前端界面语言切换

当前架构为前后端分离：

- 后端：FastAPI，负责鉴权、模型加载、推理、导出和 WebSocket
- 前端：React + TypeScript + Vite，负责登录页、主识别页和交互界面
- 模型：YOLO `.pt` 权重文件
- 数据库：SQLite

---

### 2. 当前功能

#### 2.1 已实现功能

- 账号注册与登录
- JWT 鉴权
- 修改密码
- 上传图片并返回检测结果
- 上传视频并异步处理
- 摄像头通过 WebSocket 实时识别
- 选择已有模型
- 上传新的 `.pt` 模型并立即切换
- 导出视频识别结果 CSV
- 前端支持中/英/日文案资源

#### 2.2 当前前端入口

- 左侧边栏：
  - `Camera`
  - `Image`
  - `Video`
  - `Folder`
  - `Model`
  - `Change Password`
  - 用户退出
- 顶部栏：
  - 语言切换
  - `Export CSV`

#### 2.3 当前行为说明

- `Export CSV` 只对视频识别任务有效
- `Model` 会打开模型弹窗，可查看、切换、上传模型
- `Folder` 当前实现是读取所选文件中的第一张图片进行检测，不是完整批处理
- 模型列表接口依赖后端新版本，修改后如果前端提示无法加载模型列表，请重启后端

---

### 3. 技术栈

| 层 | 技术 |
|---|---|
| Frontend | React 19, TypeScript, Vite |
| State | Zustand |
| Backend | FastAPI, Uvicorn |
| Auth | JWT, bcrypt / passlib |
| Database | SQLite, SQLAlchemy async, aiosqlite |
| Vision | Ultralytics YOLO, OpenCV, PyTorch |
| Real-time | WebSocket |
| i18n | react-i18next |

---

### 4. 目录结构

```text
.
├── backend/                  # FastAPI 后端
│   ├── core/                 # 配置、数据库、依赖、鉴权
│   ├── models/               # Pydantic schema / SQLAlchemy model
│   ├── routers/              # auth / detection / websocket 路由
│   ├── services/             # YOLO 服务、导出服务
│   ├── weights/              # 默认模型权重
│   ├── app.db                # SQLite 数据库（运行后生成/使用）
│   └── requirements.txt
│
├── frontend/                 # React 前端
│   ├── src/
│   │   ├── components/       # UI 组件
│   │   ├── hooks/            # Zustand / camera / websocket hooks
│   │   ├── i18n/             # 多语言资源
│   │   ├── pages/            # LoginPage / MainPage
│   │   ├── services/         # API 封装
│   │   ├── types/            # 前端类型定义
│   │   └── utils/            # Canvas 工具
│   ├── package.json
│   └── vite.config.ts
│
├── output_examples/          # 示例输出文件
├── scripts/                  # 小型辅助脚本
├── test_media/               # 测试图片/视频
├── ultralytics/              # 仓库内 YOLO 相关代码
├── weights/                  # 额外本地模型权重
├── Makefile                  # 常用命令入口
└── start.sh                  # 一键启动前后端
```

---

### 5. 环境要求

- Python 3.9+
- Node.js 18+
- 本地已创建 `./venv`
- 前端依赖已安装
- 至少有一个可用模型权重文件：
  - `backend/weights/*.pt`

---

### 6. 快速开始

#### 6.1 初始化环境

```bash
python3 -m venv venv
./venv/bin/pip install -r backend/requirements.txt
cd frontend && npm install
```

#### 6.2 使用 Makefile

```bash
make setup
make backend
make frontend
make dev
make build
make lint
make verify
make clean
```

#### 6.3 一键启动

```bash
./start.sh
```

或者：

```bash
make dev
```

启动后默认访问：

- 前端：`http://localhost:5173`
- 后端：`http://localhost:8000`
- API 文档：`http://localhost:8000/docs`

---

### 7. 开发命令说明

| 命令 | 作用 |
|---|---|
| `make setup` | 安装前后端依赖 |
| `make setup-backend` | 安装后端依赖到 `venv` |
| `make setup-frontend` | 安装前端依赖 |
| `make backend` | 启动 FastAPI，带 `--reload` |
| `make frontend` | 启动 Vite 开发服务器 |
| `make dev` | 同时启动前后端 |
| `make build` | 构建前端生产包 |
| `make lint` | 运行前端 ESLint |
| `make verify` | 运行轻量校验 |
| `make clean` | 清理常见构建产物 |

---

### 8. 后端说明

#### 8.1 入口文件

- `backend/main.py`

#### 8.2 核心模块

- `backend/core/config.py`
  - 应用配置
  - 默认模型路径
  - 默认阈值
  - 数据库路径
- `backend/core/database.py`
  - SQLAlchemy async engine / session
  - 数据库初始化
- `backend/core/security.py`
  - JWT 编码/解码
  - 密码哈希与验证
- `backend/core/dependencies.py`
  - `get_current_user`

#### 8.3 服务层

- `backend/services/yolo_service.py`
  - 单例 YOLO 服务
  - 加载模型
  - 执行检测
  - 返回当前模型名和类别名
- `backend/services/export_service.py`
  - 保存带框图片
  - 导出 CSV

#### 8.4 数据与产物

运行过程中通常会使用或生成这些内容：

- `backend/app.db`
- `backend/static/uploads/`
- `backend/static/exports/`
- `backend/static/avatars/`

---

### 9. 前端说明

#### 9.1 页面

- `frontend/src/pages/LoginPage.tsx`
  - 登录 / 注册
- `frontend/src/pages/MainPage.tsx`
  - 主识别页面
  - 图片/视频/摄像头/模型入口
  - 模型选择弹窗
  - CSV 导出

#### 9.2 核心组件

- `frontend/src/components/layout/Sidebar.tsx`
  - 左侧功能入口
  - 修改密码弹窗
- `frontend/src/components/layout/TitleBar.tsx`
  - 语言切换
  - 导出 CSV
- `frontend/src/components/detection/*`
  - 画布、进度条、结果表、检测信息

#### 9.3 状态与连接

- `frontend/src/hooks/useAuth.ts`
- `frontend/src/hooks/useDetection.ts`
- `frontend/src/hooks/useCamera.ts`
- `frontend/src/hooks/useWebSocket.ts`

---

### 10. API 说明

#### 10.1 健康检查

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/health` | 检查服务状态和模型是否加载 |

#### 10.2 鉴权

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/auth/register` | 注册 |
| `POST` | `/api/auth/login` | 登录 |
| `GET` | `/api/auth/me` | 当前用户信息 |
| `PUT` | `/api/auth/change-password` | 修改密码 |
| `POST` | `/api/auth/avatar` | 上传头像 |
| `DELETE` | `/api/auth/account` | 删除账户 |

#### 10.3 识别

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/detect/image` | 图片识别 |
| `POST` | `/api/detect/video` | 上传并异步处理视频 |
| `GET` | `/api/detect/video/{task_id}/status` | 查询视频处理进度 |
| `GET` | `/api/detect/video/{task_id}/results` | 获取视频检测结果 |
| `GET` | `/api/detect/export/csv/{task_id}` | 导出视频结果 CSV |

#### 10.4 模型管理

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/detect/model/list` | 获取模型列表 |
| `POST` | `/api/detect/model/select` | 切换当前模型 |
| `POST` | `/api/detect/model/upload` | 上传并加载新模型 |
| `GET` | `/api/detect/model/classes` | 获取当前模型类别 |

#### 10.5 WebSocket

| 协议 | 路径 | 说明 |
|---|---|---|
| `WS` | `/ws/detect?token=JWT` | 实时摄像头识别 |

消息约定：

- 客户端 -> 服务端
  - 二进制 JPEG 帧
  - 文本配置：`{"type":"config","conf":0.35,"iou":0.45}`
- 服务端 -> 客户端
  - `{"detections":[...],"inference_time":0.034,"frame_id":42}`

---

### 11. 模型说明

默认配置在：

- `backend/core/config.py`

当前默认模型路径：

```python
model_path = "weights/best-yolov8n.pt"
```

项目支持：

- 从已有 `.pt` 文件中切换模型
- 上传新的 `.pt` 模型
- 启动时自动尝试加载默认模型

当前权重一般可放在：

- `backend/weights/`
- `weights/`

---

### 12. 配置说明

后端通过 `Settings` 管理主要配置，典型项包括：

- `secret_key`
- `algorithm`
- `access_token_expire_minutes`
- `model_path`
- `device`
- `default_conf`
- `default_iou`
- `image_size`
- `database_url`
- `upload_dir`
- `export_dir`
- `avatar_dir`
- `max_upload_size`

如果你需要覆盖默认值，可以使用 `.env`。

---

### 13. 已知限制

- `Folder` 当前不是完整文件夹批量处理，只会处理选择结果中的第一张图片
- 视频处理任务保存在内存中，服务重启后任务状态不会保留
- 当前前端导出只支持视频结果 CSV，不支持图片结果导出按钮
- 模型列表和切换功能依赖后端已重启到新版本

---

### 14. 适合继续扩展的方向

- 完整文件夹批处理
- 视频结果可视化回放
- 模型元信息展示
- 用户头像入口前端化
- 更细粒度的权限和用户管理
- 任务持久化与队列化

---

### 15. 许可证

本项目仅用于个人学习与研究。

---

## English Guide

### 1. Overview

This repository contains a **web-based hand gesture recognition system** built with **YOLO, FastAPI, and React**. 

The system currently supports:

- User registration and login
- JWT-based authentication
- Password change
- Image detection
- Video detection
- Real-time webcam detection through WebSocket
- Model upload and model switching
- CSV export for video results
- Frontend language switching

Architecture:

- Backend: FastAPI for auth, inference, export, and WebSocket handling
- Frontend: React + TypeScript + Vite for UI and interaction
- Model runtime: YOLO `.pt` weights
- Database: SQLite

---

### 2. Features

#### 2.1 Implemented

- Register / login flow
- JWT auth
- Change password
- Detect from uploaded images
- Detect from uploaded videos
- Real-time webcam inference
- Select an existing model
- Upload a new `.pt` model and switch to it
- Export video results as CSV
- Bundled i18n resources for Chinese / English / Japanese

#### 2.2 Current frontend entry points

- Sidebar:
  - `Camera`
  - `Image`
  - `Video`
  - `Folder`
  - `Model`
  - `Change Password`
  - User logout
- Top bar:
  - Language switch
  - `Export CSV`

#### 2.3 Current behavior notes

- `Export CSV` is only meaningful for video tasks
- `Model` opens the model dialog for listing, switching, and uploading models
- `Folder` currently processes only the first selected image, not full batch processing
- If model list loading fails after code changes, restart the backend process first

---

### 3. Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, TypeScript, Vite |
| State | Zustand |
| Backend | FastAPI, Uvicorn |
| Auth | JWT, bcrypt / passlib |
| Database | SQLite, SQLAlchemy async, aiosqlite |
| Vision | Ultralytics YOLO, OpenCV, PyTorch |
| Real-time | WebSocket |
| i18n | react-i18next |

---

### 4. Project Structure

```text
.
├── backend/                  # FastAPI backend
│   ├── core/                 # config, database, dependencies, security
│   ├── models/               # Pydantic schemas / SQLAlchemy model
│   ├── routers/              # auth / detection / websocket routers
│   ├── services/             # YOLO and export services
│   ├── weights/              # default model weights
│   ├── app.db                # SQLite database
│   └── requirements.txt
│
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── i18n/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── types/
│   │   └── utils/
│   ├── package.json
│   └── vite.config.ts
│
├── output_examples/          # sample output files
├── scripts/                  # small helper scripts
├── test_media/               # test images and videos
├── ultralytics/              # YOLO-related code kept in the repo
├── weights/                  # extra local model weights
├── Makefile                  # common project commands
└── start.sh                  # one-command startup script
```

---

### 5. Requirements

- Python 3.9+
- Node.js 18+
- A local virtual environment at `./venv`
- Installed frontend dependencies
- At least one available model weight file:
  - `backend/weights/*.pt`

---

### 6. Quick Start

#### 6.1 Environment setup

```bash
python3 -m venv venv
./venv/bin/pip install -r backend/requirements.txt
cd frontend && npm install
```

#### 6.2 Using Makefile

```bash
make setup
make backend
make frontend
make dev
make build
make lint
make verify
make clean
```

#### 6.3 Start everything

```bash
./start.sh
```

or:

```bash
make dev
```

Default local endpoints:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

---

### 7. Command Reference

| Command | Description |
|---|---|
| `make setup` | Install backend and frontend dependencies |
| `make setup-backend` | Install backend dependencies into `venv` |
| `make setup-frontend` | Install frontend dependencies |
| `make backend` | Run FastAPI with `--reload` |
| `make frontend` | Run the Vite dev server |
| `make dev` | Start backend and frontend together |
| `make build` | Build the frontend production bundle |
| `make lint` | Run frontend ESLint |
| `make verify` | Run lightweight checks |
| `make clean` | Remove common generated files |

---

### 8. Backend Notes

#### 8.1 Entry point

- `backend/main.py`

#### 8.2 Core modules

- `backend/core/config.py`
  - app settings
  - default model path
  - default thresholds
  - database path
- `backend/core/database.py`
  - SQLAlchemy async engine / session
  - DB initialization
- `backend/core/security.py`
  - JWT encode / decode
  - password hashing and verification
- `backend/core/dependencies.py`
  - `get_current_user`

#### 8.3 Services

- `backend/services/yolo_service.py`
  - singleton YOLO service
  - model loading
  - inference
  - current model and class name access
- `backend/services/export_service.py`
  - save annotated images
  - export CSV results

#### 8.4 Runtime data

The app typically uses or generates:

- `backend/app.db`
- `backend/static/uploads/`
- `backend/static/exports/`
- `backend/static/avatars/`

---

### 9. Frontend Notes

#### 9.1 Pages

- `frontend/src/pages/LoginPage.tsx`
  - login / registration
- `frontend/src/pages/MainPage.tsx`
  - main detection page
  - image / video / webcam / model entry points
  - model selection dialog
  - CSV export

#### 9.2 Key components

- `frontend/src/components/layout/Sidebar.tsx`
  - left action menu
  - change password dialog
- `frontend/src/components/layout/TitleBar.tsx`
  - language switch
  - CSV export
- `frontend/src/components/detection/*`
  - canvas, progress bar, results table, detection info

#### 9.3 State and connectivity

- `frontend/src/hooks/useAuth.ts`
- `frontend/src/hooks/useDetection.ts`
- `frontend/src/hooks/useCamera.ts`
- `frontend/src/hooks/useWebSocket.ts`

---

### 10. API Reference

#### 10.1 Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Check service and model status |

#### 10.2 Auth

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/auth/register` | Register a new user |
| `POST` | `/api/auth/login` | Login |
| `GET` | `/api/auth/me` | Current user info |
| `PUT` | `/api/auth/change-password` | Change password |
| `POST` | `/api/auth/avatar` | Upload avatar |
| `DELETE` | `/api/auth/account` | Delete account |

#### 10.3 Detection

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/detect/image` | Detect from an image |
| `POST` | `/api/detect/video` | Upload and process a video asynchronously |
| `GET` | `/api/detect/video/{task_id}/status` | Query video task progress |
| `GET` | `/api/detect/video/{task_id}/results` | Get video detection results |
| `GET` | `/api/detect/export/csv/{task_id}` | Export video results as CSV |

#### 10.4 Model management

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/detect/model/list` | List available models |
| `POST` | `/api/detect/model/select` | Switch the current model |
| `POST` | `/api/detect/model/upload` | Upload and load a new model |
| `GET` | `/api/detect/model/classes` | Get classes from the current model |

#### 10.5 WebSocket

| Protocol | Path | Description |
|---|---|---|
| `WS` | `/ws/detect?token=JWT` | Real-time webcam detection |

Message contract:

- Client -> Server
  - binary JPEG frames
  - config message: `{"type":"config","conf":0.35,"iou":0.45}`
- Server -> Client
  - `{"detections":[...],"inference_time":0.034,"frame_id":42}`

---

### 11. Model Notes

The default configuration is defined in:

- `backend/core/config.py`

Current default path:

```python
model_path = "weights/best-yolov8n.pt"
```

The project supports:

- switching between existing `.pt` files
- uploading a new `.pt` model
- auto-loading the default model at startup

Typical model locations:

- `backend/weights/`
- `weights/`

---

### 12. Configuration

The backend uses `Settings` for core configuration values, including:

- `secret_key`
- `algorithm`
- `access_token_expire_minutes`
- `model_path`
- `device`
- `default_conf`
- `default_iou`
- `image_size`
- `database_url`
- `upload_dir`
- `export_dir`
- `avatar_dir`
- `max_upload_size`

Use `.env` if you need to override defaults.

---

### 13. Known Limitations

- `Folder` is not full batch inference yet; it only processes the first selected image
- Video task state is stored in memory and does not survive server restarts
- The current export button only supports video-result CSV export
- Model listing and switching require the backend to be restarted after route changes

---

### 14. Good Next Steps

- full folder batch processing
- video playback with result overlays
- model metadata display
- frontend avatar management
- more complete user management
- persistent task storage / queueing

---

### 15. License

This project is for personal learning and research purposes only.
