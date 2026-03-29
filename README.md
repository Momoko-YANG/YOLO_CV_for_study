# YOLO Hand Gesture Recognition

基于 **YOLOv8 / YOLOv5 + FastAPI + React** 的手势识别 Web 系统。

## 功能概览

- 用户注册 / 登录（JWT 鉴权）、修改密码、上传头像、删除账号
- 图片上传识别
- 视频上传异步处理
- 摄像头实时识别（WebSocket）
- 模型列表查看、切换、上传
- 视频检测结果导出 CSV
- 前端支持中 / 英 / 日三语切换
- 多环境配置（dev / staging / prod）
- Docker Compose 一键部署
- CI 自动化（Ruff lint + ESLint + pytest + Docker build）

## 技术栈

| 层 | 技术 |
|---|---|
| 前端 | React 19 · TypeScript · Vite · Tailwind CSS · Zustand · react-i18next |
| 后端 | FastAPI · Uvicorn · SQLAlchemy (async) · aiosqlite · python-jose · passlib |
| 模型 | Ultralytics YOLO · OpenCV · PyTorch (CUDA / MPS / CPU) |
| 实时通信 | WebSocket |
| 数据库 | SQLite |
| CI/CD | GitHub Actions · Docker Compose · Ruff · ESLint · pytest |
| 实验跟踪 | MLflow（可选） |

## 目录结构

```
.
├── backend/                  # FastAPI 后端
│   ├── core/                 # 配置、数据库、鉴权、依赖注入
│   ├── models/               # Pydantic 模型 + SQLAlchemy 实体
│   ├── routers/              # auth · detection · websocket 路由
│   ├── services/             # YOLO 推理服务 · 导出服务
│   ├── tests/                # pytest 测试
│   ├── weights/              # 模型权重 (.pt)
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                 # React 前端
│   ├── src/
│   │   ├── components/       # Sidebar · TitleBar · DetectionCanvas 等
│   │   ├── hooks/            # useAuth · useCamera · useWebSocket · useDetection
│   │   ├── i18n/             # zh.json · en.json · ja.json
│   │   ├── pages/            # LoginPage · MainPage
│   │   ├── services/         # Axios API 封装
│   │   ├── types/            # TypeScript 类型
│   │   └── utils/            # Canvas 绘制工具
│   ├── Dockerfile
│   └── vite.config.ts
├── training/                 # 模型训练 Pipeline
│   ├── train.py              # 微调训练
│   ├── evaluate.py           # 模型评估
│   ├── benchmark.py          # 推理基准测试
│   ├── export_onnx.py        # ONNX 导出
│   ├── validate_dataset.py   # 数据集质量校验
│   ├── compare.py            # 实验结果对比
│   ├── dataset.yaml          # 数据集配置
│   ├── configs/              # 实验超参配置
│   └── notebooks/            # EDA Notebook
├── .env.example              # 环境变量模板
├── .github/workflows/ci.yml  # GitHub Actions CI
├── docker-compose.yml        # Docker 编排
├── Makefile                  # 常用命令
├── start.sh                  # 一键启动脚本
└── pyproject.toml            # 项目元数据 + Ruff/pytest 配置
```

## 快速开始

### 环境要求

- Python 3.9+
- Node.js 18+
- 至少一个 `.pt` 模型权重文件放在 `backend/weights/`

### 安装依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

cd frontend && npm install && cd ..
```

或使用 Makefile：

```bash
make setup
```

### 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入实际值（开发环境可直接使用默认值）
```

### 启动服务

```bash
# 一键启动前后端
./start.sh

# 或分别启动
make backend   # 后端 http://localhost:8000
make frontend  # 前端 http://localhost:5173
```

启动后访问：

| 服务 | 地址 |
|---|---|
| 前端界面 | http://localhost:5173 |
| 后端 API | http://localhost:8000 |
| API 文档 | http://localhost:8000/docs （prod 环境自动关闭） |

### Docker 部署

```bash
docker compose up --build
```

包含三个服务：`backend`（FastAPI）、`frontend`（Vite）、`mlflow`（实验跟踪）。

## 环境配置

后端通过 `pydantic-settings` 管理配置，所有参数可通过 `.env` 文件或环境变量覆盖。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `APP_ENV` | `dev` | 运行环境：`dev` / `staging` / `prod` |
| `DEBUG` | `false` | 调试模式 |
| `SECRET_KEY` | 开发专用密钥 | JWT 签名密钥，staging/prod **必须**覆盖 |
| `CORS_ORIGINS` | `http://localhost:5173,...` | 允许的前端来源（逗号分隔） |
| `MODEL_PATH` | `weights/best-yolov8n.pt` | 默认模型路径 |
| `DEVICE` | 自动检测 | 推理设备：`cuda:0` / `mps` / `cpu` |
| `DEFAULT_CONF` | `0.25` | 默认置信度阈值 |
| `DEFAULT_IOU` | `0.5` | 默认 IoU 阈值 |
| `DATABASE_URL` | `sqlite+aiosqlite:///./app.db` | 数据库连接 |

> staging/prod 环境启动时如果未设置 `SECRET_KEY`，后端会直接报错拒绝启动。生成密钥：
> ```bash
> python -c "import secrets; print(secrets.token_urlsafe(32))"
> ```

## API 接口

### 鉴权

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/auth/register` | 注册新用户 |
| `POST` | `/api/auth/login` | 登录，返回 JWT |
| `GET` | `/api/auth/me` | 获取当前用户信息 |
| `PUT` | `/api/auth/change-password` | 修改密码 |
| `POST` | `/api/auth/avatar` | 上传头像 |
| `DELETE` | `/api/auth/account` | 删除账号 |

### 检测

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/detect/image` | 图片检测 |
| `POST` | `/api/detect/video` | 上传视频并异步处理 |
| `GET` | `/api/detect/video/{task_id}/status` | 视频处理进度 |
| `GET` | `/api/detect/video/{task_id}/results` | 视频检测结果 |
| `GET` | `/api/detect/export/csv/{task_id}` | 导出 CSV |

### 模型管理

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/detect/model/list` | 列出可用模型 |
| `GET` | `/api/detect/model/classes` | 获取当前模型类别名 |
| `POST` | `/api/detect/model/select` | 切换模型 |
| `POST` | `/api/detect/model/upload` | 上传新模型并加载 |

### WebSocket 实时检测

| 协议 | 路径 | 说明 |
|---|---|---|
| `WS` | `/ws/detect?token=<JWT>` | 摄像头实时检测 |

**消息协议：**

```
客户端 → 服务端：二进制 JPEG 帧 / {"type":"config","conf":0.3,"iou":0.4}
服务端 → 客户端：{"detections":[...],"inference_time":0.034,"frame_id":42}
```

### 健康检查

```
GET /api/health → {"status":"ok","model_loaded":true,"classes":[...]}
```

## 模型训练

项目包含完整的 YOLO 微调、评估、对比 Pipeline，位于 `training/` 目录。

### 准备数据集

1. 准备 YOLO 格式标注（每张图片一个 `.txt`，每行 `class_id cx cy w h`）
2. 按 train/val 划分
3. 修改 `training/dataset.yaml`

### 训练

```bash
# 默认配置
python training/train.py

# 使用预设实验配置
python training/train.py --config training/configs/finetune_v8s.yaml

# 自定义参数
python training/train.py --model yolov8m.pt --epochs 200 --freeze 0 --lr0 0.0005
```

预设配置：

| 配置文件 | 策略 | 场景 |
|---|---|---|
| `finetune_v8s.yaml` | YOLOv8s + 冻结 backbone | 推荐起步 |
| `finetune_v8m.yaml` | YOLOv8m + 冻结 backbone | 追求更高精度 |
| `finetune_v8s_fullunfreeze.yaml` | YOLOv8s + 全部解冻 | 数据充足时 |

### 评估与对比

```bash
# 评估单个模型
python training/evaluate.py --model runs/gesture/exp1/weights/best.pt

# 与基线对比
python training/evaluate.py --model best.pt --baseline yolov8s.pt

# 汇总所有实验
python training/compare.py --output results.csv
```

### 其他工具

```bash
# 数据集质量校验
python training/validate_dataset.py --data-root datasets/RPS

# 推理基准测试
python training/benchmark.py --models best-yolov8n.pt best-yolov5nu.pt

# ONNX 导出
python training/export_onnx.py --model best.pt
```

### 部署训练好的模型

训练完成后 `best.pt` 自动复制到 `backend/weights/`。通过前端模型管理面板或修改 `.env` 中的 `MODEL_PATH` 即可切换。

## 开发命令

| 命令 | 说明 |
|---|---|
| `make setup` | 安装前后端全部依赖 |
| `make backend` | 启动后端（带热重载） |
| `make frontend` | 启动前端开发服务器 |
| `make dev` | 同时启动前后端（`start.sh`） |
| `make build` | 构建前端生产包 |
| `make lint` | 运行 Ruff + ESLint |
| `make test` | 运行后端单元/集成测试 |
| `make test-quality` | 运行模型质量门禁测试 |
| `make mlflow` | 启动 MLflow 跟踪服务 |
| `make validate-data` | 运行数据集质量检查 |
| `make verify` | 轻量级本地校验 |
| `make clean` | 清理构建产物 |

## CI/CD

GitHub Actions 自动运行（`.github/workflows/ci.yml`）：

1. **Lint** — `ruff check backend/ training/` + `npm run lint`
2. **Test** — `pytest backend/tests/`（排除模型质量门禁）
3. **Docker Build** — `docker compose build`

## 已知限制

- `Folder` 功能当前只处理选中的第一张图片，非完整批处理
- 视频任务状态保存在内存中，服务重启后丢失
- CSV 导出仅支持视频检测结果

## License

本项目仅用于个人学习与研究。
