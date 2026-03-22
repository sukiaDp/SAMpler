# SAMpler Web 前端设计文档

**日期**：2026-03-23
**分支**：`feat/web-frontend`
**范围**：将现有 Gradio 应用迁移为 FastAPI 后端 + 原生 HTML 静态前端的 BS 架构

---

## 1. 背景与目标

现有 `app.py` 是一个 1068 行的 Gradio 单文件应用，功能完整但 UI 灵活性低、样式定制困难、移动端适配差。本次迁移目标：

- 后端暴露 REST API，前端独立开发，两者解耦
- 保留全部业务功能（自动标注、预览编辑、YOLO 训练、推理测试）
- 前端可自由替换（当前为原生 HTML，未来可换 Vue/React）
- 保留 `app.py`（Gradio 版）不删除，逐步废弃

---

## 2. 整体架构

```
┌─────────────────────┐     HTTP / SSE      ┌──────────────────────┐
│   前端 (Browser)     │ ◄─────────────────► │   后端 (FastAPI)      │
│                     │                     │                      │
│  原生 HTML + JS      │                     │  REST API            │
│  （ES Modules）      │                     │  SSE 训练日志         │
│                     │                     │  StaticFiles 托管     │
└─────────────────────┘                     └──────────┬───────────┘
                                                       │
                                              ┌────────▼────────┐
                                              │   核心逻辑层     │
                                              │  sam3.py        │
                                              │  ultralytics    │
                                              └─────────────────┘
```

**运行方式**：`python run.py`，访问 `http://localhost:8000`

---

## 3. 目录结构

```
SAM3/
├── run.py               # 启动入口（唯一需要运行的文件）
├── backend/
│   ├── main.py          # FastAPI 应用入口，注册路由 + 挂载 StaticFiles
│   ├── routers/
│   │   ├── images.py    # 图片列表、删除图片及标注
│   │   ├── annotate.py  # SAM3 标注任务启动与查询
│   │   ├── preview.py   # 预览图获取、单个标注实例删除
│   │   ├── train.py     # YOLO 训练任务 + SSE 日志流
│   │   └── infer.py     # YOLO 推理、模型信息查询
│   ├── tasks.py         # 后台任务注册表（task_id → 状态/进度/结果）
│   └── models.py        # Pydantic 请求/响应数据模型
├── frontend/
│   ├── index.html       # 单页应用入口（shell 结构）
│   ├── app.js           # 页面路由 + 全局状态 + API 客户端
│   ├── style.css        # Apple 风格主题，CSS 变量，深/浅色切换
│   └── views/
│       ├── annotate.js  # 自动标注页逻辑
│       ├── preview.js   # 预览编辑页逻辑
│       ├── train.js     # YOLO 训练页逻辑（含 SSE 日志）
│       └── infer.js     # 推理测试页逻辑
├── sam3.py              # 核心分割模块（不变）
├── app.py               # Gradio 版（保留，逐步废弃）
└── docs/
    ├── web-frontend-migration.md
    └── superpowers/specs/
        └── 2026-03-23-web-frontend-design.md
```

---

## 4. 启动器 `run.py`

职责：
1. 设置环境变量（`CUBLAS_WORKSPACE_CONFIG=:4096:8` 等）
2. 校验 `backend/` 和 `frontend/` 目录存在
3. 调用 `uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)`
4. 打印访问地址

---

## 5. 后端 API

### 5.1 端点一览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/images` | 列出目录图片，返回 `{files, total}` |
| `GET` | `/api/images/{id}/preview` | 返回预览图 URL + 原始标注数据列表 |
| `DELETE` | `/api/images/{id}` | 删除图片及对应标注文件 |
| `DELETE` | `/api/images/{id}/annotations/{ann_id}` | 删除单个标注实例，重写标注文件 |
| `POST` | `/api/annotate` | 启动 SAM3 标注任务，返回 `task_id` |
| `GET` | `/api/tasks/{task_id}` | 查询任务状态（`pending/running/done/error`）、进度、错误信息（轮询） |
| `POST` | `/api/train` | 启动 YOLO 训练任务，返回 `task_id` |
| `GET` | `/api/train/{task_id}/logs` | SSE 流，实时推送训练日志文本（与轮询并行，用途不同） |
| `POST` | `/api/infer` | 推理单张图片（`multipart/form-data` 上传），返回结果图 URL + 检测统计 |
| `GET` | `/api/model-info` | 查询 .pt 文件版本/架构/大小信息 |
| `POST` | `/api/segment` | 单图 SAM3 分割（`multipart/form-data` 上传，测试用） |

**任务轮询与 SSE 的关系**：`GET /api/tasks/{task_id}` 返回任务状态和进度（适用于标注和训练）；`GET /api/train/{task_id}/logs` 是独立的 SSE 长连接，只用于推送训练的文字日志行，两者是并行机制，由不同 handler 处理。

### 5.2 请求/响应 Schema

**`GET /api/images?dir=<path>`**

Query: `dir` (string) — 图片目录路径

Response:
```json
{
  "files": [
    {"id": "img_0042", "filename": "img_0042.jpg", "has_label": true}
  ],
  "total": 48
}
```
`id` 为不含扩展名的文件名（如 `img_0042`），用于构造后续 URL。

**`POST /api/annotate`**

Request (JSON):
```json
{
  "image_dir": "rawData",
  "output_dir": "dataset",
  "prompts": "person, car, dog",
  "mode": "segment",
  "sort_mode": "conf",
  "conf": 0.25,
  "val_ratio": 0.1,
  "max_instances": 7
}
```
`mode` 枚举：`"detect"` | `"segment"`

Response: `{"task_id": "abc123"}`

**`POST /api/train`**

Request (JSON):
```json
{
  "dataset_dir": "dataset",
  "task": "segment",
  "yolo_version": "yolo11",
  "model_size": "n",
  "epochs": 100,
  "imgsz": 640
}
```
`dataset_dir` 和 `task` 由前端从标注配置中显式传入，不依赖服务器全局状态。

Response: `{"task_id": "def456"}`

**`POST /api/infer`** (`multipart/form-data`)

Fields: `image` (file), `weights_path` (string), `conf` (float), `imgsz` (int)

Response:
```json
{
  "result_url": "/previews/infer_result_abc.jpg",
  "stats": {"total": 3, "classes": {"person": 2, "car": 1}}
}
```

**`POST /api/segment`** (`multipart/form-data`)

Fields: `image` (file), `prompts` (string), `conf` (float), `max_instances` (int), `sort_mode` (string)

Response: 同 `GET /api/images/{id}/preview`

### 5.3 任务管理

标注和训练为长耗时操作，统一采用 **后台线程 + task_id 轮询** 模式：

```
POST /api/annotate  →  {"task_id": "abc123"}
GET  /api/tasks/abc123  →  {"status": "running", "progress": 12, "total": 48}
GET  /api/tasks/abc123  →  {"status": "done", "result": {...}}
```

`tasks.py` 维护一个内存字典 `{task_id: TaskState}`，线程安全（`threading.Lock`）。

### 5.4 图片数据约定

- 预览图保存到 `.cache/previews/`，FastAPI 用 `StaticFiles` 将 `/previews` 路由挂载到 `.cache/previews/` 目录，API 返回相对 URL（如 `/previews/img_0042.jpg`），前端直接用 `<img src="/previews/img_0042.jpg">` 加载
- 不在 JSON 中内嵌 base64
- 图片上传（推理/单图分割）使用 `multipart/form-data`
- `/api/segment` 同样将结果图保存到 `.cache/previews/` 并返回 URL，格式同 preview 响应

### 5.5 Canvas 预留接口

`GET /api/images/{id}/preview` 响应同时包含：

```json
{
  "preview_url": "/previews/img_0042.jpg",
  "annotations": [
    {
      "id": 0,
      "class_name": "person",
      "type": "polygon",
      "color": [48, 209, 88],
      "points": [[0.12, 0.34], ...],
      "bbox": [0.10, 0.30, 0.25, 0.40]
    }
  ]
}
```

> **Canvas 升级注意**：当前前端使用 `preview_url` 展示后端渲染图。如需切换到前端 Canvas 绘制，使用 `annotations` 字段自行绘制，`preview_url` 可停用。接口无需改动。

### 5.6 错误格式

```json
{"error": "简短说明", "detail": "可选的详细信息"}
```

HTTP 状态码：`400` 参数错误，`404` 资源不存在，`500` 服务器内部错误。

---

## 6. 前端设计

### 6.1 页面结构

```
┌─────────────────────────────────────────────────────┐
│  顶栏（固定）: Logo · 当前页标题        ☀/🌙 主题切换 │
├──────────────┬──────────────────────────────────────┤
│  侧边栏       │  内容区                               │
│  （固定）     │  （由 app.js 根据导航项切换视图）       │
│              │                                      │
│  🏷 自动标注  │                                      │
│  🖼 预览编辑  │                                      │
│  ⚡ YOLO训练  │                                      │
│  🔍 推理测试  │                                      │
└──────────────┴──────────────────────────────────────┘
```

### 6.2 视觉规范

| 属性 | 深色模式 | 浅色模式 |
|------|---------|---------|
| 背景 | `#1c1c1e` | `#f2f2f7` |
| 侧边栏 | `rgba(44,44,46,0.9)` | `rgba(255,255,255,0.75)` |
| 主文字 | `#ebebf5` | `#1c1c1e` |
| 次要文字 | `rgba(235,235,245,0.4)` | `rgba(60,60,67,0.5)` |
| 分割线 | `rgba(255,255,255,0.07)` | `rgba(0,0,0,0.08)` |
| 主按钮 | `rgba(235,235,245,0.9)` 白底黑字 | `#1c1c1e` 黑底白字 |
| 危险按钮 | 透明底 + 红色边框/文字 | 同左 |

- **毛玻璃**：侧边栏和顶栏使用 `backdrop-filter: blur(20px)`
- **圆角**：卡片/输入框 8px，按钮 7px，小徽章 12px
- **字体**：`-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif`
- **主题切换**：`prefers-color-scheme` 自动跟随系统，顶栏按钮可手动覆盖，存 `localStorage`

### 6.3 预览编辑页布局

```
┌─────────────────────────────────────────────────┐
│  文件名                          ◀  12/48  ▶  🗑 │  ← 工具栏
├─────────────────────────────────────────────────┤
│                                                 │
│              预览图（后端渲染）                   │  ← ~70% 高度
│                                                 │
├─────────────────────────────────────────────────┤
│  标注:  🟢person  🟠car  🟣dog    [删除选中]     │  ← 横排胶囊标签
└─────────────────────────────────────────────────┘
```

胶囊标签可多选（click toggle），勾选后"删除选中"按钮激活，调用 `DELETE /api/images/{id}/annotations/{ann_id}`。

### 6.4 单页路由

`app.js` 监听侧边栏点击，`show/hide` 各 view 容器，无 hash 路由，无页面跳转。各 view 在首次显示时懒初始化。

### 6.5 API 客户端

`app.js` 中封装统一的 `api(path, options)` 函数，处理：
- `baseURL` 统一前缀
- 错误响应自动解析并弹出 toast 通知
- 上传进度回调

---

## 7. 错误处理

| 场景 | 后端行为 | 前端展示 |
|------|---------|---------|
| 参数错误 | 400 + error JSON | toast 提示 3 秒自动消失 |
| 文件不存在 | 404 + error JSON | toast 提示，建议检查路径 |
| 任务失败 | task status=error | 红色提示条 + 错误详情 |
| GPU/推理异常 | 500 + error JSON | toast 提示 + 建议检查显存 |
| 网络断开 | fetch 抛出异常 | toast 提示"无法连接服务器" |
| 训练任务失败（SSE） | 推送特殊事件 `event: error\ndata: <原因>` 后关闭流 | 前端监听 `error` 事件类型，展示红色提示条 |

---

## 8. 迁移阶段

### Phase 1（本分支，当前）— 后端 API + 静态前端
- 搭建 FastAPI 骨架，实现所有 API 端点
- 实现原生 HTML 前端（Apple 风格，跟随系统主题）
- `run.py` 启动入口

### Phase 2（后续）— Canvas 交互升级（可选）
- 前端用 Canvas 自绘标注，替代后端渲染图
- 支持标注实例高亮 hover
- 接口已预留，无需改动后端

### Phase 3（后续）— 框架升级（可选）
- 将 `frontend/` 替换为 Vue 3 / React 项目
- 后端不动

---

## 9. 关键约束

- `sam3.py` 核心模块不改动
- SAM3 单例模式（`get_segmentor()`）迁移到 `backend/` 中保留
- 本地单用户，无需认证，CORS 允许 `localhost`
- 依赖：`fastapi`, `uvicorn[standard]` 加入 `requirements.txt`
