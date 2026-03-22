# 前端迁移方案：Gradio → BS 架构

## 背景

当前应用使用 Gradio 构建界面，优点是快速原型，但存在明显限制：
- UI 灵活性低，难以实现复杂交互
- 样式定制困难
- 无法自由选择前端框架
- 移动端适配差

本分支目标：将前端从 Gradio 剥离，改为标准 **Browser/Server 分离架构**：后端暴露 REST API，前端独立开发，两者解耦。

---

## 架构设计

```
┌─────────────────────┐        HTTP / WebSocket       ┌──────────────────────┐
│     前端 (Browser)   │ ◄────────────────────────── ► │    后端 (Server)      │
│                     │                               │                      │
│  任意框架可选：       │                               │  FastAPI             │
│  · Vue 3 + Vite     │                               │  · /api/annotate     │
│  · React            │                               │  · /api/train        │
│  · 原生 HTML/JS      │                               │  · /api/infer        │
│                     │                               │  · /api/preview      │
└─────────────────────┘                               └──────────────────────┘
                                                              │
                                                     ┌────────▼────────┐
                                                     │   核心逻辑层     │
                                                     │  sam3.py        │
                                                     │  ultralytics    │
                                                     └─────────────────┘
```

---

## 后端 API 规划

后端使用 **FastAPI**，所有长耗时操作通过 **任务队列 + 轮询** 或 **WebSocket** 推送进度。

### 端点一览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/images` | 列出指定目录下的图片列表 |
| `POST` | `/api/annotate` | 启动 SAM3 自动标注任务 |
| `GET` | `/api/annotate/{task_id}` | 查询标注任务进度 |
| `GET` | `/api/preview/{image_id}` | 获取单张图片的标注预览（base64 图 + 标注列表） |
| `DELETE` | `/api/annotation/{image_id}/{ann_id}` | 删除单个标注实例 |
| `DELETE` | `/api/image/{image_id}` | 删除图片及其标注文件 |
| `POST` | `/api/train` | 启动 YOLO 训练任务 |
| `GET` | `/api/train/{task_id}` | 查询训练任务进度（支持 SSE 流式日志） |
| `POST` | `/api/infer` | 使用 YOLO 模型推理单张图片 |
| `GET` | `/api/model-info` | 查询 .pt 文件的模型版本/大小信息 |
| `POST` | `/api/segment` | 单图 SAM3 分割（用于单图测试标签页） |
| `GET` | `/static/{path}` | 静态文件服务（前端构建产物） |

### 数据格式约定

- 所有请求/响应使用 `application/json`
- 图片上传使用 `multipart/form-data`
- 图片数据返回 base64 或 URL，由前端决定渲染方式
- 错误统一格式：`{"error": "message", "detail": "..."}`

---

## 目录结构规划

```
SAM3/
├── backend/
│   ├── main.py          # FastAPI 应用入口
│   ├── routers/
│   │   ├── annotate.py  # 标注相关路由
│   │   ├── train.py     # 训练相关路由
│   │   ├── infer.py     # 推理相关路由
│   │   └── preview.py   # 预览/编辑路由
│   ├── tasks.py         # 后台任务管理（异步任务队列）
│   └── models.py        # Pydantic 数据模型
├── frontend/            # 前端代码（框架待定，见下节）
│   ├── index.html
│   └── ...
├── sam3.py              # 核心分割模块（不变）
├── app.py               # 保留 Gradio 版本（逐步废弃）
└── docs/
    └── web-frontend-migration.md  # 本文档
```

---

## 前端框架选型

当前尚未决定，以下为候选方案：

| 方案 | 优点 | 缺点 |
|------|------|------|
| **Vue 3 + Vite** | 轻量、中文社区好、组合式 API 直观 | 生态略小于 React |
| **React + Vite** | 生态最大、组件库丰富 | 样板代码较多 |
| **原生 HTML + Alpine.js** | 零构建依赖、极简 | 复杂交互难维护 |

> 推荐：**Vue 3 + Vite**，配合 Element Plus 组件库，适合图片预览/标注类工具。

---

## 迁移阶段计划

### Phase 1 — 后端 API（当前阶段）
- [ ] 搭建 FastAPI 项目骨架
- [ ] 实现所有 API 端点（与 `app.py` 业务逻辑等价）
- [ ] 异步任务管理（标注/训练为长耗时操作）
- [ ] 单元测试覆盖核心端点

### Phase 2 — 前端基础
- [ ] 确定前端框架，初始化项目
- [ ] 实现标注标签页（目录选择 → 启动标注 → 预览/编辑）
- [ ] 实现 YOLO 训练标签页（配置 → 启动 → 实时日志）
- [ ] 实现 YOLO 推理标签页（上传 → 推理 → 结果展示）

### Phase 3 — 迁移完成
- [ ] 前后端联调
- [ ] 去除对 Gradio 的依赖
- [ ] 打包前端静态产物，由 FastAPI 托管（或独立部署）
- [ ] 更新 README

---

## 关键技术决策

1. **长耗时任务**：SAM3 标注和 YOLO 训练可能耗时数分钟，采用后台线程 + `task_id` 轮询方案，避免 HTTP 超时。训练日志通过 **SSE（Server-Sent Events）** 实时推送。

2. **图片存储**：不在 API 响应中内嵌 base64（大图性能差），改为服务器端保存预览图后返回 URL，由前端 `<img src>` 加载。

3. **SAM3 单例**：保留 `get_segmentor()` 懒加载单例模式，后端进程生命周期内只加载一次模型。

4. **CORS**：开发阶段允许所有来源，生产部署时限制为前端域名。
