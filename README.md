# SAM3 Auto-Annotator + YOLO Trainer

基于 SAM3 (Segment Anything Model 3) 的自动标注工具，配合原生 Web 界面，实现从文本提示自动标注到 YOLO 模型训练的完整流程。

**作者：suika & Claude**

## 功能概览

| 页面 | 功能 |
|------|------|
| **标注测试** | 上传单张图片 + 输入提示词，实时预览 SAM3 分割效果，用于正式标注前调参 |
| **自动标注** | 输入图片目录和文本提示词，SAM3 批量标注并生成 YOLO 数据集 |
| **预览编辑** | 翻页浏览标注结果，支持删除整图或单个标注实例 |
| **模型训练** | 一键启动 YOLO 训练，实时显示日志和指标卡片 |
| **YOLO 推理** | 加载训练好的权重，对新图片进行批量推理 |

## 环境准备

建议先按照 [PyTorch 官网](https://pytorch.org/get-started/locally/) 的指引安装匹配 CUDA 版本的 PyTorch，再安装其余依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 1. 下载 SAM3 模型权重（约 3.5GB，首次需要）
python download.py

# 如果下载失败，可以手动下载 sam3.pt 后放到项目根目录。
# 有两个来源：
#
# 1. Meta 官方（需申请访问权限，审核通过后才能下载）：
#    https://huggingface.co/facebook/sam3/resolve/main/sam3.pt?download=true
#
# 2. 第三方镜像（无需审核，直接下载）：
#    https://huggingface.co/1038lab/sam3

# 2. 启动服务
python run.py

# 3. 浏览器打开 http://localhost:8000
```

> **注意**：YOLO 模型权重（如 `yolo11n.pt`）由 ultralytics 在首次训练或推理时自动下载，无需手动准备。

## 使用流程

### 1. 标注测试（建议先做）

在正式批量标注前，用单张图片快速验证提示词效果：

1. 上传一张代表性图片
2. 输入提示词，调整置信度和最大实例数
3. 点击「运行 SAM3」查看分割结果
4. 满意后将相同提示词和参数用于自动标注

> 右下角状态卡片实时显示 SAM3 模型状态（未下载 / 未加载 / 加载中 / 就绪 / 推理中）。未下载时点击可跳转下载页。

### 2. 自动标注

1. 设置图片目录和输出目录（默认 `dataset/`）
2. 输入文本提示词，逗号分隔（如 `tree,person,bush`）
3. 选择标注模式：
   - `detect` — 生成 YOLO 检测框标注（归一化 xywh）
   - `segment` — 生成 YOLO 分割多边形标注
4. 调整置信度阈值、验证集比例等参数
5. 点击「开始标注」，等待完成

### 3. 预览编辑

- 翻页浏览每张图的标注结果（支持左右滑动动画，自动预加载前后 5 张）
- **删除整张图**：连同 images/labels 文件一并删除
- **删除单个标注**：点击标注 tag 选中，点击「删除选中标注」

### 4. 模型训练

- 支持 YOLOv8 / YOLOv11 / YOLO26，规格 n/s/m/l/x
- 训练过程实时显示日志，并以指标卡片展示 Epoch / GPU / 各项 loss / Precision / Recall / mAP

### 5. YOLO 推理

- 填写权重文件路径（`.pt`）
- 上传图片查看推理结果，支持 SAM3 分割模式

## 项目结构

```
SAM3/
├── backend/             # FastAPI 后端
│   ├── main.py          # 应用入口，路由注册
│   ├── models.py        # Pydantic 数据模型
│   ├── tasks.py         # 异步任务注册表
│   ├── utils.py         # 坐标转换、颜色生成等工具函数
│   ├── segmentor.py     # SAM3 懒加载单例
│   └── routers/         # API 路由（images / preview / annotate / train / infer）
├── frontend/            # 原生 JS 前端（无框架）
│   ├── index.html
│   ├── app.js           # 路由、API 客户端、主题切换
│   ├── style.css
│   └── views/           # 各页面逻辑（annotate / preview / train / infer）
├── sam3.py              # SAM3Segmentor 封装 + draw_masks_on_image 可视化
├── run.py               # 启动入口（uvicorn）
├── download.py          # 下载 SAM3 模型权重
├── sam3.pt              # SAM3 模型权重（需下载）
└── requirements.txt
```

## 生成的数据集结构

```
dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## 技术细节

- **SAM3 懒加载单例**：模型仅在首次调用时加载，置信度变化时仅更新参数不重建
- **特征缓存**：同一张图片的多个 prompt 复用图像特征编码
- **异步任务**：标注和训练均在后台线程执行，通过任务注册表轮询进度
- **训练日志**：SSE 实时推流，正确处理 tqdm `\r` 覆写和 ANSI 转义码
- **所有图像使用 BGR 格式**（OpenCV 约定）

## License

MIT
