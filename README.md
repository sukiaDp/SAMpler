# SAM3 Auto-Annotator + YOLO Trainer

基于 SAM3 (Segment Anything Model 3) 的自动标注工具，配合 Gradio Web 界面，实现从文本提示自动标注到 YOLO 模型训练的完整流程。

**作者：suika & Claude**

## 功能概览

| 标签页 | 功能 |
|--------|------|
| **自动标注与训练** | SAM3 文本提示批量标注 → 预览/编辑标注 → 一键训练 YOLO |
| **单图测试** | 拖入图片，用 SAM3 实时查看分割效果 |
| **YOLO 推理** | 加载训练好的权重，对新图片进行推理 |

## 环境准备

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 1. 下载 SAM3 模型权重（约 3.5GB，首次需要）
python download.py

# 2. 启动应用
python app.py

# 3. 浏览器打开 http://localhost:7860
```

## 使用流程

### 1. 自动标注

1. 设置图片目录（默认 `rawData`）和输出目录（默认 `dataset/`）
2. 输入文本提示词，逗号分隔（如 `tree,person,bush`）
3. 选择标注模式：
   - `detect` — 生成 YOLO 检测框标注（归一化 xywh）
   - `segment` — 生成 YOLO 分割多边形标注
4. 调整参数：置信度阈值、验证集比例、每张图最大标注数
5. 点击「开始标注」，等待进度条完成

### 2. 标注预览与编辑

- 翻页浏览每张图的标注结果
- **删除整张图**：连同 images、labels 文件一并删除
- **删除单个标注**：右侧列表显示每个标注实例（含颜色色块和类型），勾选后点击「删除选中标注」，自动重绘预览并更新标签文件

### 3. 模型训练

- 支持 YOLOv8 / YOLOv11 / YOLO26，规格 n/s/m/l/x
- 任务类型自动跟随标注模式（detect / segment）
- 实时显示训练日志

### 4. YOLO 推理

- 选择训练好的权重文件（`.pt`）
- 拖入图片即可查看推理结果

## 项目结构

```
SAM3/
├── sam3.py          # SAM3Segmentor 封装 + draw_masks_on_image 可视化
├── app.py           # Gradio 应用主文件
├── download.py      # 下载 SAM3 模型权重
├── sam3.pt          # SAM3 模型权重（需下载）
├── CLAUDE.md        # Claude Code 项目指引
└── rawData/         # 待标注图片目录
```

## 生成的数据集结构

```
dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── previews/        # 标注预览图缓存
```

## 技术细节

- **SAM3 懒加载单例**：模型仅在首次调用时加载，后续复用，置信度变化时仅更新参数不重建模型
- **特征缓存**：同一张图片的多个 prompt 复用图像特征编码
- **标注数限制**：支持按置信度或框面积排序，截断超出的标注实例
- **标注编辑**：删除标注实例后从 label_line 数据重绘预览图，无需重新推理
- **所有图像使用 BGR 格式**（OpenCV 约定）

## License

MIT
