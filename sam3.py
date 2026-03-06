import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Tuple
from ultralytics.models.sam import SAM3SemanticPredictor


class SAM3Segmentor:
    """SAM3文本提示分割器"""

    def __init__(
            self,
            model_path: str = "sam3.pt",
            conf: float = 0.25,
            device: str = "0",
            half: bool = True,
            bpe_path: Optional[str] = None
    ):
        """
        初始化SAM3分割器

        Args:
            model_path: 模型权重路径
            conf: 置信度阈值
            device: 设备 ("0", "1", "cpu")
            half: 是否使用FP16
            bpe_path: BPE tokenizer路径，None时自动处理
        """
        overrides = {
            "conf": conf,
            "task": "segment",
            "mode": "predict",
            "model": model_path,
            "half": half,
            "device": device,
            "save": False,
            "verbose": False
        }

        kwargs = {"overrides": overrides}
        if bpe_path:
            kwargs["bpe_path"] = bpe_path

        self.predictor = SAM3SemanticPredictor(**kwargs)
        self._current_image = None

    def predict(
            self,
            image: Union[str, np.ndarray],
            text_prompts: Union[str, List[str]],
            force_reload: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        执行文本提示分割

        Args:
            image: 图片路径或numpy数组 (BGR格式)
            text_prompts: 单个或多个文本提示
            force_reload: 是否强制重新加载图像特征

        Returns:
            (masks, boxes, confs): masks shape=(N, H, W), boxes shape=(N, 4) xyxy格式,
                          confs shape=(N,) 置信度
                          如果没有检测到目标则返回 (None, None, None)
        """
        # 处理输入
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        # 加载图像
        if isinstance(image, str):
            image_path = image
            if not Path(image_path).exists():
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
        else:
            # numpy数组，需要先保存为临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, image)
                image_path = tmp.name

        # 只在图像改变或强制时重新加载
        if self._current_image != image_path or force_reload:
            self.predictor.set_image(image_path)
            self._current_image = image_path

        # 执行推理
        results = self.predictor(text=text_prompts)

        # 提取masks、boxes和confs
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()  # (N, H, W)
                boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                confs = result.boxes.conf.cpu().numpy()  # (N,)
                return masks, boxes, confs

        return None, None, None

    def predict_with_exemplar(
            self,
            image: Union[str, np.ndarray],
            bboxes: List[List[float]],
            text_prompts: Optional[List[str]] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        使用bbox示例进行分割

        Args:
            image: 图片路径或numpy数组
            bboxes: bbox列表 [[x1,y1,x2,y2], ...]
            text_prompts: 可选的文本提示

        Returns:
            (masks, boxes, confs)
        """
        if isinstance(image, str):
            image_path = image
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, image)
                image_path = tmp.name

        if self._current_image != image_path:
            self.predictor.set_image(image_path)
            self._current_image = image_path

        kwargs = {"bboxes": bboxes}
        if text_prompts:
            kwargs["text"] = text_prompts

        results = self.predictor(**kwargs)

        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                return masks, boxes, confs

        return None, None, None


def draw_masks_on_image(
        image: np.ndarray,
        masks: np.ndarray,
        boxes: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        alpha: float = 0.5,
        colors: Optional[List[Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    在图像上绘制分割mask

    Args:
        image: 原始图像 (H, W, 3) BGR格式
        masks: 分割mask (N, H, W) bool或float格式
        boxes: 可选的边界框 (N, 4) xyxy格式
        labels: 可选的标签列表
        alpha: mask透明度 0-1
        colors: 自定义颜色列表 [(B,G,R), ...], None时自动生成

    Returns:
        绘制后的图像
    """
    output = image.copy()
    h, w = image.shape[:2]

    # 确保masks是bool类型
    if masks.dtype != bool:
        masks = masks > 0.5

    num_masks = masks.shape[0]

    # 生成颜色
    if colors is None:
        np.random.seed(42)
        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_masks)]

    # 绘制每个mask
    for i in range(num_masks):
        mask = masks[i]
        color = colors[i % len(colors)]

        # 调整mask尺寸（如果需要）
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        # 创建彩色mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = color

        # 叠加到输出图像
        output = cv2.addWeighted(output, 1, colored_mask, alpha, 0)

        # 绘制边界
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, color, 2)

        # 绘制bbox和标签
        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            if labels is not None and i < len(labels):
                label_text = labels[i]
                # 计算文本尺寸
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                # 绘制背景
                cv2.rectangle(output, (x1, y1 - text_h - baseline - 5),
                              (x1 + text_w, y1), color, -1)
                # 绘制文本
                cv2.putText(output, label_text, (x1, y1 - baseline - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return output


# 使用示例
if __name__ == "__main__":
    # 初始化分割器
    segmentor = SAM3Segmentor(
        model_path="sam3.pt",
        conf=0.3,
        device="0",
        half=True
    )

    # 示例1: 单个文本提示
    image_path = "3/2026_02_27_11_00_26_IMG_6999.JPG"
    masks, boxes, confs = segmentor.predict(image_path, "tree")

    if masks is not None:
        print(f"检测到 {len(masks)} 个目标")

        # 读取原图
        image = cv2.imread(image_path)

        # 绘制结果
        result_img = draw_masks_on_image(
            image,
            masks,
            boxes,
            labels=["person"] * len(masks),
            alpha=0.4
        )

        # 保存结果
        cv2.imwrite("result_single.jpg", result_img)
        print("结果已保存到 result_single.jpg")
    else:
        print("未检测到目标")

    # 示例2: 多个文本提示
    masks, boxes, confs = segmentor.predict(
        image_path,
        ["bush", "tree", "face"]
    )

    if masks is not None:
        image = cv2.imread(image_path)
        result_img = draw_masks_on_image(image, masks, boxes, alpha=0.5)
        cv2.imwrite("result_multiple.jpg", result_img)
        print(f"多类别检测: {len(masks)} 个目标")

    # 示例3: 特征重用（高效）
    segmentor.predict(image_path, "person")  # 第一次加载图像
    masks1, _, _ = segmentor.predict(image_path, "car", force_reload=False)  # 重用特征
    masks2, _, _ = segmentor.predict(image_path, "bicycle", force_reload=False)  # 重用特征

    # 示例4: 使用numpy数组输入
    image_array = cv2.imread(image_path)
    masks, boxes, confs = segmentor.predict(image_array, "yellow school bus")

    if masks is not None:
        result_img = draw_masks_on_image(image_array, masks, boxes)
        cv2.imwrite("result_array.jpg", result_img)

    # 示例5: 使用bbox exemplar
    # 假设已知第一个人的框
    example_box = [[100, 100, 300, 400]]
    masks, boxes, confs = segmentor.predict_with_exemplar(
        image_path,
        bboxes=example_box,
        text_prompts=["person"]  # 可选
    )

    if masks is not None:
        image = cv2.imread(image_path)
        result_img = draw_masks_on_image(image, masks, boxes)
        cv2.imwrite("result_exemplar.jpg", result_img)