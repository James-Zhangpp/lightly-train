"""L8 FPC 实例分割模型专用 ONNX 推理与可视化脚本。

特性：
1. 专为 tools/toOnnx_l8_fpc_inst.py 导出的 NHWC [1, 512, 512, 3] 内嵌预处理 ONNX 模型设计。
2. 兼容 Windows 中文路径读写，规避 OpenCV 默认中文路径读取失败的问题。
3. 输入仅需将图像 Resize 到 512x512，送入模型推理。
4. 推理结果 Mask 自动双线性插值高保真还原回原始图像分辨率。
5. 丰富多彩的半透明 Mask、高对比度轮廓线与标签置信度可视化。
6. 支持单张图像测试，也支持整个文件夹批量测试与统计。

用法：
  # 单图测试
  python tools/infer_l8_fpc_inst.py --image path/to/image.png

  # 批量测试（默认自动定位验证集目录）
  python tools/infer_l8_fpc_inst.py

  # 指定自定义模型与阈值
  python tools/infer_l8_fpc_inst.py --onnx out/l4_fpc_dinov3_vits16_eomt_inst/exported_models/model_l8_fpc_nhwc.onnx --threshold 0.6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 默认路径配置（与训练和导出脚本保持完全一致）
# ---------------------------------------------------------------------------
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists() or (p / ".git").exists()), Path(__file__).resolve().parents[3])
DEFAULT_EXPERIMENT = PROJECT_ROOT / "out" / "NP_BZ_vitb16_v2"
DEFAULT_ONNX = DEFAULT_EXPERIMENT / "exported_models" / "model_l8_fpc_nhwc.onnx"
DEFAULT_META = DEFAULT_EXPERIMENT / "exported_models" / "export_meta.json"

# 默认数据集验证集路径
DEFAULT_IMAGE_DIR = r"C:\Users\ASUS\Desktop\NP-LOCAL\分类文件夹\6-有镍片没焊接过且能找到左右两条镍片边缘-OK\crop"
DEFAULT_OUT_DIR = DEFAULT_EXPERIMENT / "infer_results"

# ===========================================================================
# ⭐【单独直接运行配置区】（无需命令行，直接在 IDE 右键/F5 运行修改此处即可）
# ===========================================================================
ONNX_MODEL_PATH: Path | None = None     # ONNX 模型路径，None 则默认使用 DEFAULT_ONNX
META_FILE_PATH: Path | None = None      # 元数据路径，None 则默认使用 DEFAULT_META
TEST_IMAGE_FILE: Path | None = None     # 单张测试图片路径（如 Path(r"D:/img.png")），为 None 时自动测试 TEST_IMAGE_FOLDER 目录
TEST_IMAGE_FOLDER: Path | None = None   # 待测试的图像目录路径，None 则默认使用验证集目录 DEFAULT_IMAGE_DIR
OUTPUT_DIRECTORY: Path | None = None    # 可视化结果保存目录，None 则默认使用 DEFAULT_OUT_DIR
CONF_THRESHOLD: float = 0.9             # 置信度过滤阈值 (0.0 ~ 1.0)
MIN_MASK_AREA: int = 10                 # 最小有效像素面积过滤
KEEP_LARGEST_COMPONENT: bool = True     # 是否仅保留最大连通域（过滤边缘碎屑/反光孤立噪点，让 BBox 紧致贴合铜箔主体）
MASK_THRESHOLD: float = 0.65            # 掩膜二值化概率阈值 (默认 0.65；值越大边界越紧致内收，有效杜绝外扩超界)
NMS_IOU_THRESHOLD: float = 0.50         # 同类别 Mask-IoU NMS 去重阈值 (0.0~1.0；<=0 时关闭，默认 0.50 消除 EoMT 重复 query)
# ===========================================================================

# 支持的图像文件后缀
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# 工业检测色彩调色板 (BGR 顺序)
COLOR_PALETTE = [
    (0, 255, 0),     # 鲜绿
    (255, 128, 0),   # 亮蓝
    (0, 165, 255),   # 暖橙
    (255, 0, 255),   # 洋红
    (0, 255, 255),   # 明黄
    (180, 105, 255), # 亮粉
    (255, 191, 0),   # 青蓝
    (0, 69, 255),    # 橙红
]


def imread_unicode(file_path: Path | str) -> np.ndarray:
    """安全读取包含中文字符路径的图像，返回 RGB 格式。"""
    path_str = str(file_path)
    data = np.fromfile(path_str, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法读取图片或图片已损坏: {file_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imwrite_unicode(file_path: Path | str, img_bgr: np.ndarray) -> bool:
    """安全保存包含中文字符路径的图像。"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    success, encoded = cv2.imencode(ext, img_bgr)
    if success:
        encoded.tofile(str(path))
    return success


def load_meta_classes(meta_path: Path) -> dict[int, str]:
    """读取 export_meta.json 中的类别字典。"""
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            classes = meta.get("classes", {})
            return {int(k): str(v) for k, v in classes.items()}
        except Exception as e:
            print(f"⚠️ 解析元数据 {meta_path} 失败: {e}，将使用默认类别")
    return {1: "FPC"}


def collect_images(
    image_arg: Path | None,
    image_dir_arg: Path | None,
) -> list[Path]:
    """收集待测试的图片列表。"""
    if image_arg and image_arg.is_file():
        return [image_arg.resolve()]

    search_dir = image_dir_arg or (image_arg if image_arg and image_arg.is_dir() else None)
    if search_dir is None:
        search_dir = DEFAULT_IMAGE_DIR

    search_dir = Path(search_dir).resolve()
    if not search_dir.is_dir():
        return []

    images = [
        p
        for p in sorted(search_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images


def get_best_ort_providers() -> list[str]:
    """优先尝试 GPU 加速，不可用时回退 CPU。"""
    import onnxruntime as ort

    available = ort.get_available_providers()
    selected = []
    if "CUDAExecutionProvider" in available:
        selected.append("CUDAExecutionProvider")
    selected.append("CPUExecutionProvider")
    return selected


class L8FPCPredictor:
    """L8 FPC 实例分割 ONNX 推理器。"""

    def __init__(
        self,
        onnx_path: Path,
        meta_path: Path | None = None,
        input_size: tuple[int, int] = (512, 512),
    ) -> None:
        import onnxruntime as ort

        self.onnx_path = onnx_path.resolve()
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"❌ ONNX 模型文件不存在: {self.onnx_path}")

        meta_file = meta_path or (self.onnx_path.parent / "export_meta.json")
        self.classes = load_meta_classes(meta_file)
        self.input_h, self.input_w = input_size

        providers = get_best_ort_providers()
        print(f"[INIT] 初始化 ONNXRuntime 推理引擎...")
        print(f"   模型路径: {self.onnx_path}")
        print(f"   使用 Provider: {providers}")
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"   输入节点: '{self.input_name}', 目标尺寸: {self.input_w}x{self.input_h}")
        print(f"   类别映射: {self.classes}")

    @staticmethod
    def _apply_mask_nms(
        instances: list[dict[str, Any]],
        iou_threshold: float = 0.50,
    ) -> list[dict[str, Any]]:
        """按置信度降序执行同类别 Mask-IoU NMS，消除 EoMT 多 query 重复检出同一目标。"""
        if iou_threshold <= 0.0 or len(instances) <= 1:
            return instances

        kept: list[dict[str, Any]] = []
        for cand in instances:
            cand_label = cand["label"]
            cand_mask = cand["mask"]
            cand_area = int(cand["area"])
            cx, cy, cw, ch = cand["bbox"]

            is_duplicate = False
            for ref in kept:
                if ref["label"] != cand_label:
                    continue

                # 先用 BBox 快速排斥无交集目标，避免无谓的整图掩膜像素计算
                rx, ry, rw, rh = ref["bbox"]
                if cx + cw <= rx or rx + rw <= cx or cy + ch <= ry or ry + rh <= cy:
                    continue

                # 仅在两者的最小包围盒并集 ROI 内计算精确 Mask IoU，提升工业大图推理速度
                x1 = max(cx, rx)
                y1 = max(cy, ry)
                x2 = min(cx + cw, rx + rw)
                y2 = min(cy + ch, ry + rh)
                if x2 <= x1 or y2 <= y1:
                    continue

                inter = int(
                    np.count_nonzero(
                        (cand_mask[y1:y2, x1:x2] > 0) & (ref["mask"][y1:y2, x1:x2] > 0)
                    )
                )
                if inter <= 0:
                    continue

                union = cand_area + int(ref["area"]) - inter
                iou = inter / float(union) if union > 0 else 0.0
                if iou >= iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(cand)

        return kept

    def predict(
        self,
        image_rgb: np.ndarray,
        threshold: float = 0.50,
        min_area: int = 10,
        keep_largest_component: bool = True,
        mask_threshold: float = 0.65,
        nms_iou_threshold: float = 0.50,
    ) -> list[dict[str, Any]]:
        """执行推理并完成掩膜向原图尺寸的高保真映射还原。

        返回实例列表：
          [
            {
              "label": int,
              "class_name": str,
              "score": float,
              "mask": np.ndarray (原图尺寸二值掩膜, uint8: 0/1),
              "bbox": (x, y, w, h),
              "area": int
            },
            ...
          ]
        """
        orig_h, orig_w = image_rgb.shape[:2]

        # 1. 预处理：Resize 到 512x512 并构造 [1, 512, 512, 3] float32 (数值 0~255)
        resized = cv2.resize(image_rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        input_tensor = resized.astype(np.float32)[np.newaxis, ...]

        # 2. ONNX 前向推理
        labels_batch, masks_batch, scores_batch = self.session.run(
            None, {self.input_name: input_tensor}
        )

        labels = labels_batch[0]    # [Q]
        masks = masks_batch[0]      # [Q, 512, 512]
        scores = scores_batch[0]    # [Q]

        # 3. 置信度阈值过滤
        keep_indices = np.where(scores >= threshold)[0]

        instances: list[dict[str, Any]] = []
        for idx in keep_indices:
            score = float(scores[idx])
            label_id = int(labels[idx])
            class_name = self.classes.get(label_id, f"Class_{label_id}")

            mask_512 = masks[idx]  # uint8: 0 或 1

            # 4. 将 512x512 掩膜双线性插值映射回原图尺寸
            mask_resized = cv2.resize(
                mask_512.astype(np.float32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )
            # 使用 mask_threshold 二值化，默认 0.65，使边缘紧致收敛消除外扩
            mask_orig = (mask_resized > mask_threshold).astype(np.uint8)

            if keep_largest_component:
                # 5. 工业级连通域清洗：仅提取面积最大的主体连通块，剔除边缘微弱反光/碎屑噪点
                num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(
                    mask_orig, connectivity=8
                )
                if num_labels <= 1:
                    continue
                # 背景为 label 0，前景为 1 ~ num_labels-1
                areas = stats[1:, cv2.CC_STAT_AREA]
                max_label = int(np.argmax(areas)) + 1
                max_area = int(stats[max_label, cv2.CC_STAT_AREA])

                if max_area < min_area:
                    continue

                mask_final = (labels_im == max_label).astype(np.uint8)
                area = max_area
            else:
                mask_final = mask_orig
                area = int(np.sum(mask_final))
                if area < min_area:
                    continue

            # 计算边界框与轮廓
            contours, _ = cv2.findContours(
                mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            # 使用清洗后的轮廓提取紧凑边界框
            x, y, w, h = cv2.boundingRect(np.vstack(contours))

            instances.append(
                {
                    "label": label_id,
                    "class_name": class_name,
                    "score": score,
                    "mask": mask_final,
                    "bbox": (x, y, w, h),
                    "area": area,
                    "contours": contours,
                }
            )

        # 按置信度降序排列，并执行 Mask-IoU NMS 过滤重复检测
        instances.sort(key=lambda item: item["score"], reverse=True)
        instances = self._apply_mask_nms(instances, iou_threshold=nms_iou_threshold)
        return instances


def draw_instances_visualization(
    image_rgb: np.ndarray,
    instances: list[dict[str, Any]],
    alpha: float = 0.45,
) -> np.ndarray:
    """在原图上绘制高对比度多彩掩膜、轮廓线与标签文字。"""
    # 转为 BGR 供 OpenCV 绘图
    vis_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = vis_bgr.copy()

    for idx, inst in enumerate(instances):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        mask = inst["mask"]  # (H, W) uint8

        # 1. 填充半透明颜色
        overlay[mask == 1] = color

        # 2. 绘制鲜明边界轮廓
        contours = inst.get("contours", [])
        if contours:
            cv2.drawContours(vis_bgr, contours, -1, color, 2)

    # 融合半透明图层
    cv2.addWeighted(overlay, alpha, vis_bgr, 1 - alpha, 0, vis_bgr)

    # 3. 绘制文字标签与包围框
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale1 = 0.50
    font_scale2 = 0.46
    thickness = 1

    for idx, inst in enumerate(instances):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        x, y, w, h = inst["bbox"]
        area = inst.get("area", 0)

        line1 = f"{inst['class_name']} {inst['score']:.2f}"
        line2 = f"{w}x{h} Area:{area}"

        # 绘制边界框
        cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

        # 计算两行文字尺寸
        (tw1, th1), b1 = cv2.getTextSize(line1, font, font_scale1, thickness)
        (tw2, th2), b2 = cv2.getTextSize(line2, font, font_scale2, thickness)
        max_tw = max(tw1, tw2)
        total_th = th1 + th2 + 10  # 两行文字总高度加内边距

        # 边界自适应：若距顶部太近无法容纳两行，则翻转至框内上侧显示
        if y - total_th - 4 < 0:
            bg_y1 = y + 2
            bg_y2 = y + 2 + total_th
            text1_y = bg_y1 + th1 + 3
            text2_y = text1_y + th2 + 4
        else:
            bg_y1 = y - total_th - 4
            bg_y2 = y - 2
            text1_y = bg_y1 + th1 + 3
            text2_y = text1_y + th2 + 4

        bg_x1 = max(0, x)
        bg_x2 = min(vis_bgr.shape[1], x + max_tw + 8)

        # 绘制背景底框
        cv2.rectangle(
            vis_bgr,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            color,
            -1,
        )
        # 第一行文字：类别与置信度
        cv2.putText(
            vis_bgr,
            line1,
            (bg_x1 + 4, text1_y),
            font,
            font_scale1,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        # 第二行文字：尺寸与面积
        cv2.putText(
            vis_bgr,
            line2,
            (bg_x1 + 4, text2_y),
            font,
            font_scale2,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    # 在左上角标注检出实例统计
    summary_text = f"Detections: {len(instances)}"
    cv2.putText(
        vis_bgr,
        summary_text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if instances else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return vis_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L8 FPC 实例分割 ONNX 模型推理与可视化")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=DEFAULT_ONNX,
        help=f"ONNX 模型路径 (默认: {DEFAULT_ONNX})",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=DEFAULT_META,
        help=f"元数据 export_meta.json 路径 (默认: {DEFAULT_META})",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="待推理的单张图像路径",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="待推理的图像目录路径 (默认尝试验证集目录)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"可视化结果保存目录 (默认: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="置信度过滤阈值 (默认 0.5)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=10,
        help="最小掩膜有效像素面积过滤 (默认 10)",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.65,
        help="掩膜二值化概率阈值 (默认 0.65；值越大边界越紧致内收，有效解决外扩)",
    )
    parser.add_argument(
        "--all-components",
        action="store_true",
        help="保留全部连通域（默认仅提取最大主体连通块，过滤边缘微弱反光/碎屑噪点）",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.50,
        help="同类别 Mask-IoU NMS 去重阈值 (默认 0.50；设为 0 关闭去重)",
    )
    return parser.parse_args()


def main() -> None:
    # 判断是否直接在 IDE 中运行（未传入命令行参数）
    if len(sys.argv) <= 1:
        print("[提示] 当前未检测到命令行传参，已自动采用脚本顶部的【单独直接运行配置区】参数。")
        raw_onnx = ONNX_MODEL_PATH or DEFAULT_ONNX
        onnx_path = raw_onnx.resolve()
        raw_meta = META_FILE_PATH or DEFAULT_META
        meta_path = raw_meta.resolve()
        image_arg = TEST_IMAGE_FILE.resolve() if TEST_IMAGE_FILE else None
        image_dir_arg = TEST_IMAGE_FOLDER.resolve() if TEST_IMAGE_FOLDER else None
        raw_out = OUTPUT_DIRECTORY or DEFAULT_OUT_DIR
        out_dir = raw_out.resolve()
        threshold = CONF_THRESHOLD
        mask_threshold = MASK_THRESHOLD
        min_area = MIN_MASK_AREA
        keep_largest = KEEP_LARGEST_COMPONENT
        nms_iou = NMS_IOU_THRESHOLD
    else:
        args = parse_args()
        onnx_path = args.onnx.resolve()
        meta_path = args.meta.resolve()
        image_arg = args.image.resolve() if args.image else None
        image_dir_arg = args.image_dir.resolve() if args.image_dir else None
        out_dir = args.out_dir.resolve()
        threshold = args.threshold
        mask_threshold = args.mask_threshold
        min_area = args.min_area
        keep_largest = not args.all_components
        nms_iou = args.nms_iou

    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"[错误] 未找到 ONNX 模型: {onnx_path}\n"
            f"请先运行导出脚本: python tools/toOnnx_l8_fpc_inst.py"
        )

    predictor = L8FPCPredictor(onnx_path=onnx_path, meta_path=meta_path)

    # 收集图片
    image_paths = collect_images(image_arg, image_dir_arg)
    if not image_paths:
        print(f"[警告] 未找到待测试的图像！")
        print(f"请在脚本顶部修改 TEST_IMAGE_FILE / TEST_IMAGE_FOLDER 或通过命令行参数指定。")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==================== 开始批量推理 ====================")
    print(f"找到 {len(image_paths)} 张测试图片，输出保存至: {out_dir}\n")

    total_time = 0.0
    for idx, img_path in enumerate(image_paths, 1):
        try:
            t0 = time.perf_counter()
            img_rgb = imread_unicode(img_path)
            read_cost = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            instances = predictor.predict(
                image_rgb=img_rgb,
                threshold=threshold,
                min_area=min_area,
                keep_largest_component=keep_largest,
                mask_threshold=mask_threshold,
                nms_iou_threshold=nms_iou,
            )
            infer_cost = (time.perf_counter() - t1) * 1000

            # 可视化与保存
            vis_bgr = draw_instances_visualization(img_rgb, instances)
            save_path = out_dir / f"{img_path.stem}_pred.png"
            imwrite_unicode(save_path, vis_bgr)

            cost_total = (time.perf_counter() - t0) * 1000
            total_time += infer_cost

            inst_details = ""
            if instances:
                details = [
                    f"#{i+1}({inst['class_name']} {inst['score']:.2f} {inst['bbox'][2]}x{inst['bbox'][3]} Area:{inst.get('area',0)})"
                    for i, inst in enumerate(instances)
                ]
                inst_details = " " + " ".join(details)

            print(
                f"[{idx}/{len(image_paths)}] {img_path.name} | "
                f"检出: {len(instances)}{inst_details} | "
                f"耗时: {infer_cost:.1f}ms -> {save_path.name}"
            )

        except Exception as e:
            print(f"[错误] 处理图像 {img_path.name} 失败: {e}")

    avg_infer = total_time / max(len(image_paths), 1)
    print(f"\n[完成] 全部推理完成！平均推理耗时: {avg_infer:.2f}ms/张")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
