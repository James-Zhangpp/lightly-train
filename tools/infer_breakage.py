"""破损微小缺陷语义分割模型推理与可视化测试脚本（支持 ONNX 模型与 PyTorch Checkpoint）。

加速优化版功能特性：
1. 硬件加速诊断：自动检测 CUDA GPU 执行引擎，若处于 CPU 模式将主动弹出加速提示。
2. 毫秒级推理耗时测算：控制台与结果图片实时显示单张推理耗时（ms）与全图平均耗时。
3. 双模式引擎支持：优先加载 ONNX 模型；若未导出 ONNX，自动退回无缝加载 PyTorch Checkpoint (.pt) 并自动开启 AMP FP16 半精度加速。
4. 自动 Center Padding 机制：针对不同分辨率的原图（如 618x1502），自动中心无损补齐至 ONNX 要求的 624x1536 尺寸并裁切还原。
5. 8-连通域分析：精准定位微小破损瑕疵，绘制外接矩形框，标注瑕疵编号、宽高尺寸（WxH）及像素面积。
6. 可可选磁盘渲染控制：可选择 `--only-ng`（仅保存缺陷图）或 `--no-render`（纯推理测速），消除磁盘 IO 瓶颈。
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ===========================================================================
# 1. ⚙️【核心路径配置】（可在 PyCharm 中直接在此修改路径）
# ===========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 💡【测试图片或文件夹路径】（支持单张图片 Path(r"D:\test.png")，或图片文件夹路径）
DEFAULT_IMAGE = Path(r"C:\Users\ASUS\Desktop\PackAOI\NG")
# 示例 1 (测试桌面某个图片文件夹): DEFAULT_IMAGE = Path(r"C:\Users\ASUS\Desktop\LINE4\原图")
# 示例 2 (测试某张具体图片):     DEFAULT_IMAGE = Path(r"C:\Users\ASUS\Desktop\aoi_ng_001.png")

# 💡【测试结果保存目录】（绘制了 Mask 与瑕疵框的可视化图像保存位置）
DEFAULT_OUT_DIR = Path(r"C:\Users\ASUS\Desktop\PackAOI\test")
# 示例 (保存到桌面 test 文件夹): DEFAULT_OUT_DIR = Path(r"C:\Users\ASUS\Desktop\LINE4\test")

# 💡【实验模型目录】
DEFAULT_EXPERIMENT = PROJECT_ROOT / "out" / "Breakage_small_seg_1024x416_full_vitl16_eomt"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COMPONENT_COLORS = [
    (255, 0, 0),    # 纯红
    (255, 165, 0),  # 橙色
    (255, 215, 0),  # 金黄
    (0, 255, 255),  # 青色
    (255, 0, 255),  # 洋红
]


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def find_model_file(custom_path: Path | None = None) -> tuple[Path, str]:
    """智能查找可用的模型文件（优先查找 ONNX，若无则查找 PyTorch .pt）。"""
    if custom_path is not None:
        p = custom_path.resolve()
        if p.is_file():
            mode = "onnx" if p.suffix.lower() == ".onnx" else "pytorch"
            return p, mode
        raise FileNotFoundError(f"❌ 指定的模型文件不存在: {custom_path}")

    # 1. 检索 ONNX 默认产物路径
    onnx_candidates = [
        DEFAULT_EXPERIMENT / "exported_models" / "breakage_model_embedded.onnx",
        DEFAULT_EXPERIMENT / "exported_models" / "breakage_model.onnx",
        DEFAULT_EXPERIMENT / "exported_models" / "model_with_preprocessing.onnx",
    ]
    for cand in onnx_candidates:
        if cand.is_file():
            return cand, "onnx"

    # 2. 检索 PyTorch Checkpoint (.pt)
    pt_candidates = [
        DEFAULT_EXPERIMENT / "exported_models" / "exported_best.pt",
        DEFAULT_EXPERIMENT / "exported_models" / "exported_last.pt",
    ]
    for cand in pt_candidates:
        if cand.is_file():
            return cand, "pytorch"

    # 3. 广度检索 out 目录下任何 onnx / pt
    found_onnx = list(PROJECT_ROOT.glob("out/**/*.onnx"))
    if found_onnx:
        return found_onnx[0], "onnx"

    found_pt = list(PROJECT_ROOT.glob("out/**/*.pt"))
    if found_pt:
        return found_pt[0], "pytorch"

    raise FileNotFoundError(
        "❌ 未找到任何可用的 ONNX 模型或 PyTorch Checkpoint！\n"
        "请先运行训练 (python tools/train_breakage.py) 或通过 --onnx / --model 参数指定路径。"
    )


def find_image_paths(
    custom_images: list[str] | None = None,
    custom_dir: Path | None = None,
) -> list[Path]:
    """智能查找待测试的图片（优先读取参数，其次读取 DEFAULT_IMAGE 全局变量）。"""
    paths: list[Path] = []

    # 1. 命令行 --image 传入
    if custom_images:
        for item in custom_images:
            p = Path(item).expanduser()
            if any(ch in str(p) for ch in "*?[]"):
                paths.extend([Path(f) for f in glob.glob(str(p)) if Path(f).is_file()])
            elif p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(p.resolve())

    # 2. 命令行 --image-dir 传入
    if not paths and custom_dir:
        d = custom_dir.expanduser().resolve()
        if d.is_dir():
            paths.extend(sorted([f for f in d.glob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]))

    # 3. 默认从全局配置 DEFAULT_IMAGE 读取（支持单图或文件夹）
    if not paths and DEFAULT_IMAGE:
        d = DEFAULT_IMAGE.expanduser().resolve()
        if d.is_file() and d.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(d)
        elif d.is_dir():
            paths.extend(sorted([f for f in d.glob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]))

    # 4. 备用降级扫描 train 目录
    if not paths:
        train_dir = PROJECT_ROOT / "tools" / "Breakage_small_seg" / "img_dir" / "train"
        if train_dir.is_dir():
            paths.extend(sorted([f for f in train_dir.glob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]))

    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            unique.append(p.resolve())
    return unique


class ONNXPredictor:
    def __init__(self, onnx_path: Path):
        import onnxruntime as ort
        
        available = ort.get_available_providers()
        print(f"🔍 检测到 ONNXRuntime 可用引擎: {available}")
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("⚡ 成功启用 NVIDIA CUDA GPU 硬件加速引擎！")
        else:
            providers = ["CPUExecutionProvider"]
            print("⚠️ 警告: 当前使用的是 CPU 推理引擎（较慢）！")
            print("💡 提速建议: 运行 pip install onnxruntime-gpu 开启 GPU 硬件加速。")

        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.inp = self.session.get_inputs()[0]
        outputs = self.session.get_outputs()

        shape = self.inp.shape
        self.model_h = shape[2] if len(shape) == 4 and isinstance(shape[2], int) else 416
        self.model_w = shape[3] if len(shape) == 4 and isinstance(shape[3], int) else 1024

        if len(outputs) > 1:
            self.mode = "official"
        elif "float" in self.inp.type and len(shape) == 4 and shape[1] in (3, "3"):
            self.mode = "embedded"
        else:
            self.mode = "official"

        self.out_name = outputs[0].name

    def predict(self, image_pil: Image.Image, threshold: float = 0.0) -> np.ndarray:
        w, h = image_pil.size

        # 1. 维度补齐对齐（Center Padding 对齐到模型期望的尺寸）
        pad_h = self.model_h - h
        pad_w = self.model_w - w

        need_unpad = False
        need_resize = False

        if pad_h >= 0 and pad_w >= 0 and (pad_h > 0 or pad_w > 0):
            top_pad = pad_h // 2
            bottom_pad = pad_h - top_pad
            left_pad = pad_w // 2
            right_pad = pad_w - left_pad

            img_np = np.array(image_pil)
            img_padded = np.pad(
                img_np,
                ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            need_unpad = True
        elif pad_h < 0 or pad_w < 0:
            print(
                f"⚠️ 警告: 输入图像 ({w}x{h}) 大于 ONNX 模型的固定尺寸 ({self.model_w}x{self.model_h})！"
                f"双线性插值下采样可能会导致 1~3 像素的极微小破损被抹除。建议使用 PyTorch 模型全尺寸推理或重新导出对应尺寸的 ONNX。"
            )
            img_padded = np.array(image_pil.resize((self.model_w, self.model_h), Image.BILINEAR))
            need_resize = True
        else:
            img_padded = np.array(image_pil)

        # 2. 送入 ONNX 推理
        x = img_padded.astype(np.float32)
        x = np.transpose(x, (2, 0, 1))[np.newaxis, ...]  # NCHW

        if self.mode == "official":
            x = x / 255.0
            x = (x - IMAGENET_MEAN.reshape(3, 1, 1)) / IMAGENET_STD.reshape(3, 1, 1)

        result = self.session.run([self.out_name], {self.inp.name: x.astype(np.float32)})[0]
        
        if result.ndim == 4:
            mask = result[0, 0]
        elif result.ndim == 3:
            mask = result[0]
        else:
            mask = result

        # 判定二值掩码（使用指定的 threshold，默认 0.0）
        binary_mask = (mask > threshold).astype(np.uint8)

        # 3. 还原 Mask 尺寸到原图真实宽高 (h, w)
        if need_unpad:
            binary_mask = binary_mask[top_pad : top_pad + h, left_pad : left_pad + w]
        elif need_resize:
            binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        return binary_mask


class PyTorchPredictor:
    def __init__(self, pt_path: Path):
        import lightly_train
        print(f"📦 正在加载 PyTorch 模型 Checkpoint: {pt_path}")
        self.model = lightly_train.load_model(pt_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == "cuda":
            print(f"⚡ 已启用 CUDA GPU 设备 ({torch.cuda.get_device_name(0)})")
            print("⚡ 开启 PyTorch AMP FP16 自动半精度混合加速！")
        else:
            print("⚠️ 警告: 当前运行在 CPU 设备上（较慢）！")
            
        self.model.to(self.device).eval()

    def predict(self, image_pil: Image.Image, threshold: float = 0.0) -> np.ndarray:
        w, h = image_pil.size
        
        # 向上补齐到 16 的倍数 (ViT patch 16 对齐)，无需下采样缩小
        aligned_h = ((h + 15) // 16) * 16
        aligned_w = ((w + 15) // 16) * 16
        pad_h = aligned_h - h
        pad_w = aligned_w - w

        need_unpad = False
        if pad_h > 0 or pad_w > 0:
            top_pad = pad_h // 2
            bottom_pad = pad_h - top_pad
            left_pad = pad_w // 2
            right_pad = pad_w - left_pad

            img_np = np.array(image_pil)
            img_padded = np.pad(
                img_np,
                ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            need_unpad = True
        else:
            img_padded = np.array(image_pil)

        img_np = img_padded.astype(np.float32) / 255.0
        img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = self.model(tensor)
            else:
                outputs = self.model(tensor)

            if isinstance(outputs, (tuple, list)):
                mask_tensor = outputs[0]
            else:
                mask_tensor = outputs

            if mask_tensor.ndim == 4:
                mask_tensor = mask_tensor[0, 0]
            elif mask_tensor.ndim == 3:
                mask_tensor = mask_tensor[0]

            # 判定二值掩码（使用指定的 threshold，默认 0.0）
            binary_mask = (mask_tensor > threshold).cpu().numpy().astype(np.uint8)

        # 如果做了 Center Padding，推理完成后无损裁切还原为原图真实尺寸 (h, w)
        if need_unpad:
            binary_mask = binary_mask[top_pad : top_pad + h, left_pad : left_pad + w]

        return binary_mask


def compute_mask_connected_bboxes(
    mask: np.ndarray,
    *,
    foreground_class: int = 1,
    min_area: int = 1,
    border_margin: int = 0,
) -> list[dict]:
    """通过 8-连通域分析提取所有破损瑕疵，返回外接框与瑕疵属性列表。"""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    binary = (m == foreground_class).astype(np.uint8)
    if binary.sum() == 0:
        binary = (m > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []

    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    img_h, img_w = binary.shape[:2]
    results: list[dict] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        # 边缘死区过滤：如果瑕疵过于贴近图像最外侧边框，视为夹具/边框干扰噪点剔除
        if border_margin > 0:
            if x <= border_margin or y <= border_margin or (x + w) >= (img_w - border_margin) or (y + h) >= (img_h - border_margin):
                continue

        results.append({
            "bbox": (x, y, x + w, y + h),
            "width": w,
            "height": h,
            "area": area,
        })
    return results


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "arial.ttf",
        "Arial.ttf",
        "msyh.ttc",
        "simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def visualize_breakage_result(
    image_pil: Image.Image,
    mask: np.ndarray,
    components: list[dict],
    infer_ms: float = 0.0,
    alpha: float = 0.45,
) -> Image.Image:
    """在原图上叠加透明 Mask、瑕疵外接矩形框、推理耗时及 OK/NG 印章。"""
    w, h = image_pil.size
    orig_np = np.array(image_pil)

    # 1. 构造半透明 Mask 叠加图
    color_mask = np.zeros_like(orig_np)
    color_mask[mask == 1] = (255, 0, 0)  # 破损点红色半透明

    overlay_np = cv2.addWeighted(orig_np, 1.0 - alpha, color_mask, alpha, 0)
    overlay_pil = Image.fromarray(overlay_np)
    draw = ImageDraw.Draw(overlay_pil)

    font_size = max(14, min(w, h) // 28)
    font = _load_font(font_size)
    line_w = max(2, min(w, h) // 300)

    # 2. 绘制每一个连通域瑕疵的外接框与标注标签
    for idx, item in enumerate(components, start=1):
        x0, y0, x1, y1 = item["bbox"]
        bw, bh, area = item["width"], item["height"], item["area"]
        color = COMPONENT_COLORS[(idx - 1) % len(COMPONENT_COLORS)]

        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_w)

        tag = f"#{idx} {bw}x{bh} ({area}px)"
        tb = draw.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        tx, ty = x0, max(0, y0 - th - 4)

        draw.rectangle([tx, ty, tx + tw + 6, ty + th + 2], fill=(0, 0, 0))
        draw.text((tx + 3, ty + 1), tag, fill=(255, 235, 0), font=font)

    # 3. 左上角绘制推理耗时、图像分辨率及瑕疵总面积 Banner 标签
    total_area = sum(item["area"] for item in components)
    info_tag = f"Time: {infer_ms:.1f}ms | Size: {w}x{h} | Defect: {len(components)} ({total_area}px)"
    info_font_size = max(14, min(w, h) // 32)
    info_font = _load_font(info_font_size)
    info_tb = draw.textbbox((0, 0), info_tag, font=info_font)
    info_w, info_h = info_tb[2] - info_tb[0], info_tb[3] - info_tb[1]
    draw.rectangle([8, 8, 8 + info_w + 12, 8 + info_h + 8], fill=(0, 0, 0))
    draw.text((14, 12), info_tag, fill=(0, 255, 255), font=info_font)

    # 4. 右上角绘制全图结论 Stamp (OK / NG 印章)
    stamp_text = "NG" if len(components) > 0 else "OK"
    stamp_color = (255, 30, 30) if stamp_text == "NG" else (30, 220, 30)
    stamp_font_size = max(36, min(w, h) // 10)
    stamp_font = _load_font(stamp_font_size)

    tb = draw.textbbox((0, 0), stamp_text, font=stamp_font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    
    pad = 12
    margin_x = w - tw - pad * 2 - 10
    margin_y = 10
    draw.rectangle(
        [margin_x, margin_y, margin_x + tw + pad * 2, margin_y + th + pad * 2],
        fill=(0, 0, 0),
        outline=stamp_color,
        width=3,
    )
    draw.text((margin_x + pad, margin_y + pad // 2), stamp_text, fill=stamp_color, font=stamp_font)

    return overlay_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="破损微小缺陷模型推理与可视化测试脚本",
    )
    parser.add_argument(
        "--onnx", "--model",
        type=Path,
        default=None,
        help="模型路径（支持 ONNX .onnx 文件或 PyTorch .pt Checkpoint）",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="*",
        default=None,
        help=f"测试单张或多张图像路径 (默认: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help=f"测试图像目录路径 (默认: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"测试结果输出目录 (默认: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1,
        help="连通域过滤最小像素面积 (默认: 1，保留所有微小缺陷像素)",
    )
    parser.add_argument(
        "--border-margin",
        type=int,
        default=0,
        help="边缘死区屏蔽宽度 (像素，默认 0；若设为 15，则忽略贴近边缘 15px 内的夹具/边框噪点)",
    )
    parser.add_argument(
        "--only-ng",
        action="store_true",
        help="仅保存检出缺陷的 NG 图像结果（加速批处理）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="判定二值缺陷的 Logits 阈值 (默认: 0.0，数值越大越严格，越小召回率越高)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="不保存可视化结果图，仅在控制台测算纯模型推理耗时 FPS",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. 查找模型文件
    model_path, model_engine = find_model_file(args.onnx)

    # 2. 查找测试图像
    image_paths = find_image_paths(args.image, args.image_dir)

    if not image_paths:
        raise FileNotFoundError(
            f"❌ 未找到任何可用于测试的图片！请检查 DEFAULT_IMAGE 路径配置: {DEFAULT_IMAGE}"
        )

    out_dir = (args.out_dir or DEFAULT_OUT_DIR).resolve()
    if not args.no_render:
        out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("🔬 启动【破损微小缺陷】模型推理与可视化测试系统 (极速优化版)")
    print(f"📦 载入模型: {model_path} (引擎: {model_engine.upper()})")
    print(f"🖼️ 找到待测试图像: {len(image_paths)} 张")
    if not args.no_render:
        print(f"📂 保存目录: {out_dir}")
    print("=" * 68)

    # 初始化推理器
    if model_engine == "onnx":
        predictor = ONNXPredictor(model_path)
    else:
        predictor = PyTorchPredictor(model_path)

    ng_count = 0
    ok_count = 0
    total_infer_ms = 0.0

    # 💡 热身预热 Warmup (1 次)
    if len(image_paths) > 0:
        warmup_pil = load_rgb(image_paths[0])
        predictor.predict(warmup_pil)

    for idx, path in enumerate(image_paths, start=1):
        print(f"\n[{idx}/{len(image_paths)}] 正在处理: {path.name}")
        orig_pil = load_rgb(path)

        # 测算纯模型推理耗时 (ms)
        t0 = time.perf_counter()
        binary_mask = predictor.predict(orig_pil, threshold=args.threshold)
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000.0
        total_infer_ms += infer_ms

        # 连通域检测
        components = compute_mask_connected_bboxes(
            binary_mask, foreground_class=1, min_area=args.min_area, border_margin=args.border_margin
        )

        is_ng = len(components) > 0
        if is_ng:
            ng_count += 1
            print(f"  ❌ 检测结果: [NG 瑕疵品] | 发现 {len(components)} 处破损缺陷 | ⏱️ 推理耗时: {infer_ms:.2f} ms")
            for c_idx, comp in enumerate(components, start=1):
                bw, bh, area = comp["width"], comp["height"], comp["area"]
                print(f"     瑕疵 #{c_idx}: 框={comp['bbox']} | 尺寸={bw}x{bh} | 面积={area}px")
        else:
            ok_count += 1
            print(f"  ✅ 检测结果: [OK 良品] | 未发现破损缺陷 | ⏱️ 推理耗时: {infer_ms:.2f} ms")

        # 是否跳过渲染或仅保存 NG 图
        if args.no_render or (args.only_ng and not is_ng):
            continue

        # 生成可视化叠加图（带耗时标注）
        overlay_pil = visualize_breakage_result(
            orig_pil, binary_mask, components, infer_ms=infer_ms, alpha=0.45
        )

        out_file = out_dir / f"{path.stem}_overlay.png"
        overlay_pil.save(out_file)
        print(f"  💾 可视化图像已保存: {out_file}")

    avg_infer_ms = total_infer_ms / max(1, len(image_paths))
    print("\n" + "=" * 68)
    print("🎉 推理与可视化测试全部完成！")
    print(f"📊 汇总统计: 总测试 {len(image_paths)} 张 | OK良品: {ok_count} 张 | NG缺陷品: {ng_count} 张")
    print(f"⏱️ 性能指标: 平均纯模型推理耗时 {avg_infer_ms:.2f} ms/张")
    if not args.no_render:
        print(f"📂 结果文件所在目录: {out_dir}")
    print("=" * 68)


if __name__ == "__main__":
    main()
