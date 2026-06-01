"""ONNX 语义分割推理与 mask 可视化（与 tools/toOnnx.py 配套）。

预处理模式（由 ONNX 图自动检测 detect_onnx_mode）：
  float_nchw_raw  embedded 单输出：input 为 [0,255] float NCHW，图内已归一化
  float_nchw      official 多输出：外部 /255 + ImageNet mean/std
  uint8_hwc       少数 uint8 HWC 输入（本仓库导出一般不用）

默认流程：按 CROP_ROI 裁剪 -> resize 到 --size（512）推理 -> 将 mask/bbox 映射回原图。

默认输出：每张图仅 {stem}_overlay.png
  - mask 叠加；在原图上按**连通域**分别画外接矩形并标注各区域宽高
  - 无前景连通域时标注红色 NG

--save-all：额外保存 crop、mask、512 输入等中间图（调试用）。
--min-area：过滤面积过小的连通域（像素）。
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# 路径（与 tools/train.py / tools/toOnnx.py 实验目录一致）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "out" / "welding_dinov3_vits16_eomt_v2"
# embedded 默认产物名（toOnnx.py 在未指定 --out 时生成）
DEFAULT_ONNX = DEFAULT_EXPERIMENT / "exported_models" / "model_with_preprocessing.onnx"
# official 模式请改用：.../lightly_float32.onnx 或自行 --onnx 指定

# 本机默认输入/输出目录（可按环境修改）
DEFAULT_IMAGE = Path(r"C:\Users\ASUS\Desktop\DL 5\原图")
DEFAULT_OUT_DIR = Path(r"C:\Users\ASUS\Desktop\DL 5\onnxtest")

# 现场 ROI：(x, y, width, height)，再 resize 到 --size；默认与训练视野一致时可改
CROP_ROI: tuple[int, int, int, int] = (900, 700, 768, 768)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# 与 train.py / toOnnx.py 一致（仅 float_nchw 模式在外部做归一化）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 类别 id -> 可视化 RGB（0=background 黑，1=fpc 绿）
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 255, 0),
}


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def list_images_in_dir(directory: Path, *, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [
        p
        for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files)


def collect_image_paths(
    images: list[Path | str] | None,
    image_dir: Path | str | None,
    *,
    recursive: bool = False,
) -> list[Path]:
    """解析单文件、多文件、目录或 glob 模式。"""
    paths: list[Path] = []

    for raw in images or []:
        source = Path(raw).expanduser()
        source_str = str(source)

        if any(ch in source_str for ch in "*?[]"):
            matched = [Path(p) for p in glob.glob(source_str, recursive=True)]
            paths.extend(p for p in matched if p.is_file())
            continue

        source = source.resolve()
        if source.is_file():
            if source.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(source)
            else:
                print(f"  skip (unsupported extension): {source}")
        elif source.is_dir():
            paths.extend(list_images_in_dir(source, recursive=recursive))
        else:
            raise FileNotFoundError(f"Image path not found: {source}")

    if image_dir is not None:
        directory = Path(image_dir).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Image directory not found: {directory}")
        paths.extend(list_images_in_dir(directory, recursive=recursive))

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def crop_roi(image: Image.Image, roi: tuple[int, int, int, int]) -> Image.Image:
    """按 (x, y, width, height) 裁剪；越界时裁剪到图像范围内并打印 warning。"""
    x, y, w, h = roi
    img_w, img_h = image.size
    if w <= 0 or h <= 0:
        raise ValueError(f"ROI width/height must be positive, got ({w}, {h})")

    x0 = max(0, min(x, img_w))
    y0 = max(0, min(y, img_h))
    x1 = min(x0 + w, img_w)
    y1 = min(y0 + h, img_h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI {roi} is outside image bounds ({img_w}x{img_h})")
    if (x0, y0, x1 - x0, y1 - y0) != (x, y, w, h):
        print(
            f"  warning: ROI clipped from ({x},{y},{w},{h}) "
            f"to ({x0},{y0},{x1 - x0},{y1 - y0}) for image {img_w}x{img_h}"
        )
    return image.crop((x0, y0, x1, y1))


def resize_square(image: Image.Image, size: int) -> Image.Image:
    """双线性缩放到 size×size；默认 size=512 与训练 image_size 一致。"""
    return image.resize((size, size), Image.BILINEAR)


def detect_onnx_mode(session) -> str:
    """根据 ONNX 输入/输出结构选择预处理分支。

    多输出 -> float_nchw（official，外部归一化）
    单输出 float NCHW -> float_nchw_raw（embedded，像素 [0,255]）
    uint8 HWC -> uint8_hwc
    """
    inp = session.get_inputs()[0]
    outputs = session.get_outputs()
    shape = inp.shape
    if len(outputs) > 1:
        return "float_nchw"
    if len(shape) == 4 and shape[-1] in (3, "3"):
        return "uint8_hwc"
    if "float" in inp.type and len(shape) == 4 and shape[1] in (3, "3"):
        return "float_nchw_raw"
    return "float_nchw"


def prepare_input(image_rgb: np.ndarray, mode: str) -> np.ndarray:
    """HxWx3 uint8 -> batch；float_nchw_raw 不做 /255 与 mean/std。"""
    if mode == "uint8_hwc":
        return image_rgb[np.newaxis, ...].astype(np.uint8)

    x = image_rgb.astype(np.float32)
    x = np.transpose(x, (2, 0, 1))
    x = x[np.newaxis, ...]

    if mode == "float_nchw_raw":
        return x

    x = x / 255.0
    x = (x - IMAGENET_MEAN.reshape(3, 1, 1)) / IMAGENET_STD.reshape(3, 1, 1)
    return x.astype(np.float32)


def run_onnx(session, batch: np.ndarray) -> np.ndarray:
    """返回 2D mask [H,W]；多输出时优先名称含 mask 的输出。"""
    inp = session.get_inputs()[0]
    outputs = session.get_outputs()

    if len(outputs) == 1:
        out_names = [outputs[0].name]
    else:
        mask_out = next((o for o in outputs if "mask" in o.name.lower()), outputs[0])
        out_names = [mask_out.name]

    result = session.run(out_names, {inp.name: batch})[0]
    if result.ndim == 4 and result.shape[1] == 1:
        return result[0, 0]
    if result.ndim == 3:
        return result[0]
    return result


def mask_to_color(mask: np.ndarray, colors: dict[int, tuple[int, int, int]]) -> np.ndarray:
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, rgb in colors.items():
        color[mask == class_id] = rgb
    return color


def _load_annotation_font(size: int) -> ImageFont.ImageFont:
    """尽量加载 TrueType 字体；失败则用默认位图字体。"""
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


def _to_rgb_pil(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image.astype(np.uint8), mode="L").convert("RGB")
        return Image.fromarray(image.astype(np.uint8))
    pil = image.copy()
    if pil.mode not in ("RGB", "RGBA"):
        return pil.convert("RGB")
    if pil.mode == "RGBA":
        return pil.convert("RGB")
    return pil


def compute_mask_bbox(
    mask: np.ndarray,
    *,
    foreground_class: int = 1,
) -> tuple[int, int, int, int] | None:
    """全部前景像素合并后的外接矩形（单框）。"""
    bboxes = compute_mask_connected_bboxes(
        mask, foreground_class=foreground_class, min_area=1
    )
    if not bboxes:
        return None
    return union_bbox(bboxes)


def compute_mask_connected_bboxes(
    mask: np.ndarray,
    *,
    foreground_class: int = 1,
    min_area: int = 16,
) -> list[tuple[int, int, int, int]]:
    """8 连通域分析，每个前景连通块一个外接矩形 (x0, y0, x1, y1)。"""
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
    bboxes: list[tuple[int, int, int, int]] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        bboxes.append((x, y, x + w, y + h))
    return bboxes


def union_bbox(
    bboxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    if not bboxes:
        return None
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return x0, y0, x1, y1


def largest_bbox_index(bboxes: list[tuple[int, int, int, int]]) -> int:
    areas = [bbox_wh(b)[0] * bbox_wh(b)[1] for b in bboxes]
    return int(np.argmax(areas))


def bbox_wh(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, y1 = bbox
    return x1 - x0, y1 - y0


def scale_bbox(
    bbox: tuple[int, int, int, int],
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """将 bbox 从 src_size 坐标系缩放到 dst_size。"""
    x0, y0, x1, y1 = bbox
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    sx = dst_w / src_w
    sy = dst_h / src_h
    return (
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        int(round(x1 * sx)),
        int(round(y1 * sy)),
    )


def offset_bbox(
    bbox: tuple[int, int, int, int],
    offset_x: int,
    offset_y: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return x0 + offset_x, y0 + offset_y, x1 + offset_x, y1 + offset_y


def draw_ng_stamp(image: Image.Image | np.ndarray) -> Image.Image:
    """未检出时在图上绘制红色 NG。"""
    pil = _to_rgb_pil(image)
    w, h = pil.size
    text = "NG"
    font_size = max(32, min(w, h) // 8)
    font = _load_annotation_font(font_size)
    draw = ImageDraw.Draw(pil)
    tb = draw.textbbox((0, 0), text, font=font)
    tw = tb[2] - tb[0]
    th = tb[3] - tb[1]
    tx = (w - tw) // 2
    ty = (h - th) // 2
    pad = 8
    draw.rectangle(
        [tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
        fill=(0, 0, 0),
    )
    draw.text((tx, ty), text, fill=(255, 0, 0), font=font)
    return pil


# 多连通域框颜色（RGB）
COMPONENT_COLORS = [
    (255, 64, 0),
    (0, 200, 255),
    (255, 0, 200),
    (0, 255, 120),
    (255, 200, 0),
]


def draw_mask_connected_bboxes(
    image: Image.Image | np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
) -> Image.Image:
    """为每个连通域画框，并在框旁标注 #序号 与宽高。"""
    pil = _to_rgb_pil(image)
    w, h = pil.size
    font_size = max(14, min(w, h) // 28)
    font = _load_annotation_font(font_size)
    draw = ImageDraw.Draw(pil)
    line_w = max(2, min(w, h) // 256)

    for idx, bbox in enumerate(bboxes, start=1):
        x0, y0, x1, y1 = bbox
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(x0 + 1, min(x1, w))
        y1 = max(y0 + 1, min(y1, h))
        bw, bh = bbox_wh((x0, y0, x1, y1))
        color = COMPONENT_COLORS[(idx - 1) % len(COMPONENT_COLORS)]
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=line_w)
        tag = f"#{idx} {bw}x{bh}"
        tb = draw.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        tx, ty = x0, max(0, y0 - th - 4)
        draw.rectangle([tx, ty, tx + tw + 4, ty + th + 2], fill=(0, 0, 0))
        draw.text((tx + 2, ty + 1), tag, fill=(255, 220, 0), font=font)

    if bboxes:
        primary = largest_bbox_index(bboxes)
        pb = bboxes[primary]
        pbw, pbh = bbox_wh(pb)
        summary = f"regions={len(bboxes)} primary={pbw}x{pbh} (original)"
        tb = draw.textbbox((0, 0), summary, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = 6
        draw.rectangle([pad, pad, pad + tw + pad * 2, pad + th + pad], fill=(0, 0, 0))
        draw.text((pad + pad, pad + pad // 2), summary, fill=(255, 220, 0), font=font)

    return pil


def finalize_visualization(
    image: Image.Image | np.ndarray,
    bboxes: list[tuple[int, int, int, int]] | None,
) -> Image.Image:
    """有连通域则逐块画框；无则画 NG。"""
    if not bboxes:
        return draw_ng_stamp(image)
    return draw_mask_connected_bboxes(image, bboxes)


def crop_offset(
    full_size: tuple[int, int],
    roi: tuple[int, int, int, int],
) -> tuple[int, int]:
    """与 crop_roi 一致：返回实际粘贴到原图上的左上角 (x0, y0)。"""
    x, y, _, _ = roi
    img_w, img_h = full_size
    x0 = max(0, min(x, img_w))
    y0 = max(0, min(y, img_h))
    return x0, y0


def paste_mask_to_full(
    mask: np.ndarray,
    full_size: tuple[int, int],
    crop_image: Image.Image,
    crop_roi: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """将模型 mask 缩放到裁剪区后贴到原图尺寸的栅格上，用于原图坐标系 bbox。"""
    full_w, full_h = full_size
    full_mask = np.zeros((full_h, full_w), dtype=np.uint8)

    mask_on_crop = np.array(
        Image.fromarray(mask.astype(np.uint8), mode="L").resize(
            crop_image.size, Image.NEAREST
        )
    )
    crop_w, crop_h = crop_image.size
    if crop_roi is not None:
        off_x, off_y = crop_offset(full_size, crop_roi)
    else:
        off_x, off_y = 0, 0

    y1 = min(off_y + crop_h, full_h)
    x1 = min(off_x + crop_w, full_w)
    rh = y1 - off_y
    rw = x1 - off_x
    if rh > 0 and rw > 0:
        full_mask[off_y:y1, off_x:x1] = mask_on_crop[:rh, :rw]
    return full_mask


def bboxes_on_original(
    mask: np.ndarray,
    crop_image: Image.Image,
    *,
    full_size: tuple[int, int],
    crop_roi: tuple[int, int, int, int] | None,
    min_area: int = 16,
) -> list[tuple[int, int, int, int]]:
    """在原图像素网格上对 mask 做连通域，返回各块外接矩形。"""
    full_mask = paste_mask_to_full(mask, full_size, crop_image, crop_roi)
    return compute_mask_connected_bboxes(full_mask, min_area=min_area)


def build_original_overlay(
    full_image: Image.Image,
    mask: np.ndarray,
    crop_image: Image.Image,
    *,
    input_size: int,
    crop_roi: tuple[int, int, int, int] | None,
    min_area: int = 16,
    alpha: float = 0.5,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """在原图上叠加 mask 着色，并计算原图坐标系下的最小外接矩形。"""
    full_rgb = np.array(full_image)
    mask_on_crop = np.array(
        Image.fromarray(mask.astype(np.uint8), mode="L").resize(
            crop_image.size, Image.NEAREST
        )
    )
    mask_color_crop = mask_to_color(mask_on_crop, CLASS_COLORS)

    cw, ch = crop_image.size
    if crop_roi is not None:
        off_x, off_y = crop_offset(full_image.size, crop_roi)
    else:
        off_x, off_y = 0, 0

    full_out = full_rgb.copy()
    y1 = min(off_y + ch, full_out.shape[0])
    x1 = min(off_x + cw, full_out.shape[1])
    region_h = y1 - off_y
    region_w = x1 - off_x
    if region_h > 0 and region_w > 0:
        region = full_out[off_y:y1, off_x:x1]
        mask_region = mask_color_crop[:region_h, :region_w]
        full_out[off_y:y1, off_x:x1] = overlay_mask(region, mask_region, alpha=alpha)

    bboxes_full = bboxes_on_original(
        mask,
        crop_image,
        full_size=full_image.size,
        crop_roi=crop_roi,
        min_area=min_area,
    )
    return full_out, bboxes_full


def overlay_mask(
    image_rgb: np.ndarray,
    mask_color: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """仅在 mask 非黑区域叠加；alpha 默认 0.5。"""
    base = image_rgb.astype(np.float32)
    fg = mask_color.astype(np.float32)
    fg_region = mask_color.sum(axis=2) > 0
    out = base.copy()
    out[fg_region] = base[fg_region] * (1 - alpha) + fg[fg_region] * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def save_results(
    full_image: Image.Image,
    crop_image: Image.Image,
    resized_rgb: np.ndarray,
    mask: np.ndarray,
    out_dir: Path,
    stem: str,
    *,
    input_size: int,
    crop_roi: tuple[int, int, int, int] | None,
    min_area: int = 16,
    save_all: bool = False,
) -> Path:
    """默认仅保存原图 overlay；连通域 bbox 均在原图坐标系。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_rgb, bboxes_full = build_original_overlay(
        full_image,
        mask,
        crop_image,
        input_size=input_size,
        crop_roi=crop_roi,
        min_area=min_area,
    )

    bboxes_model = compute_mask_connected_bboxes(mask, min_area=min_area)
    print(f"  mask components @ model {mask.shape[1]}x{mask.shape[0]}: {len(bboxes_model)}")
    for i, bb in enumerate(bboxes_model, start=1):
        mw, mh = bbox_wh(bb)
        print(f"    #{i} {mw}x{mh}")

    if bboxes_full:
        print(
            f"  mask components on original ({full_image.size[0]}x{full_image.size[1]}): "
            f"{len(bboxes_full)}"
        )
        for i, bb in enumerate(bboxes_full, start=1):
            bw, bh = bbox_wh(bb)
            print(f"    #{i} {bw}x{bh}")
    else:
        print("  result: NG (no foreground components on original)")

    overlay_path = out_dir / f"{stem}_overlay.png"
    finalize_visualization(overlay_rgb, bboxes_full).save(overlay_path)
    if bboxes_full:
        pri = largest_bbox_index(bboxes_full)
        bw, bh = bbox_wh(bboxes_full[pri])
        print(
            f"  saved: {overlay_path} "
            f"(regions={len(bboxes_full)}, primary={bw}x{bh})"
        )
    else:
        print(f"  saved: {overlay_path} (NG)")

    if not save_all:
        return overlay_path

    mask_img = Image.fromarray(mask.astype(np.uint8), mode="L")
    mask_on_crop = np.array(mask_img.resize(crop_image.size, Image.NEAREST))
    crop_rgb = np.array(crop_image)
    mask_color_crop = mask_to_color(mask_on_crop, CLASS_COLORS)
    overlay_crop = overlay_mask(crop_rgb, mask_color_crop)

    bboxes_crop = compute_mask_connected_bboxes(mask_on_crop, min_area=min_area)

    extras: dict[str, tuple[Image.Image | np.ndarray, list[tuple[int, int, int, int]]]] = {
        "crop.png": (crop_image, bboxes_crop),
        "input_resized.png": (Image.fromarray(resized_rgb), bboxes_model),
        "mask.png": (mask_on_crop, bboxes_crop),
        "mask_color.png": (mask_color_crop, bboxes_crop),
        "overlay_crop.png": (overlay_crop, bboxes_crop),
    }
    for name, (img, bboxes) in extras.items():
        path = out_dir / f"{stem}_{name}"
        finalize_visualization(img, bboxes).save(path)
        print(f"  saved (debug): {path}")

    return overlay_path


def infer_one(
    session,
    mode: str,
    image_path: Path,
    *,
    out_dir: Path,
    input_size: int,
    use_crop: bool,
    crop_roi_used: tuple[int, int, int, int] | None,
    min_area: int = 16,
    save_all: bool = False,
) -> Path:
    """单张推理；返回原图 overlay 路径 {stem}_overlay.png。"""
    full_image = load_rgb(image_path)
    print(f"\n[{image_path.name}] full {full_image.size[0]}x{full_image.size[1]}")

    if use_crop:
        assert crop_roi_used is not None
        crop_image = crop_roi(full_image, crop_roi_used)
        print(f"  crop ROI={crop_roi_used} -> {crop_image.size[0]}x{crop_image.size[1]}")
    else:
        crop_image = full_image
        print("  crop: disabled")

    resized_rgb = np.array(resize_square(crop_image, input_size))
    mask = run_onnx(session, prepare_input(resized_rgb, mode))

    unique = np.unique(mask)
    print(
        f"  mask {mask.shape}, classes={unique.tolist()}, "
        f"range=[{mask.min()}, {mask.max()}]"
    )

    stem = image_path.stem
    return save_results(
        full_image,
        crop_image,
        resized_rgb,
        mask,
        out_dir,
        stem,
        input_size=input_size,
        crop_roi=crop_roi_used if use_crop else None,
        min_area=min_area,
        save_all=save_all,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ONNX inference for one or many images; save mask visualizations",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=DEFAULT_ONNX,
        help=f"ONNX 路径，默认 {DEFAULT_ONNX.name}（embedded）；official 用 lightly_float32.onnx",
    )
    parser.add_argument(
        "--image",
        type=Path,
        nargs="*",
        default=None,
        help="一张或多张图（文件/目录/glob）；默认 None -> 使用 DEFAULT_IMAGE",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="图片目录，可与 --image 组合；默认 None",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="扫描目录时递归子目录；默认 False（仅顶层）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"可视化输出目录，默认 {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="模型输入边长 H=W；默认 512，须与导出 ONNX 时 height/width 一致",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help=f"裁剪 ROI；默认 None 时使用脚本常量 CROP_ROI={CROP_ROI}",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="整图 resize 到 --size，不用 ROI；默认启用 crop（除非加此参数）",
    )
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="onnxruntime 执行提供者；默认 CPU，GPU 可用 CUDAExecutionProvider",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=16,
        help="mask 连通域最小面积（像素），小于此值的块忽略，默认 16",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="额外保存 crop/mask/512 等调试图；默认仅 {stem}_overlay.png（原图坐标）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="matplotlib 显示 overlay；默认 False，且仅单张图时有效",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path = args.onnx.resolve()
    out_dir = args.out_dir.resolve()

    images_arg = list(args.image) if args.image else [DEFAULT_IMAGE]
    image_paths = collect_image_paths(
        images_arg,
        args.image_dir,
        recursive=args.recursive,
    )
    if not image_paths:
        raise FileNotFoundError("No images found. Use --image and/or --image-dir.")

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX not found: {onnx_path}\n"
            "Run: python tools/toOnnx.py  (embedded) or pass --onnx"
        )

    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("Install onnxruntime: pip install onnxruntime") from e

    print(f"ONNX:   {onnx_path}")
    print(f"Output: {out_dir}")
    print(f"Images: {len(image_paths)}")

    session = ort.InferenceSession(
        str(onnx_path),
        providers=[args.provider],
    )
    mode = detect_onnx_mode(session)
    print(f"Mode:   {mode} (input: {session.get_inputs()[0].name})")
    if not args.save_all:
        print("Output: *_overlay.png only (mask+bbox on original image)")
    else:
        print("Output: *_overlay.png + debug images (--save-all)")

    use_crop = not args.no_crop
    crop_roi_used = None if args.no_crop else (
        tuple(args.crop) if args.crop is not None else CROP_ROI
    )

    overlay_paths: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for i, image_path in enumerate(image_paths, start=1):
        print(f"--- {i}/{len(image_paths)}: {image_path} ---")
        try:
            overlay_paths.append(
                infer_one(
                    session,
                    mode,
                    image_path,
                    out_dir=out_dir,
                    input_size=args.size,
                    use_crop=use_crop,
                    crop_roi_used=crop_roi_used,
                    min_area=args.min_area,
                    save_all=args.save_all,
                )
            )
        except Exception as e:
            failed.append((image_path, str(e)))
            print(f"  ERROR: {e}")

    print(f"\nDone: {len(overlay_paths)}/{len(image_paths)} succeeded -> {out_dir}")
    if failed:
        print(f"Failed ({len(failed)}):")
        for path, err in failed:
            print(f"  {path}: {err}")

    if args.show:
        if len(overlay_paths) != 1:
            print("--show ignored: only supported for a single image")
        else:
            import matplotlib.pyplot as plt

            plt.imshow(Image.open(overlay_paths[0]))
            plt.axis("off")
            plt.title("ONNX segmentation overlay")
            plt.show()


if __name__ == "__main__":
    main()
