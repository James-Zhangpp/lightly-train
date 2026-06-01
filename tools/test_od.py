"""ONNX 目标检测推理与可视化（与 tools/toOnnx_od.py、tools/train_od.py 配套）。

默认流程：原图 ROI 裁剪 -> resize 到训练尺寸 -> 归一化 -> ONNX -> 框映射回原图。

默认输出：每张图仅 {stem}_overlay.png
  - 绘制检测框、类别、置信度；主目标框标注宽高（原图像素）
  - 无检出（score < threshold）时标注红色 NG

依赖同目录 export_meta.json（由 toOnnx_od.py 生成）。
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "out" / "welding_dinov3_vits16_ltdetr_od_v1"
DEFAULT_ONNX = DEFAULT_EXPERIMENT / "exported_models" / "model.onnx"
DEFAULT_META = DEFAULT_EXPERIMENT / "exported_models" / "export_meta.json"

DEFAULT_IMAGE = Path(r"C:\Users\ASUS\Desktop\D2 5L\原图")
DEFAULT_OUT_DIR = Path(r"C:\Users\ASUS\Desktop\D2 5L\od_test")

CROP_ROI: tuple[int, int, int, int] = (900, 700, 768, 768)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

BOX_COLOR = (255, 64, 0)
TEXT_COLOR = (255, 220, 0)


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def list_images_in_dir(directory: Path, *, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_image_paths(
    images: list[Path | str] | None,
    image_dir: Path | str | None,
    *,
    recursive: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    for raw in images or []:
        source = Path(raw).expanduser()
        source_str = str(source)
        if any(ch in source_str for ch in "*?[]"):
            paths.extend(
                Path(p) for p in glob.glob(source_str, recursive=True) if Path(p).is_file()
            )
            continue
        source = source.resolve()
        if source.is_file():
            if source.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(source)
        elif source.is_dir():
            paths.extend(list_images_in_dir(source, recursive=recursive))
        else:
            raise FileNotFoundError(f"Image path not found: {source}")
    if image_dir is not None:
        directory = Path(image_dir).expanduser().resolve()
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
    x, y, w, h = roi
    img_w, img_h = image.size
    x0 = max(0, min(x, img_w))
    y0 = max(0, min(y, img_h))
    x1 = min(x0 + w, img_w)
    y1 = min(y0 + h, img_h)
    return image.crop((x0, y0, x1, y1))


def crop_offset(
    full_size: tuple[int, int],
    roi: tuple[int, int, int, int],
) -> tuple[int, int]:
    x, y, _, _ = roi
    img_w, img_h = full_size
    return max(0, min(x, img_w)), max(0, min(y, img_h))


def resize_square(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.BILINEAR)


def load_export_meta(meta_path: Path) -> dict:
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing {meta_path}\nRun: python tools/toOnnx_od.py"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["image_size"] = tuple(meta["image_size"])
    return meta


def prepare_od_input(
    image_rgb: np.ndarray,
    *,
    input_h: int,
    input_w: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    pil = Image.fromarray(image_rgb).resize((input_w, input_h), Image.BILINEAR)
    x = np.array(pil, dtype=np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    mean_arr = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.array(std, dtype=np.float32).reshape(3, 1, 1)
    x = (x - mean_arr) / std_arr
    return x[np.newaxis, ...].astype(np.float32)


def run_od_onnx(
    session,
    batch: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 labels[N], boxes[N,4], scores[N]（模型输入分辨率 xyxy）。"""
    labels, boxes, scores = session.run(None, {"images": batch})
    labels = np.asarray(labels).reshape(-1)
    boxes = np.asarray(boxes).reshape(-1, 4)
    scores = np.asarray(scores).reshape(-1)
    keep = scores > threshold
    return labels[keep], boxes[keep], scores[keep]


def boxes_model_to_original(
    boxes: np.ndarray,
    *,
    model_size: tuple[int, int],
    crop_size: tuple[int, int],
    full_size: tuple[int, int],
    crop_roi: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """模型输入尺度 xyxy -> 原图 xyxy。"""
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    model_h, model_w = model_size
    crop_w, crop_h = crop_size
    sx = crop_w / model_w
    sy = crop_h / model_h
    out = boxes.copy().astype(np.float64)
    out[:, [0, 2]] *= sx
    out[:, [1, 3]] *= sy
    if crop_roi is not None:
        off_x, off_y = crop_offset(full_size, crop_roi)
        out[:, [0, 2]] += off_x
        out[:, [1, 3]] += off_y
    return out


def bbox_wh_xyxy(box: np.ndarray) -> tuple[int, int]:
    x0, y0, x1, y1 = box
    return int(round(x1 - x0)), int(round(y1 - y0))


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "msyh.ttc",
        "C:/Windows/Fonts/msyh.ttc",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_ng(image: Image.Image) -> Image.Image:
    pil = image.copy()
    w, h = pil.size
    font = _load_font(max(32, min(w, h) // 8))
    draw = ImageDraw.Draw(pil)
    text = "NG"
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx, ty = (w - tw) // 2, (h - th) // 2
    draw.rectangle([tx - 8, ty - 8, tx + tw + 8, ty + th + 8], fill=(0, 0, 0))
    draw.text((tx, ty), text, fill=(255, 0, 0), font=font)
    return pil


def draw_detections(
    image: Image.Image,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    classes: dict[str, str],
    *,
    primary_idx: int | None,
) -> Image.Image:
    pil = image.copy()
    draw = ImageDraw.Draw(pil)
    w, h = pil.size
    font = _load_font(max(14, min(w, h) // 28))
    line_w = max(2, min(w, h) // 256)

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x0, y0, x1, y1 = box
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=BOX_COLOR, width=line_w)
        name = classes.get(str(int(label)), str(int(label)))
        bw, bh = bbox_wh_xyxy(box)
        tag = f"{name} {score:.2f}"
        if primary_idx is not None and i == primary_idx:
            tag = f"bbox: {bw}x{bh} (original) {tag}"
        tb = draw.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y0 - th - 4)
        draw.rectangle([x0, ty, x0 + tw + 6, ty + th + 4], fill=(0, 0, 0))
        draw.text((x0 + 3, ty + 2), tag, fill=TEXT_COLOR, font=font)

    if primary_idx is None and len(boxes) == 0:
        return draw_ng(pil)
    return pil


def pick_primary_box(boxes: np.ndarray) -> int | None:
    if len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))


def infer_one(
    session,
    meta: dict,
    image_path: Path,
    *,
    out_dir: Path,
    threshold: float,
    use_crop: bool,
    crop_roi_used: tuple[int, int, int, int] | None,
) -> Path:
    full_image = load_rgb(image_path)
    print(f"\n[{image_path.name}] full {full_image.size[0]}x{full_image.size[1]}")

    if use_crop:
        assert crop_roi_used is not None
        crop_image = crop_roi(full_image, crop_roi_used)
        print(f"  crop ROI={crop_roi_used} -> {crop_image.size[0]}x{crop_image.size[1]}")
    else:
        crop_image = full_image

    input_h, input_w = meta["image_size"]
    norm = meta["image_normalize"]
    mean = tuple(norm["mean"])
    std = tuple(norm["std"])

    crop_rgb = np.array(crop_image)
    batch = prepare_od_input(
        crop_rgb, input_h=input_h, input_w=input_w, mean=mean, std=std
    )

    labels, boxes, scores = run_od_onnx(session, batch, threshold=threshold)
    boxes_orig = boxes_model_to_original(
        boxes,
        model_size=(input_h, input_w),
        crop_size=crop_image.size,
        full_size=full_image.size,
        crop_roi=crop_roi_used if use_crop else None,
    )

    print(f"  detections: {len(boxes_orig)} (threshold={threshold})")
    if len(boxes_orig) > 0:
        idx = pick_primary_box(boxes_orig)
        assert idx is not None
        bw, bh = bbox_wh_xyxy(boxes_orig[idx])
        print(f"  primary bbox on original: {bw}x{bh}")

    classes = meta.get("classes", {"0": "fpc"})
    primary_idx = pick_primary_box(boxes_orig)
    overlay = draw_detections(
        full_image,
        boxes_orig,
        labels,
        scores,
        classes,
        primary_idx=primary_idx,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    out_path = out_dir / f"{stem}_overlay.png"
    overlay.save(out_path)
    print(f"  saved: {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX object-detection inference")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--image", type=Path, nargs="*", default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="置信度阈值，默认 0.5",
    )
    parser.add_argument("--size", type=int, default=None, help="覆盖 meta 中的输入边长（正方形）")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None)
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path = args.onnx.resolve()
    meta_path = args.meta.resolve()
    out_dir = args.out_dir.resolve()

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}\nRun: python tools/toOnnx_od.py")

    meta = load_export_meta(meta_path)
    if args.size is not None:
        meta["image_size"] = (args.size, args.size)

    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("pip install onnxruntime") from e

    images_arg = list(args.image) if args.image else [DEFAULT_IMAGE]
    image_paths = collect_image_paths(images_arg, args.image_dir, recursive=args.recursive)
    if not image_paths:
        raise FileNotFoundError("No images found.")

    print(f"ONNX:   {onnx_path}")
    print(f"Meta:   {meta_path}")
    print(f"Output: {out_dir}")
    print(f"Images: {len(image_paths)}")
    print(f"Input:  {meta['image_size']}, threshold={args.threshold}")

    session = ort.InferenceSession(str(onnx_path), providers=[args.provider])

    use_crop = not args.no_crop
    crop_roi_used = None if args.no_crop else (
        tuple(args.crop) if args.crop is not None else CROP_ROI
    )

    overlay_paths: list[Path] = []
    for i, image_path in enumerate(image_paths, start=1):
        print(f"--- {i}/{len(image_paths)}: {image_path} ---")
        try:
            overlay_paths.append(
                infer_one(
                    session,
                    meta,
                    image_path,
                    out_dir=out_dir,
                    threshold=args.threshold,
                    use_crop=use_crop,
                    crop_roi_used=crop_roi_used,
                )
            )
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone: {len(overlay_paths)}/{len(image_paths)} -> {out_dir}")

    if args.show and len(overlay_paths) == 1:
        import matplotlib.pyplot as plt

        plt.imshow(Image.open(overlay_paths[0]))
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
