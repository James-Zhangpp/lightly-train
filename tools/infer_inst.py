"""ONNX 实例分割推理与可视化（与 tools/toOnnx_inst.py、tools/train_inst.py 配套）。

流程（与 test.py 类似）：
  全幅图 -> ROI 裁切(可选) -> resize 到模型 image_size -> 归一化 -> ONNX
  -> 各实例 mask 映射回裁切图/原图 -> 画 mask 与 bbox

输出：每张图 {stem}_overlay.png；无检出实例时标 NG。

直接运行：python tools/infer_inst.py
依赖 export_meta.json（python tools/toOnnx_inst.py 生成）。
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "out" / "welding_dinov3_vits16_eomt_inst_v1"
DEFAULT_ONNX = DEFAULT_EXPERIMENT / "exported_models" / "model.onnx"
DEFAULT_META = DEFAULT_EXPERIMENT / "exported_models" / "export_meta.json"

DEFAULT_IMAGE_DIR = Path(r"C:\Users\ASUS\Desktop\D2 L12\原图")
DEFAULT_OUT_DIR = Path(r"C:\Users\ASUS\Desktop\D2 L12\inst_test")

FULL_IMAGE_CROP_ROI = (0, 0, 2048, 2048)
DEFAULT_CROP_ROI: tuple[int, int, int, int] | None = FULL_IMAGE_CROP_ROI
DEFAULT_RESIZE_AFTER_CROP = 1024
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_AREA = 16
DEFAULT_ALPHA = 0.45

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

INSTANCE_COLORS = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 200, 255),
    (255, 64, 128),
    (255, 220, 0),
    (180, 0, 255),
]
NG_COLOR = (255, 0, 0)
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
        if not directory.is_dir():
            raise FileNotFoundError(f"Image directory not found: {directory}")
        paths.extend(list_images_in_dir(directory, recursive=recursive))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def crop_roi(image: Image.Image, roi: tuple[int, int, int, int]) -> Image.Image | None:
    x, y, w, h = roi
    img_w, img_h = image.size
    x0 = max(0, min(x, img_w))
    y0 = max(0, min(y, img_h))
    x1 = min(x0 + w, img_w)
    y1 = min(y0 + h, img_h)
    if x1 <= x0 or y1 <= y0:
        return None
    return image.crop((x0, y0, x1, y1))


def crop_offset(
    full_size: tuple[int, int],
    roi: tuple[int, int, int, int],
) -> tuple[int, int]:
    x, y, _, _ = roi
    img_w, img_h = full_size
    return max(0, min(x, img_w)), max(0, min(y, img_h))


def resize_square(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.BILINEAR)


def load_export_meta(meta_path: Path) -> dict:
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing {meta_path}\nRun: python tools/toOnnx_inst.py"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["image_size"] = tuple(meta["image_size"])
    meta["id_to_name"] = {int(k): v for k, v in meta["classes"].items()}
    if not meta.get("image_normalize"):
        meta["image_normalize"] = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    return meta


def prepare_inst_input(image_rgb: np.ndarray, meta: dict) -> np.ndarray:
    h, w = meta["image_size"]
    pil = Image.fromarray(image_rgb).resize((w, h), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    mean = np.array(meta["image_normalize"]["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(meta["image_normalize"]["std"], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr.transpose(2, 0, 1) - mean) / std
    return arr[np.newaxis, ...].astype(np.float32)


def get_ort_providers() -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    order = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in available] or ["CPUExecutionProvider"]


def parse_instances(
    labels: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    min_area: int,
    id_to_name: dict[int, str],
) -> list[dict]:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    masks = np.asarray(masks)
    if masks.ndim == 4:
        masks = masks[0]
    if masks.ndim != 3:
        raise ValueError(f"Unexpected masks shape: {masks.shape}")

    instances: list[dict] = []
    for i in range(len(scores)):
        score = float(scores[i])
        if score < threshold:
            continue
        m = masks[i]
        if m.dtype == bool:
            binary = m
        else:
            binary = m > 0.5
        area = int(binary.sum())
        if area < min_area:
            continue
        class_id = int(labels[i])
        instances.append({
            "class_id": class_id,
            "class_name": id_to_name.get(class_id, str(class_id)),
            "score": score,
            "mask_model": binary.astype(np.uint8),
        })
    return instances


def mask_model_to_crop(
    mask_model: np.ndarray,
    *,
    model_size: tuple[int, int],
    crop_size: tuple[int, int],
) -> np.ndarray:
    pil = Image.fromarray((mask_model * 255).astype(np.uint8), mode="L")
    pil = pil.resize(crop_size, Image.NEAREST)
    return (np.asarray(pil) > 127).astype(np.uint8)


def mask_bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


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
    out = image.copy()
    w, h = out.size
    draw = ImageDraw.Draw(out)
    font = _load_font(max(32, min(w, h) // 8))
    text = "NG"
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx = (w - tw) // 2
    ty = (h - th) // 2
    draw.rectangle([tx - 12, ty - 8, tx + tw + 12, ty + th + 8], fill=(0, 0, 0))
    draw.text((tx, ty), text, fill=NG_COLOR, font=font)
    return out


def draw_instances_overlay(
    image: Image.Image,
    instances: list[dict],
    *,
    alpha: float,
) -> Image.Image:
    if not instances:
        return draw_ng(image)

    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    blended = base.copy()
    for idx, inst in enumerate(instances):
        color = INSTANCE_COLORS[idx % len(INSTANCE_COLORS)]
        m = inst["mask_crop"]
        sel = m > 0
        if not np.any(sel):
            continue
        color_arr = np.array(color, dtype=np.float32)
        blended[sel] = blended[sel] * (1 - alpha) + color_arr * alpha

    out = Image.fromarray(blended.clip(0, 255).astype(np.uint8))
    w, h = out.size
    draw = ImageDraw.Draw(out)
    font = _load_font(max(14, min(w, h) // 28))
    line_w = max(2, min(w, h) // 256)

    for idx, inst in enumerate(instances):
        color = INSTANCE_COLORS[idx % len(INSTANCE_COLORS)]
        m = inst["mask_crop"]
        bbox = mask_bbox_xyxy(m)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=line_w)
        tag = (
            f"#{idx + 1} {inst['class_name']} {inst['score']:.2f} "
            f"{x1 - x0}x{y1 - y0}"
        )
        tb = draw.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        tx, ty = x0, max(0, y0 - th - 4)
        draw.rectangle([tx, ty, tx + tw + 4, ty + th + 2], fill=(0, 0, 0))
        draw.text((tx + 2, ty + 1), tag, fill=TEXT_COLOR, font=font)

    has_ng = any(inst["class_name"].upper() == "NG" for inst in instances)
    if has_ng:
        out = draw_ng(out)

    return out


def process_one(
    image_path: Path,
    session,
    meta: dict,
    out_dir: Path,
    *,
    crop_roi_arg: tuple[int, int, int, int] | None,
    resize_after_crop: int,
    threshold: float,
    min_area: int,
    alpha: float,
    save_all: bool,
) -> None:
    full_image = load_rgb(image_path)
    crop_image = full_image
    did_crop = False

    if crop_roi_arg is not None:
        cropped = crop_roi(full_image, crop_roi_arg)
        if cropped is None:
            print(f"[skip] ROI 无效: {image_path.name} size={full_image.size}")
            return
        crop_image = cropped
        did_crop = True
        if resize_after_crop > 0:
            crop_image = resize_square(crop_image, resize_after_crop)

    tensor = prepare_inst_input(np.asarray(crop_image), meta)
    labels, masks, scores = session.run(None, {"images": tensor})

    instances = parse_instances(
        labels,
        masks,
        scores,
        threshold=threshold,
        min_area=min_area,
        id_to_name=meta["id_to_name"],
    )

    crop_w, crop_h = crop_image.size
    for inst in instances:
        inst["mask_crop"] = mask_model_to_crop(
            inst.pop("mask_model"),
            model_size=meta["image_size"],
            crop_size=(crop_w, crop_h),
        )

    vis_crop = draw_instances_overlay(crop_image, instances, alpha=alpha)

    if did_crop and crop_roi_arg is not None:
        full_vis = full_image.copy()
        off_x, off_y = crop_offset(full_image.size, crop_roi_arg)
        full_vis.paste(vis_crop, (off_x, off_y))
        vis = full_vis
    else:
        vis = vis_crop

    stem = image_path.stem
    out_path = out_dir / f"{stem}_overlay.png"
    vis.save(out_path)

    n = len(instances)
    size_note = f"crop={did_crop}"
    if did_crop and resize_after_crop > 0:
        size_note += f"->{resize_after_crop}"
    has_ng = any(inst["class_name"].upper() == "NG" for inst in instances)
    status = "NG" if (n == 0 or has_ng) else "OK"
    print(
        f"{image_path.name} ({size_note}) instances={n} "
        f"{status} -> {out_path.name}"
    )
    for idx, inst in enumerate(instances):
        print(f"  - inst #{idx + 1}: class_id={inst['class_id']}, name={inst['class_name']}, score={inst['score']:.4f}")

    if save_all and instances:
        for idx, inst in enumerate(instances):
            m = Image.fromarray(inst["mask_crop"] * 255, mode="L")
            m.save(out_dir / f"{stem}_inst{idx + 1}_mask.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX 实例分割推理可视化")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--crop-roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--full-image-roi", action="store_true")
    parser.add_argument(
        "--resize-after-crop",
        type=int,
        default=DEFAULT_RESIZE_AFTER_CROP,
        help="裁切后缩放到 N×N，再送入模型；0=不缩放",
    )
    parser.add_argument("--save-all", action="store_true", help="额外保存各实例 mask")
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError as e:
        raise SystemExit("请安装: pip install onnxruntime") from e

    onnx_path = args.onnx.resolve()
    if not onnx_path.is_file():
        raise SystemExit(f"ONNX 不存在: {onnx_path}\n请先: python tools/toOnnx_inst.py")

    meta = load_export_meta(args.meta.resolve())

    if args.no_crop:
        crop = None
    elif args.crop_roi is not None:
        crop = tuple(args.crop_roi)
    elif args.full_image_roi:
        crop = FULL_IMAGE_CROP_ROI
    else:
        crop = DEFAULT_CROP_ROI

    resize_after_crop = max(0, args.resize_after_crop)

    paths = collect_image_paths(
        list(args.images),
        args.image_dir if not args.images else None,
        recursive=args.recursive,
    )
    if not paths:
        raise SystemExit(f"未找到图像: {args.image_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(onnx_path), providers=get_ort_providers())

    print(f"ONNX={onnx_path}")
    print(f"classes={meta['id_to_name']}, image_size={meta['image_size']}")
    print(f"crop={crop}, resize_after_crop={resize_after_crop}, threshold={args.threshold}")
    print(f"images={len(paths)}, out={args.out_dir}")

    for path in paths:
        process_one(
            path,
            session,
            meta,
            args.out_dir,
            crop_roi_arg=crop,
            resize_after_crop=resize_after_crop,
            threshold=args.threshold,
            min_area=args.min_area,
            alpha=args.alpha,
            save_all=args.save_all,
        )


if __name__ == "__main__":
    main()
