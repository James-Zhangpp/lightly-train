"""ONNX 图像分类推理与可视化（与 tools/toOnnx_cls.py、tools/train_cls.py 配套）。

预处理与 lightly_train ImageClassification.predict 一致：
  RGB -> [0,1] -> ImageNet 归一化 -> 保持宽高比缩放到 image_size 并左上角 padding。

默认输出：每张图 {stem}_cls.png（类别名 + 置信度）；低于阈值或类别不符时标 NG。

依赖同目录 export_meta.json（由 toOnnx_cls.py 生成）。
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "out" / "welding_dinov3_vitt16_cls_v1"
DEFAULT_ONNX = DEFAULT_EXPERIMENT / "exported_models" / "model.onnx"
DEFAULT_META = DEFAULT_EXPERIMENT / "exported_models" / "export_meta.json"
DEFAULT_CHECKPOINT = DEFAULT_EXPERIMENT / "exported_models" / "exported_best.pt"

DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent / "weldingcls-data" / "val"
DEFAULT_OUT_DIR = PROJECT_ROOT / "out" / "welding_cls_test"

CROP_ROI: tuple[int, int, int, int] | None = (900, 700, 768, 768)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

TEXT_COLOR = (255, 220, 0)
NG_COLOR = (255, 0, 0)


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


def resize_and_pad_rgb(
    image: Image.Image,
    image_size: tuple[int, int],
) -> Image.Image:
    """与 ImageClassification.resize_and_pad 一致：等比缩放后 pad 到 image_size（左上对齐）。"""
    target_h, target_w = image_size
    img_w, img_h = image.size
    resize_factor = min(target_h / img_h, target_w / img_w)
    crop_h = max(1, round(img_h * resize_factor))
    crop_w = max(1, round(img_w * resize_factor))
    resized = image.resize((crop_w, crop_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(resized, (0, 0))
    return canvas


def load_export_meta(meta_path: Path) -> dict:
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing {meta_path}\nRun: python tools/toOnnx_cls.py"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["image_size"] = tuple(meta["image_size"])
    id_to_name = {int(k): v for k, v in meta["classes"].items()}
    meta["id_to_name"] = id_to_name
    meta["name_to_id"] = {v: k for k, v in id_to_name.items()}
    return meta


def prepare_cls_input(
    image_rgb: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """HxWx3 uint8 -> (1,3,H,W) float32，已归一化。"""
    pil = Image.fromarray(image_rgb)
    pil = resize_and_pad_rgb(pil, meta["image_size"])
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    mean = np.array(meta["image_normalize"]["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(meta["image_normalize"]["std"], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr.transpose(2, 0, 1) - mean) / std
    return arr[np.newaxis, ...].astype(np.float32)


def run_onnx(session, tensor: np.ndarray) -> tuple[int, float]:
    input_name = session.get_inputs()[0].name
    labels, scores = session.run(None, {input_name: tensor})
    label = int(labels.reshape(-1)[0])
    score = float(scores.reshape(-1)[0])
    return label, score


def draw_result(
    image: Image.Image,
    *,
    class_name: str,
    class_id: int,
    score: float,
    is_ng: bool,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()

    if is_ng:
        tag = f"NG  score={score:.3f}"
        color = NG_COLOR
    else:
        tag = f"{class_name} (id={class_id})  {score:.3f}"
        color = TEXT_COLOR

    tb = draw.textbbox((0, 0), tag, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    pad = 8
    draw.rectangle([pad, pad, pad + tw + pad * 2, pad + th + pad * 2], fill=(0, 0, 0))
    draw.text((pad + pad, pad + pad // 2), tag, fill=color, font=font)
    return out


def process_one(
    image_path: Path,
    session,
    meta: dict,
    out_dir: Path,
    *,
    crop_roi_arg: tuple[int, int, int, int] | None,
    threshold: float,
    positive_class: str,
    compare_torch: bool,
    checkpoint: Path,
) -> None:
    image = load_rgb(image_path)
    if crop_roi_arg is not None:
        image = crop_roi(image, crop_roi_arg)

    tensor = prepare_cls_input(np.asarray(image), meta)
    class_id, score = run_onnx(session, tensor)
    class_name = meta["id_to_name"].get(class_id, str(class_id))

    expect_id = meta["name_to_id"].get(positive_class)
    is_ng = score < threshold
    if expect_id is not None and class_id != expect_id:
        is_ng = True

    vis = draw_result(
        image,
        class_name=class_name,
        class_id=class_id,
        score=score,
        is_ng=is_ng,
    )
    out_path = out_dir / f"{image_path.stem}_cls.png"
    vis.save(out_path)
    print(f"{image_path.name} -> {class_name} score={score:.4f} ng={is_ng} -> {out_path.name}")

    if compare_torch and checkpoint.is_file():
        import lightly_train

        model = lightly_train.load_model(checkpoint)
        ref = model.predict(image, topk=1)
        ref_id = int(ref["labels"].reshape(-1)[0].item())
        ref_score = float(ref["scores"].reshape(-1)[0].item())
        print(f"  torch ref: id={ref_id} score={ref_score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX 分类推理可视化")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5, help="置信度低于此值标 NG")
    parser.add_argument(
        "--positive-class",
        type=str,
        default="fpc",
        help="期望类别名；预测不符也标 NG",
    )
    parser.add_argument(
        "--crop-roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
    )
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--compare-torch", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError as e:
        raise SystemExit("请安装: pip install onnxruntime") from e

    meta = load_export_meta(args.meta.resolve())

    if args.no_crop:
        crop: tuple[int, int, int, int] | None = None
    elif args.crop_roi is not None:
        crop = tuple(args.crop_roi)
    else:
        crop = CROP_ROI

    paths = collect_image_paths(
        list(args.images),
        args.image_dir,
        recursive=args.recursive,
    )
    if not paths:
        raise SystemExit("未找到输入图像")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(
        str(args.onnx.resolve()),
        providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
    )

    print(f"ONNX={args.onnx}")
    print(f"classes={meta['id_to_name']}, image_size={meta['image_size']}, crop={crop}")
    print(f"images={len(paths)}, out={args.out_dir}")

    for path in paths:
        process_one(
            path,
            session,
            meta,
            args.out_dir,
            crop_roi_arg=crop,
            threshold=args.threshold,
            positive_class=args.positive_class,
            compare_torch=args.compare_torch,
            checkpoint=args.checkpoint.resolve(),
        )


if __name__ == "__main__":
    main()
