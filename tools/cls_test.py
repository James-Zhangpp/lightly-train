"""ONNX 图像分类推理与可视化（与 tools/toOnnx_cls.py、tools/train_cls.py 配套）。

注意：请用 Python 运行本脚本，不要用 pytest。
  文件名勿用 test_ 前缀，否则 IDE 会当成单元测试（collected 0 items）。

直接运行（无需参数）：
  python tools/infer_cls.py

使用脚本顶部 DEFAULT_* 配置；可选命令行参数覆盖。

全幅原图流程（与 test.py）：--full-image-roi -> 裁 768×768 -> 缩放到 512 -> 再按模型 image_size 送 ONNX。
已是 512 小图目录：加 --no-crop（不裁 ROI、不缩放到 512）。

依赖 export_meta.json（python tools/toOnnx_cls.py 生成）。
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

# 零参数运行时的默认路径（按需修改）
DEFAULT_IMAGE_DIR = Path(r"C:\Users\ASUS\Desktop\NG\原图")
DEFAULT_OUT_DIR = Path(r"C:\Users\ASUS\Desktop\NG\clstest")
# 产线 NG 样本：语义上为「无有效 FPC」，期望 no_fpc（非类别 id=2 的 NG）
DEFAULT_EXPECT_CLASS = "no_fpc"
DEFAULT_PREPROCESS = "val"
DEFAULT_THRESHOLD = 0.8
DEFAULT_RECURSIVE = False

_FALLBACK_IMAGE_DIR = Path(__file__).resolve().parent / "DataSet" / "weldingcls-data" / "val"
_FALLBACK_OUT_DIR = PROJECT_ROOT / "out" / "welding_cls_test"

# 与 tools/test.py 一致：裁 ROI 后再缩放到 512
FULL_IMAGE_CROP_ROI = (900, 700, 768, 768)
# 零参数默认裁切（DEFAULT_IMAGE_DIR 为全幅原图）；已是 512 小图时用 --no-crop
DEFAULT_CROP_ROI: tuple[int, int, int, int] | None = FULL_IMAGE_CROP_ROI
DEFAULT_RESIZE_AFTER_CROP = 512
VAL_RESIZE_SCALE = 1.143
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


def resize_square(image: Image.Image, size: int) -> Image.Image:
    """双线性缩放到 size×size（与 tools/test.py 一致）。"""
    return image.resize((size, size), Image.Resampling.BILINEAR)


def resize_and_pad_rgb(image: Image.Image, image_size: tuple[int, int]) -> Image.Image:
    target_h, target_w = image_size
    img_w, img_h = image.size
    resize_factor = min(target_h / img_h, target_w / img_w)
    crop_h = max(1, round(img_h * resize_factor))
    crop_w = max(1, round(img_w * resize_factor))
    resized = image.resize((crop_w, crop_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(resized, (0, 0))
    return canvas


def resize_val_style(
    image: Image.Image,
    image_size: tuple[int, int],
    *,
    resize_scale: float = VAL_RESIZE_SCALE,
) -> Image.Image:
    target_h, target_w = image_size
    max_size = int(target_h * resize_scale)
    img_w, img_h = image.size
    scale = max_size / max(img_w, img_h)
    new_w = max(1, round(img_w * scale))
    new_h = max(1, round(img_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    if new_w < target_w or new_h < target_h:
        return resize_and_pad_rgb(image, image_size)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def preprocess_image(image: Image.Image, meta: dict, *, mode: str) -> Image.Image:
    image_size = meta["image_size"]
    if mode == "val":
        return resize_val_style(image, image_size)
    if mode == "predict":
        return resize_and_pad_rgb(image, image_size)
    raise ValueError(f"Unknown preprocess mode: {mode}")


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
    norm = meta.get("image_normalize")
    if not norm:
        meta["image_normalize"] = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    return meta


def prepare_cls_input(pil_rgb: Image.Image, meta: dict) -> np.ndarray:
    arr = np.asarray(pil_rgb, dtype=np.float32) / 255.0
    mean = np.array(meta["image_normalize"]["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(meta["image_normalize"]["std"], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr.transpose(2, 0, 1) - mean) / std
    return arr[np.newaxis, ...].astype(np.float32)


def get_ort_providers() -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    order = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in available] or ["CPUExecutionProvider"]


def run_onnx(session, tensor: np.ndarray) -> tuple[int, float]:
    input_name = session.get_inputs()[0].name
    labels, scores = session.run(None, {input_name: tensor})
    label = int(labels.reshape(-1)[0])
    score = float(scores.reshape(-1)[0])
    return label, score


def is_result_ng(
    class_id: int,
    score: float,
    *,
    threshold: float,
    expect_class: str | None,
    name_to_id: dict[str, int],
) -> bool:
    if score < threshold:
        return True
    if expect_class is None:
        return False
    expect_id = name_to_id.get(expect_class)
    if expect_id is None:
        raise ValueError(
            f"--expect-class '{expect_class}' 不在模型类别中: {list(name_to_id.keys())}"
        )
    return class_id != expect_id


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
        tag = f"NG  pred={class_name}({class_id}) score={score:.3f}"
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
    resize_after_crop: int,
    preprocess_mode: str,
    threshold: float,
    expect_class: str | None,
    compare_torch: bool,
    checkpoint: Path,
) -> dict[str, int]:
    image = load_rgb(image_path)
    did_crop = False
    if crop_roi_arg is not None:
        cropped = crop_roi(image, crop_roi_arg)
        if cropped is None:
            print(
                f"  [warn] ROI 无效，跳过裁切: {image_path.name} size={image.size} roi={crop_roi_arg}"
            )
        else:
            image = cropped
            did_crop = True
            if resize_after_crop > 0:
                image = resize_square(image, resize_after_crop)

    preprocessed = preprocess_image(image, meta, mode=preprocess_mode)
    tensor = prepare_cls_input(preprocessed, meta)
    class_id, score = run_onnx(session, tensor)
    class_name = meta["id_to_name"].get(class_id, str(class_id))

    is_ng = is_result_ng(
        class_id,
        score,
        threshold=threshold,
        expect_class=expect_class,
        name_to_id=meta["name_to_id"],
    )

    vis = draw_result(
        image,
        class_name=class_name,
        class_id=class_id,
        score=score,
        is_ng=is_ng,
    )
    out_path = out_dir / f"{image_path.stem}_cls.png"
    vis.save(out_path)
    size_note = f"crop={did_crop}"
    if did_crop and resize_after_crop > 0:
        size_note += f"->{resize_after_crop}"
    size_note += f" onnx_in={preprocessed.size}"
    print(
        f"{image_path.name} ({size_note}) -> {class_name} score={score:.4f} ng={is_ng} "
        f"[{preprocess_mode}] -> {out_path.name}"
    )

    if compare_torch and checkpoint.is_file():
        import lightly_train

        model = lightly_train.load_model(checkpoint)
        ref = model.predict(image, topk=1)
        ref_id = int(ref["labels"].reshape(-1)[0].item())
        ref_score = float(ref["scores"].reshape(-1)[0].item())
        ref_name = meta["id_to_name"].get(ref_id, str(ref_id))
        print(f"  torch predict: {ref_name} id={ref_id} score={ref_score:.4f}")

    return {"ng": int(is_ng), "ok": int(not is_ng)}


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX 分类推理可视化")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=DEFAULT_RECURSIVE,
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--expect-class", type=str, default=DEFAULT_EXPECT_CLASS)
    parser.add_argument("--positive-class", type=str, default=None)
    parser.add_argument(
        "--preprocess",
        choices=("val", "predict"),
        default=DEFAULT_PREPROCESS,
    )
    parser.add_argument("--crop-roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--no-crop", action="store_true", help="不裁 ROI（已是 512 小图时用）")
    parser.add_argument(
        "--full-image-roi",
        action="store_true",
        help=f"全幅图 ROI {FULL_IMAGE_CROP_ROI}（默认已启用 DEFAULT_CROP_ROI）",
    )
    parser.add_argument(
        "--resize-after-crop",
        type=int,
        default=DEFAULT_RESIZE_AFTER_CROP,
        metavar="N",
        help=f"裁切后缩放到 N×N，默认 {DEFAULT_RESIZE_AFTER_CROP}；0 表示不缩放",
    )
    parser.add_argument("--compare-torch", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    expect_class = args.expect_class or args.positive_class

    image_dir = args.image_dir.expanduser()
    if not image_dir.is_dir() and _FALLBACK_IMAGE_DIR.is_dir():
        print(f"[warn] image-dir 不存在，改用: {_FALLBACK_IMAGE_DIR}")
        image_dir = _FALLBACK_IMAGE_DIR
    args.image_dir = image_dir
    args.out_dir = args.out_dir.expanduser()

    try:
        import onnxruntime as ort
    except ImportError as e:
        raise SystemExit("请安装: pip install onnxruntime") from e

    onnx_path = args.onnx.resolve()
    if not onnx_path.is_file():
        raise SystemExit(f"ONNX 不存在: {onnx_path}\n请先: python tools/toOnnx_cls.py")

    meta = load_export_meta(args.meta.resolve())

    if args.no_crop:
        crop: tuple[int, int, int, int] | None = None
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
        raise SystemExit(f"未找到输入图像: {args.image_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    providers = get_ort_providers()
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    print(f"ONNX={onnx_path}")
    print(f"providers={providers}")
    print(f"classes={meta['id_to_name']}, image_size={meta['image_size']}")
    print(
        f"preprocess={args.preprocess}, crop={crop}, "
        f"resize_after_crop={resize_after_crop}, expect_class={expect_class}"
    )
    print(f"images={len(paths)}, out={args.out_dir}")

    total_ng = 0
    total_ok = 0
    for path in paths:
        stats = process_one(
            path,
            session,
            meta,
            args.out_dir,
            crop_roi_arg=crop,
            resize_after_crop=resize_after_crop,
            preprocess_mode=args.preprocess,
            threshold=args.threshold,
            expect_class=expect_class,
            compare_torch=args.compare_torch,
            checkpoint=args.checkpoint.resolve(),
        )
        total_ng += stats["ng"]
        total_ok += stats["ok"]

    print(f"\nDone: ok={total_ok}, ng={total_ng}, total={len(paths)}")


if __name__ == "__main__":
    main()
