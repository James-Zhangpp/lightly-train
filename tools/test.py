"""Run ONNX semantic segmentation and save mask visualizations."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "out" / "welding_dinov3_vits16_eomt_lr1e4_q50"
DEFAULT_ONNX = DEFAULT_EXPERIMENT / "exported_models" / "model_with_preprocessing.onnx"
DEFAULT_IMAGE = Path(r"C:\Users\ASUS\Desktop\D2 5L\原图")
DEFAULT_OUT_DIR = Path(r"C:\Users\ASUS\Desktop\D2 5L\onnxtest")

# ROI on full image: (x, y, width, height), then resize to --size (default 512).
CROP_ROI: tuple[int, int, int, int] = (900, 700, 768, 768)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# class_id -> RGB for mask visualization (background, fpc)
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
    """Resolve single file, multiple files, directory, or glob patterns."""
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

    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def crop_roi(image: Image.Image, roi: tuple[int, int, int, int]) -> Image.Image:
    """Crop (x, y, width, height). Clips to image bounds."""
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
    return image.resize((size, size), Image.BILINEAR)


def detect_onnx_mode(session) -> str:
    """Return 'embedded' (uint8 HWC) or 'official' (float NCHW)."""
    inp = session.get_inputs()[0]
    shape = inp.shape
    if len(shape) == 4 and shape[-1] in (3, "3"):
        return "embedded"
    return "official"


def prepare_input(image_rgb: np.ndarray, mode: str) -> np.ndarray:
    """image_rgb: HxWx3 uint8 -> model input batch."""
    if mode == "embedded":
        return image_rgb[np.newaxis, ...].astype(np.uint8)

    x = image_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))
    return x[np.newaxis, ...].astype(np.float32)


def run_onnx(session, batch: np.ndarray) -> np.ndarray:
    inp = session.get_inputs()[0]
    outputs = session.get_outputs()

    if len(outputs) == 1:
        out_names = [outputs[0].name]
    else:
        mask_out = next((o for o in outputs if "mask" in o.name.lower()), outputs[0])
        out_names = [mask_out.name]

    result = session.run(out_names, {inp.name: batch})[0]
    if result.ndim == 3:
        return result[0]
    return result


def mask_to_color(mask: np.ndarray, colors: dict[int, tuple[int, int, int]]) -> np.ndarray:
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, rgb in colors.items():
        color[mask == class_id] = rgb
    return color


def overlay_mask(
    image_rgb: np.ndarray,
    mask_color: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    base = image_rgb.astype(np.float32)
    fg = mask_color.astype(np.float32)
    fg_region = mask_color.sum(axis=2) > 0
    out = base.copy()
    out[fg_region] = base[fg_region] * (1 - alpha) + fg[fg_region] * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def save_results(
    crop_image: Image.Image,
    resized_rgb: np.ndarray,
    mask: np.ndarray,
    out_dir: Path,
    stem: str,
    *,
    full_image: Image.Image | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray(mask.astype(np.uint8), mode="L")
    mask_on_crop = mask_img.resize(crop_image.size, Image.NEAREST)

    crop_rgb = np.array(crop_image)
    mask_color_model = mask_to_color(mask, CLASS_COLORS)
    mask_color_crop = mask_to_color(np.array(mask_on_crop), CLASS_COLORS)

    overlay_model = overlay_mask(resized_rgb, mask_color_model)
    overlay_crop = overlay_mask(crop_rgb, mask_color_crop)

    paths: dict[str, Image.Image] = {
        "crop.png": crop_image,
        "input_resized.png": Image.fromarray(resized_rgb),
        "mask.png": mask_on_crop,
        "mask_color.png": Image.fromarray(mask_color_crop),
        "overlay.png": Image.fromarray(overlay_crop),
        "overlay_resized_input.png": Image.fromarray(overlay_model),
    }
    if full_image is not None:
        paths["input_full.png"] = full_image

    for name, img in paths.items():
        path = out_dir / f"{stem}_{name}"
        img.save(path)
        print(f"  saved: {path}")


def infer_one(
    session,
    mode: str,
    image_path: Path,
    *,
    out_dir: Path,
    input_size: int,
    use_crop: bool,
    crop_roi_used: tuple[int, int, int, int] | None,
) -> Path:
    """Run inference on one image; return path to overlay.png."""
    full_image = load_rgb(image_path)
    print(f"\n[{image_path.name}] full {full_image.size[0]}x{full_image.size[1]}")

    if use_crop:
        assert crop_roi_used is not None
        crop_image = crop_roi(full_image, crop_roi_used)
        print(f"  crop ROI={crop_roi_used} -> {crop_image.size[0]}x{crop_image.size[1]}")
        full_for_save = full_image
    else:
        crop_image = full_image
        full_for_save = None
        print("  crop: disabled")

    resized_rgb = np.array(resize_square(crop_image, input_size))
    mask = run_onnx(session, prepare_input(resized_rgb, mode))

    unique = np.unique(mask)
    print(
        f"  mask {mask.shape}, classes={unique.tolist()}, "
        f"range=[{mask.min()}, {mask.max()}]"
    )

    stem = image_path.stem
    save_results(
        crop_image,
        resized_rgb,
        mask,
        out_dir,
        stem,
        full_image=full_for_save,
    )
    return out_dir / f"{stem}_overlay.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ONNX inference for one or many images; save mask visualizations",
    )
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="ONNX model path")
    parser.add_argument(
        "--image",
        type=Path,
        nargs="*",
        default=None,
        help="One or more image paths (file, directory, or glob). Default: built-in sample",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory of images (use with or without --image)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan --image-dir (or directory passed to --image)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for output mask images",
    )
    parser.add_argument("--size", type=int, default=512, help="Model input H=W")
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help=f"Crop ROI x y width height (default: {CROP_ROI})",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Use full image (resize to --size) instead of ROI crop",
    )
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="onnxruntime provider, e.g. CUDAExecutionProvider",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show overlay (single image only)",
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
            f"ONNX not found: {onnx_path}\nRun: python tools/toOnnx.py"
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
