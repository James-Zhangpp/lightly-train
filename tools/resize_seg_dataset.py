# -*- coding: utf-8 -*-
"""Resize a semantic-segmentation dataset with paired images and masks.

Expected dataset layout:

    <data-root>/
      img_dir/train/*.png
      img_dir/val/*.png
      ann_dir/train/*.png
      ann_dir/val/*.png

Images are resized with linear interpolation. Masks are resized with nearest-neighbor
interpolation so class IDs such as 0 and 1 are not polluted by gray values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "Breakage_small_seg"
DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "Breakage_small_seg_1024x416"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize paired segmentation images and masks to a fixed size.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Input dataset root. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help=f"Output dataset root. Default: {DEFAULT_OUT_ROOT}",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output width. Default: 1024",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=416,
        help="Output height. Default: 416",
    )
    parser.add_argument(
        "--mode",
        choices=("letterbox", "stretch"),
        default="letterbox",
        help=(
            "letterbox keeps aspect ratio and pads to target size; "
            "stretch directly resizes to target size. Default: letterbox"
        ),
    )
    parser.add_argument(
        "--image-pad",
        type=int,
        default=0,
        help="Padding value for images in letterbox mode. Default: 0",
    )
    parser.add_argument(
        "--mask-pad",
        type=int,
        default=0,
        help="Padding class ID for masks in letterbox mode. Default: 0",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to process. Default: train val",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned operations without writing files.",
    )
    return parser.parse_args()


def read_image(path: Path, flags: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    encode_suffix = suffix if suffix in IMAGE_EXTENSIONS else ".png"
    params: list[int] = []
    if encode_suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    ok, buffer = cv2.imencode(encode_suffix, image, params)
    if not ok:
        raise ValueError(f"Failed to encode output image: {path}")
    buffer.tofile(str(path))


def resize_stretch(
    image: np.ndarray,
    target_w: int,
    target_h: int,
    interpolation: int,
) -> np.ndarray:
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def resize_letterbox(
    image: np.ndarray,
    target_w: int,
    target_h: int,
    interpolation: int,
    pad_value: int,
) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid image size: {src_w}x{src_h}")

    scale = min(target_w / src_w, target_h / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))

    resized = cv2.resize(
        image,
        (resized_w, resized_h),
        interpolation=interpolation,
    )

    if image.ndim == 2:
        canvas = np.full((target_h, target_w), pad_value, dtype=image.dtype)
    else:
        canvas = np.full(
            (target_h, target_w, image.shape[2]),
            pad_value,
            dtype=image.dtype,
        )

    top = (target_h - resized_h) // 2
    left = (target_w - resized_w) // 2
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas


def resize_one(
    image: np.ndarray,
    target_w: int,
    target_h: int,
    mode: str,
    interpolation: int,
    pad_value: int,
) -> np.ndarray:
    if mode == "stretch":
        return resize_stretch(image, target_w, target_h, interpolation)
    return resize_letterbox(image, target_w, target_h, interpolation, pad_value)


def iter_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_split(args: argparse.Namespace, split: str) -> tuple[int, int]:
    img_dir = args.data_root / "img_dir" / split
    ann_dir = args.data_root / "ann_dir" / split
    out_img_dir = args.out_root / "img_dir" / split
    out_ann_dir = args.out_root / "ann_dir" / split

    image_paths = iter_files(img_dir)
    mask_paths = {path.name: path for path in iter_files(ann_dir)}

    if not image_paths:
        print(f"[WARN] No images found: {img_dir}")
        return 0, 0

    iterator = image_paths
    if tqdm is not None:
        iterator = tqdm(image_paths, desc=f"resize {split}", unit="img")

    ok_count = 0
    missing_mask_count = 0
    for image_path in iterator:
        mask_path = mask_paths.get(image_path.name)
        if mask_path is None:
            missing_mask_count += 1
            print(f"[WARN] Missing mask, skip: {image_path.name}")
            continue

        image = read_image(image_path, cv2.IMREAD_COLOR)
        mask = read_image(mask_path, cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3 and mask.shape[2] == 4:
            mask = mask[:, :, 0]

        resized_image = resize_one(
            image,
            args.width,
            args.height,
            args.mode,
            cv2.INTER_LINEAR,
            args.image_pad,
        )
        resized_mask = resize_one(
            mask,
            args.width,
            args.height,
            args.mode,
            cv2.INTER_NEAREST,
            args.mask_pad,
        )

        if args.dry_run:
            print(
                f"[DRY] {split}/{image_path.name}: "
                f"{image.shape[1]}x{image.shape[0]} -> {args.width}x{args.height}"
            )
        else:
            write_image(out_img_dir / image_path.name, resized_image)
            write_image(out_ann_dir / mask_path.name, resized_mask)
        ok_count += 1

    return ok_count, missing_mask_count


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.out_root = args.out_root.resolve()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive integers.")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Input dataset root does not exist: {args.data_root}")
    if args.data_root == args.out_root:
        raise ValueError("Output root must be different from input root.")

    print("=" * 72)
    print("Segmentation dataset resize")
    print(f"Input : {args.data_root}")
    print(f"Output: {args.out_root}")
    print(f"Size  : {args.width}x{args.height}")
    print(f"Mode  : {args.mode}")
    print(f"DryRun: {args.dry_run}")
    print("=" * 72)

    total_ok = 0
    total_missing = 0
    for split in args.splits:
        ok_count, missing_mask_count = process_split(args, split)
        total_ok += ok_count
        total_missing += missing_mask_count

    print("=" * 72)
    print(f"Done. Processed pairs: {total_ok}, missing masks: {total_missing}")
    if not args.dry_run:
        print(f"Resized dataset saved to: {args.out_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
