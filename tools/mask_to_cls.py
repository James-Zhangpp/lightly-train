"""从语义分割数据生成图像分类目录（按 mask 是否含前景类划分）。

输入（与 tools/weldingseg-data 一致）：
  img_dir/{train,val}/*.png
  ann_dir/{train,val}/*.png   # 像素值 = 类别 id

输出（ImageFolder 结构）：
  weldingcls-data/
    train/<class_name>/*.png
    val/<class_name>/*.png

默认：mask 中类别 1（fpc）像素数 >= --min-area -> fpc，否则 no_fpc。
默认带 ROI 裁剪 (900,700,768,768)，与 test.py 一致；--no-crop 则不裁。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_SEG_ROOT = Path(__file__).resolve().parent / "DataSet" /"weldingseg-data"
DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "DataSet" /"weldingcls-data"

DEFAULT_FOREGROUND_ID = 1
CLASS_NAMES = {0: "no_fpc", 1: "fpc"}
DEFAULT_CROP_ROI = (900, 700, 768, 768)


def crop_roi(image: Image.Image, roi: tuple[int, int, int, int]) -> Image.Image:
    x, y, w, h = roi
    img_w, img_h = image.size
    x0 = max(0, min(x, img_w))
    y0 = max(0, min(y, img_h))
    x1 = min(x0 + w, img_w)
    y1 = min(y0 + h, img_h)
    return image.crop((x0, y0, x1, y1))


def label_from_mask(
    mask_path: Path,
    *,
    foreground_id: int,
    min_area: int,
) -> int:
    mask = np.array(Image.open(mask_path))
    fg_pixels = int((mask == foreground_id).sum())
    return 1 if fg_pixels >= min_area else 0


def process_split(
    seg_root: Path,
    out_root: Path,
    split: str,
    *,
    foreground_id: int,
    min_area: int,
    crop_roi: tuple[int, int, int, int] | None,
) -> dict[str, int]:
    img_dir = seg_root / "img_dir" / split
    ann_dir = seg_root / "ann_dir" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing image dir: {img_dir}")

    counts: dict[str, int] = {name: 0 for name in CLASS_NAMES.values()}

    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }:
            continue
        ann_path = ann_dir / f"{img_path.stem}.png"
        if not ann_path.is_file():
            ann_path = ann_dir / f"{img_path.stem}{img_path.suffix}"
        if not ann_path.is_file():
            print(f"[skip] no mask: {img_path.name}")
            continue

        class_id = label_from_mask(
            ann_path, foreground_id=foreground_id, min_area=min_area
        )
        class_name = CLASS_NAMES[class_id]
        out_dir = out_root / split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / img_path.name

        image = Image.open(img_path).convert("RGB")
        if crop_roi is not None:
            image = crop_roi(image, crop_roi)
        image.save(out_path)

        counts[class_name] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="mask 语义分割数据 -> 分类 ImageFolder")
    parser.add_argument("--seg-root", type=Path, default=DEFAULT_SEG_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--foreground-id", type=int, default=DEFAULT_FOREGROUND_ID)
    parser.add_argument(
        "--min-area",
        type=int,
        default=16,
        help="前景像素数下限，低于则标为 no_fpc",
    )
    parser.add_argument(
        "--crop-roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help=f"裁剪 ROI；默认 {DEFAULT_CROP_ROI}",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="不裁剪",
    )
    args = parser.parse_args()

    if args.no_crop:
        crop: tuple[int, int, int, int] | None = None
    elif args.crop_roi is not None:
        crop = tuple(args.crop_roi)
    else:
        crop = DEFAULT_CROP_ROI

    args.out_root.mkdir(parents=True, exist_ok=True)

    print(f"seg_root={args.seg_root}")
    print(f"out_root={args.out_root}")
    print(f"classes={CLASS_NAMES}")
    print(f"foreground_id={args.foreground_id}, min_area={args.min_area}, crop_roi={crop}")

    for split in ("train", "val"):
        counts = process_split(
            args.seg_root,
            args.out_root,
            split,
            foreground_id=args.foreground_id,
            min_area=args.min_area,
            crop_roi=crop,
        )
        print(f"[{split}] {counts}")


if __name__ == "__main__":
    main()
