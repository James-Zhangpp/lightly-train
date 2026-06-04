"""从语义分割数据生成图像分类目录（按 mask 像素类别 id 划分）。

输入：
  img_dir/{train,val}/*.png
  ann_dir/{train,val}/*.png   # 像素值 = 语义分割类别 id

输出（ImageFolder 结构）：
  <out-root>/train/<class_name>/*.png
  <out-root>/val/<class_name>/*.png

分类规则（与产线一致）：
  mask 中 fpc(1) 像素 >= --min-area -> 文件夹 fpc
  否则 -> no_fpc（含 mask 像素 2「NG」、全背景、无 FPC 等）

分割 mask 里的 2 仅表示缺陷区域，图像级分类统一为 no_fpc，不单独建 NG 类。

默认不裁剪（512×512 数据）；全幅相机图用 --full-image-roi。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_SEG_ROOT = Path(__file__).resolve().parent / "DataSet" / "weldingseg-data"
DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "DataSet" / "weldingcls-data"

# 仅两类目录（与 train_cls.py CLASSES 一致）
CLASS_NAMES: dict[int, str] = {
    0: "no_fpc",
    1: "fpc",
}
FPC_CLASS_ID = 1
# 全幅原图 + 现场 ROI 时用；512 训练集不要设默认 ROI
FULL_IMAGE_CROP_ROI = (900, 700, 768, 768)


def crop_roi(
    image: Image.Image, roi: tuple[int, int, int, int]
) -> Image.Image | None:
    """裁剪 ROI；若原图小于 ROI 起点导致宽高为 0，返回 None。"""
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return None
    img_w, img_h = image.size
    x0 = max(0, min(x, img_w))
    y0 = max(0, min(y, img_h))
    x1 = min(x0 + w, img_w)
    y1 = min(y0 + h, img_h)
    if x1 <= x0 or y1 <= y0:
        return None
    return image.crop((x0, y0, x1, y1)).copy()


def classify_from_mask_array(mask: np.ndarray, *, min_area: int) -> int:
    """有足够 fpc(1) 像素则为 fpc，否则 no_fpc（mask 为 2 的 NG 图也归 no_fpc）。"""
    if int((mask == FPC_CLASS_ID).sum()) >= min_area:
        return FPC_CLASS_ID
    return 0


def load_mask_for_label(mask_path: Path, roi: tuple[int, int, int, int] | None) -> np.ndarray | None:
    with Image.open(mask_path) as src:
        mask_img = src.convert("L")
        if roi is None:
            return np.array(mask_img)
        cropped = crop_roi(mask_img, roi)
        if cropped is None:
            return None
        return np.array(cropped)


def process_split(
    seg_root: Path,
    out_root: Path,
    split: str,
    *,
    min_area: int,
    roi: tuple[int, int, int, int] | None,
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

        mask_arr = load_mask_for_label(ann_path, roi)
        if mask_arr is None:
            with Image.open(img_path) as src:
                size = src.size
            print(
                f"[skip] ROI 超出图像: {img_path.name} size={size} roi={roi}"
            )
            continue

        class_id = classify_from_mask_array(mask_arr, min_area=min_area)
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        out_dir = out_root / split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / img_path.name

        with Image.open(img_path) as src:
            image = src.convert("RGB").copy()
        if roi is not None:
            cropped = crop_roi(image, roi)
            assert cropped is not None
            image = cropped
        image.save(out_path, format="PNG")

        counts[class_name] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="mask 语义分割数据 -> 分类 ImageFolder")
    parser.add_argument("--seg-root", type=Path, default=DEFAULT_SEG_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--min-area",
        type=int,
        default=16,
        help="某类 mask 像素数 >= 此值才归入该类，否则为 no_fpc(0)",
    )
    parser.add_argument(
        "--crop-roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="启用 ROI 裁剪 (x y w h)；512 数据勿用。全幅图示例: --crop-roi 900 700 768 768",
    )
    parser.add_argument(
        "--full-image-roi",
        action="store_true",
        help=f"等价于 --crop-roi {' '.join(map(str, FULL_IMAGE_CROP_ROI))}（仅全幅相机图）",
    )
    args = parser.parse_args()

    if args.full_image_roi:
        crop: tuple[int, int, int, int] | None = FULL_IMAGE_CROP_ROI
    elif args.crop_roi is not None:
        crop = tuple(args.crop_roi)
    else:
        crop = None

    args.out_root.mkdir(parents=True, exist_ok=True)

    print(f"seg_root={args.seg_root}")
    print(f"out_root={args.out_root}")
    print(f"classes={CLASS_NAMES}")
    print(f"rule: fpc(1)>={args.min_area}px else no_fpc; mask id 2 -> no_fpc")
    print(f"min_area={args.min_area}, crop_roi={crop}")

    for split in ("train", "val"):
        counts = process_split(
            args.seg_root,
            args.out_root,
            split,
            min_area=args.min_area,
            roi=crop,
        )
        print(f"[{split}] {counts}")


if __name__ == "__main__":
    main()
