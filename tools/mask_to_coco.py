"""将 weldingseg mask 转为 COCO instance segmentation + 每图 YOLO-seg txt（每行一个对象）。

输入：
  weldingseg-data/img_dir/{train,val}/
  weldingseg-data/ann_dir/{train,val}/   # 像素 0=background, 1=fpc（或 0/255）

输出：
  weldingseg-coco/images/{train,val}/*.png
  weldingseg-coco/annotations/instances_{train,val}.json
  weldingseg-coco/labels/{train,val}/{stem}.txt   # 每行一个连通域对象

COCO 约定：file_name 仅为文件名（如 6L_001_crop.png），与 image_dir 拼接加载。
YOLO txt：class_id x1 y1 x2 y2 ...（归一化到 0~1），class_id 从 0 起（fpc=0）。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# 与 tools/train.py 一致
CLASSES = {
    0: "background",
    1: "fpc",
}
SKIP_CLASS_IDS = {0}
REMAP_255_TO_CLASS = 1

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "weldingseg-data"
OUT_ROOT = PROJECT_ROOT / "weldingseg-coco"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def normalize_mask_ids(mask: np.ndarray) -> np.ndarray:
    """统一为类别 id 栅格（uint8）。"""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    uniq = set(np.unique(m).tolist())
    if uniq <= {0, 255} or uniq <= {0, 255, 1}:
        out = np.zeros(m.shape, dtype=np.uint8)
        out[m > 0] = REMAP_255_TO_CLASS
        return out
    return m.astype(np.uint8)


def _simplify_contour(cnt: np.ndarray, *, epsilon_ratio: float = 0.002) -> np.ndarray:
    peri = cv2.arcLength(cnt, True)
    eps = max(epsilon_ratio * peri, 1.0)
    return cv2.approxPolyDP(cnt, eps, True)


def connected_component_polygons(
    class_mask: np.ndarray,
    *,
    min_area: int = 16,
) -> list[list[float]]:
    """每个连通域生成一个 polygon（一个 COCO / txt 对象）。"""
    binary = (class_mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    polygons: list[list[float]] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        cnt = _simplify_contour(cnt)
        if cnt.shape[0] < 3:
            continue
        pts = cnt.squeeze(1)
        if pts.ndim != 2 or len(pts) < 3:
            continue
        poly = pts.flatten().astype(float).tolist()
        if len(poly) >= 6:
            polygons.append(poly)
    return polygons


def polygon_area(poly: list[float]) -> float:
    pts = np.array(poly, dtype=np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_bbox(poly: list[float]) -> list[float]:
    pts = np.array(poly, dtype=np.float64).reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def coco_category_to_yolo_id(category_id: int, cat_ids: list[int]) -> int:
    """COCO category_id -> YOLO 0-based class index。"""
    ordered = sorted(cat_ids)
    return ordered.index(category_id)


def polygon_to_yolo_line(
    poly: list[float],
    *,
    yolo_class_id: int,
    width: int,
    height: int,
) -> str:
    pts = np.array(poly, dtype=np.float64).reshape(-1, 2)
    parts = [str(yolo_class_id)]
    for x, y in pts:
        parts.append(f"{x / width:.6f}")
        parts.append(f"{y / height:.6f}")
    return " ".join(parts)


def convert_split(
    split: str,
    *,
    copy_images: bool = True,
    min_area: int = 16,
    export_txt: bool = True,
) -> dict:
    img_dir = DATA_ROOT / "img_dir" / split
    ann_dir = DATA_ROOT / "ann_dir" / split
    if not img_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(f"Missing {img_dir} or {ann_dir}")

    out_img_dir = OUT_ROOT / "images" / split
    out_label_dir = OUT_ROOT / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    if export_txt:
        out_label_dir.mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": cid, "name": name, "supercategory": "object"}
        for cid, name in sorted(CLASSES.items())
        if cid not in SKIP_CLASS_IDS
    ]
    cat_ids = [c["id"] for c in categories]

    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1
    image_id = 1

    mask_files = sorted(
        p for p in ann_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )

    skipped_no_image = 0
    images_empty_mask = 0

    for mask_path in mask_files:
        stem = mask_path.stem
        img_path = None
        for ext in IMG_EXTS:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.is_file():
                img_path = candidate
                break
        if img_path is None:
            skipped_no_image += 1
            print(f"[skip] no image for mask: {mask_path.name}")
            continue

        with Image.open(img_path) as im:
            width, height = im.size

        if copy_images:
            dst = out_img_dir / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)

        # 仅文件名，避免 images/train/train/xxx 路径错误
        file_name = img_path.name

        images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        })

        mask = normalize_mask_ids(np.array(Image.open(mask_path)))
        yolo_lines: list[str] = []

        for class_id in cat_ids:
            class_mask = (mask == class_id).astype(np.uint8)
            for poly in connected_component_polygons(class_mask, min_area=min_area):
                bbox = polygon_bbox(poly)
                area = polygon_area(poly)
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "segmentation": [poly],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                })
                ann_id += 1

                if export_txt:
                    yolo_cls = coco_category_to_yolo_id(class_id, cat_ids)
                    yolo_lines.append(
                        polygon_to_yolo_line(
                            poly,
                            yolo_class_id=yolo_cls,
                            width=width,
                            height=height,
                        )
                    )

        if export_txt:
            txt_path = out_label_dir / f"{stem}.txt"
            txt_path.write_text(
                "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
                encoding="utf-8",
            )

        if not yolo_lines and export_txt:
            images_empty_mask += 1

        image_id += 1

    print(
        f"  [{split}] skip(no image)={skipped_no_image}, "
        f"empty mask (no object)={images_empty_mask}"
    )
    return {
        "info": {
            "description": "weldingseg converted to COCO",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert mask dataset to COCO + YOLO txt")
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_ROOT,
        help=f"输出根目录，默认 {OUT_ROOT}",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=16,
        help="连通域最小面积（像素），小于此忽略，默认 16",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="不复制图像到输出目录",
    )
    parser.add_argument(
        "--no-txt",
        action="store_true",
        help="不导出 labels/{split}/*.txt（每行一个对象）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global OUT_ROOT
    OUT_ROOT = args.out.resolve()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "annotations").mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        coco = convert_split(
            split,
            copy_images=not args.no_copy_images,
            min_area=args.min_area,
            export_txt=not args.no_txt,
        )
        out_json = OUT_ROOT / "annotations" / f"instances_{split}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False, indent=2)
        print(
            f"{split}: images={len(coco['images'])}, "
            f"annotations={len(coco['annotations'])}, "
            f"txt_dir={OUT_ROOT / 'labels' / split}, -> {out_json}"
        )


if __name__ == "__main__":
    main()
