"""FPC_QK_140 语义分割 Mask 转 COCO 实例分割数据集脚本。

输入：
  tools/DataSet/FPC_QK_140/
    img_dir/{train,val}/   # 原图
    ann_dir/{train,val}/   # 语义分割 Mask（0=背景，>0 为前景 FPC）

输出：
  tools/DataSet/FPC_QK_140_InSeg_coco/
    train/
      *.png
      _annotations.coco.json
    valid/
      *.png
      _annotations.coco.json
    test/
      *.png
      _annotations.coco.json
    visualizations/
      *_vis.png (30 张左右对比抽检可视化图)

功能特性：
  1. 比例分配参数调整：支持 CLI 传入 `--split 7:2:1` 或 `--split 8:1:1` 等灵活比例；
  2. 结构统一：每个训练子集（train/valid/test）目录内部包含各自原图与 `_annotations.coco.json`；
  3. 连通域实例提取：通过 cv2.connectedComponentsWithStats 提取独立实例并过滤小噪点；
  4. 负样本保留：无目标的背景图/NG 图登记在 images 列表，不生成 annotation，增强抗误报训练能力；
  5. 随机可视化抽检：随机抽取 30 张样本生成“左侧原图 + 右侧实例分割 Mask/BBox/标签”对比图，直观核验转换质量。
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# 保证 Windows 命令行控制台输出 utf-8 编码兼容
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 默认常量配置
# ---------------------------------------------------------------------------
DEFAULT_CLASSES = [
    {
        "id": 1,
        "name": "FPC",
        "supercategory": "none",
    }
]
CATEGORY_ID = 1
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_split_ratios(split_arg: str) -> Tuple[float, float, float]:
    """解析如 '7:2:1'、'8:1:1' 或 '0.7:0.2:0.1' 形式的比例字符串，并自动归一化。"""
    normalized = split_arg.replace("/", ":").replace(",", ":").replace(" ", ":")
    parts = [float(x.strip()) for x in normalized.split(":") if x.strip()]
    if len(parts) == 2:
        parts.append(0.0)  # 仅 train 和 test
    if len(parts) != 3:
        raise ValueError(
            f"划分比例参数格式错误: '{split_arg}'，期望格式如 '7:2:1' 或 '8:1:1'"
        )
    total = sum(parts)
    if total <= 0:
        raise ValueError("划分比例各分量之和必须大于 0！")
    return (parts[0] / total, parts[1] / total, parts[2] / total)


def simplify_contour(cnt: np.ndarray, *, epsilon_ratio: float = 0.002) -> np.ndarray:
    """对轮廓进行微量多边形逼近，既保留精细边界又去除冗余共线点。"""
    peri = cv2.arcLength(cnt, True)
    eps = max(epsilon_ratio * peri, 1.0)
    return cv2.approxPolyDP(cnt, eps, True)


def extract_instances_from_mask(
    mask: np.ndarray,
    min_area: int = 16,
) -> List[Dict[str, Any]]:
    """从语义分割 Mask 中按连通域提取实例的多边形与边界框。

    参数:
        mask: 单通道图像矩阵 (H, W)，非 0 像素代表前景
        min_area: 连通域最小面积（像素数），低于该阈值的视为噪点忽略

    返回:
        instances 列表，每个元素包含:
            - segmentation: [[x1, y1, x2, y2, ...]]
            - bbox: [x, y, w, h]
            - area: float
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # 二值化前景
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    instances: List[Dict[str, Any]] = []

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        comp_mask = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        # 取当前连通域最大外轮廓
        cnt = max(contours, key=cv2.contourArea)
        cnt = simplify_contour(cnt)
        if cnt.shape[0] < 3:
            continue

        pts = cnt.squeeze(1)
        if pts.ndim != 2 or len(pts) < 3:
            continue

        # 计算外接矩形 [x, y, w, h]
        x, y, w, h = cv2.boundingRect(cnt)
        poly = pts.flatten().astype(float).tolist()
        if len(poly) < 6:
            continue

        # COCO area: 使用轮廓几何面积
        c_area = float(cv2.contourArea(cnt))
        if c_area < 1.0:
            c_area = float(area)

        instances.append(
            {
                "segmentation": [poly],
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": c_area,
            }
        )

    return instances


def collect_dataset_pairs(
    src_dir: Path,
) -> List[Tuple[str, Path, Path]]:
    """扫描原数据集目录并校验图像与标注对齐。

    返回:
        列表，每个元素为 (stem, img_path, mask_path)
    """
    img_dir = src_dir / "img_dir"
    ann_dir = src_dir / "ann_dir"

    if not img_dir.exists() or not ann_dir.exists():
        raise FileNotFoundError(
            f"源数据集中找不到 img_dir 或 ann_dir: {src_dir}"
        )

    # 递归查找所有图像文件
    img_files: Dict[str, Path] = {}
    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            img_files[p.stem] = p

    ann_files: Dict[str, Path] = {}
    for p in ann_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            ann_files[p.stem] = p

    # 校验配对
    matched_stems = sorted(list(set(img_files.keys()) & set(ann_files.keys())))
    missing_ann = set(img_files.keys()) - set(ann_files.keys())
    missing_img = set(ann_files.keys()) - set(img_files.keys())

    if missing_ann:
        print(f"[WARN] 发现 {len(missing_ann)} 张图像缺少对应 Mask，将被忽略！")
    if missing_img:
        print(f"[WARN] 发现 {len(missing_img)} 个 Mask 缺少对应原图，将被忽略！")

    pairs = [(stem, img_files[stem], ann_files[stem]) for stem in matched_stems]
    return pairs


def split_dataset(
    pairs: List[Tuple[str, Path, Path]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Tuple[str, Path, Path]]]:
    """全局打乱并按指定比例划分 train / valid / test。"""
    total = len(pairs)
    indices = list(range(total))
    rnd = random.Random(seed)
    rnd.shuffle(indices)

    num_train = int(round(total * train_ratio))
    num_val = int(round(total * val_ratio))
    # 剩余的全部划入 test，保证总和完全等于 total
    num_test = total - num_train - num_val
    if num_test < 0:
        num_test = 0
        num_train = total - num_val

    train_idx = indices[:num_train]
    val_idx = indices[num_train : num_train + num_val]
    test_idx = indices[num_train + num_val :]

    splits = {
        "train": [pairs[i] for i in train_idx],
        "valid": [pairs[i] for i in val_idx],
        "test": [pairs[i] for i in test_idx],
    }
    return splits


def convert_and_export(
    src_dir: Path,
    out_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    min_area: int = 16,
    seed: int = 42,
) -> Dict[str, Any]:
    """执行完整的转换与导出流程，并将 _annotations.coco.json 放入各个划分目录。"""
    print("=" * 70)
    print("开始执行 FPC_QK_140 语义分割 -> COCO 实例分割转换")
    print(f"源数据集目录: {src_dir}")
    print(f"目标输出目录: {out_dir}")
    print(
        f"划分比例: train={train_ratio:.3f}, valid={val_ratio:.3f}, test={test_ratio:.3f} (随机种子={seed})"
    )
    print(f"连通域面积过滤阈值: min_area >= {min_area}")
    print("=" * 70)

    # 1. 扫描与配对
    pairs = collect_dataset_pairs(src_dir)
    print(f"[OK] 成功找到并配对样本对: {len(pairs)} 组")

    # 2. 划分数据集
    splits = split_dataset(
        pairs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    for split_name, split_pairs in splits.items():
        print(f"  - {split_name} 子集: {len(split_pairs)} 张图片")

    # 清理可能存在的旧结构残留，避免混乱
    old_images_dir = out_dir / "images"
    old_ann_dir = out_dir / "annotations"
    if old_images_dir.exists():
        shutil.rmtree(old_images_dir, ignore_errors=True)
    if old_ann_dir.exists():
        shutil.rmtree(old_ann_dir, ignore_errors=True)

    summary_stats = {}
    global_ann_id = 1

    # 3. 逐个子集转换（图片与 _annotations.coco.json 统一保存在各子集目录下）
    for split_name in ["train", "valid", "test"]:
        split_pairs = splits[split_name]
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        coco_data: Dict[str, Any] = {
            "info": {
                "description": f"FPC QK 140 Instance Segmentation Dataset ({split_name})",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "Lightly-Train",
                "date_created": datetime.date.today().isoformat(),
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": DEFAULT_CLASSES,
        }

        image_id = 1
        instance_count = 0
        empty_image_count = 0

        print(f"\n正在处理 {split_name} 子集...")
        for stem, img_path, mask_path in tqdm(
            split_pairs, desc=f"Converting {split_name}"
        ):
            # 读取图像尺寸
            with Image.open(img_path) as im:
                width, height = im.size

            # 拷贝原图到目标子目录下: <split_dir>/<file_name>
            dest_img_path = split_dir / img_path.name
            shutil.copy2(img_path, dest_img_path)

            # 登记 image 记录（file_name 保持为相对文件名）
            image_record = {
                "id": image_id,
                "file_name": img_path.name,
                "width": int(width),
                "height": int(height),
            }
            coco_data["images"].append(image_record)

            # 读取 Mask 提取实例
            mask_mat = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask_mat is None:
                print(f"[WARN] 读取 Mask 失败: {mask_path}，按空标注处理")
                instances = []
            else:
                instances = extract_instances_from_mask(mask_mat, min_area=min_area)

            if not instances:
                empty_image_count += 1
            else:
                for inst in instances:
                    ann_record = {
                        "id": global_ann_id,
                        "image_id": image_id,
                        "category_id": CATEGORY_ID,
                        "segmentation": inst["segmentation"],
                        "area": inst["area"],
                        "bbox": inst["bbox"],
                        "iscrowd": 0,
                    }
                    coco_data["annotations"].append(ann_record)
                    global_ann_id += 1
                    instance_count += 1

            image_id += 1

        # 保存为各子目录内部的标准命名: <split_dir>/_annotations.coco.json
        json_path = split_dir / "_annotations.coco.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)

        summary_stats[split_name] = {
            "image_count": len(split_pairs),
            "instance_count": instance_count,
            "empty_image_count": empty_image_count,
            "avg_instances_per_image": (
                round(instance_count / len(split_pairs), 2) if split_pairs else 0
            ),
            "json_path": str(json_path),
        }

    return summary_stats


def generate_visualizations(
    out_dir: Path,
    num_samples: int = 30,
    seed: int = 42,
) -> Path:
    """从生成的数据集中随机抽取 30 张图片，绘制左右对比可视化图并保存。

    左侧：原图
    右侧：叠加半透明 Mask + 轮廓线 + BBox + 类别标签
    """
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 聚合所有子集的 (image_path, annotations)
    candidate_records: List[Tuple[Path, List[Dict[str, Any]]]] = []

    for split_name in ["train", "valid", "test"]:
        split_dir = out_dir / split_name
        json_path = split_dir / "_annotations.coco.json"
        if not json_path.exists():
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        img_id_to_file = {
            img["id"]: img["file_name"] for img in coco_data.get("images", [])
        }
        img_id_to_anns = defaultdict(list)
        for ann in coco_data.get("annotations", []):
            img_id_to_anns[ann["image_id"]].append(ann)

        for img_id, file_name in img_id_to_file.items():
            img_path = split_dir / file_name
            if img_path.exists():
                candidate_records.append((img_path, img_id_to_anns[img_id]))

    if not candidate_records:
        print("[WARN] 未能找到任何图片用于生成可视化！")
        return vis_dir

    rnd = random.Random(seed)
    # 优先抽取含实例的图片，同时也抽取少量背景负样本
    with_inst = [r for r in candidate_records if len(r[1]) > 0]
    without_inst = [r for r in candidate_records if len(r[1]) == 0]

    rnd.shuffle(with_inst)
    rnd.shuffle(without_inst)

    # 选取样本（例如 26 张正样本 + 4 张负样本）
    num_pos = min(len(with_inst), max(1, num_samples - len(without_inst[:4])))
    num_neg = min(len(without_inst), num_samples - num_pos)
    selected = with_inst[:num_pos] + without_inst[:num_neg]
    rnd.shuffle(selected)
    selected = selected[:num_samples]

    print(f"\n正在生成 {len(selected)} 张随机对比抽检可视化图片...")
    for idx, (img_path, anns) in enumerate(
        tqdm(selected, desc="Generating Visualizations")
    ):
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            continue

        h, w = raw_bgr.shape[:2]

        # 右侧图：在原图上叠加实例标注
        vis_overlay = raw_bgr.copy()
        mask_overlay = np.zeros_like(raw_bgr, dtype=np.uint8)

        # 调配高对比度鲜明颜色（如工业视觉常用的荧光绿与亮青）
        color_fill = (0, 255, 128)  # BGR 荧光绿
        color_contour = (0, 200, 255)  # 边框金色/青橙
        color_bbox = (255, 140, 0)  # BBox 亮蓝/橙

        for ann in anns:
            seg = ann.get("segmentation", [])
            for poly in seg:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                # 填充 mask
                cv2.fillPoly(mask_overlay, [pts], color_fill)
                # 绘制精细轮廓边线
                cv2.polylines(vis_overlay, [pts], True, color_contour, 2)

            # 绘制 BBox
            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                bx, by, bw, bh = [int(v) for v in bbox]
                cv2.rectangle(
                    vis_overlay, (bx, by), (bx + bw, by + bh), color_bbox, 2
                )
                # 类别文字标签徽章
                label_text = f"FPC (area={int(ann.get('area', 0))})"
                (tw, th), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
                )
                label_y = max(by, th + 4)
                cv2.rectangle(
                    vis_overlay,
                    (bx, label_y - th - 4),
                    (bx + tw + 4, label_y + baseline),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    vis_overlay,
                    label_text,
                    (bx + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 透明度融合: 70% 原图 + 30% Mask 填充
        mask_bool = np.any(mask_overlay > 0, axis=-1)
        vis_overlay[mask_bool] = cv2.addWeighted(
            vis_overlay, 0.65, mask_overlay, 0.35, 0
        )[mask_bool]

        # 构建顶部文字标识栏
        header_h = 36
        banner_left = np.zeros((header_h, w, 3), dtype=np.uint8)
        banner_right = np.zeros((header_h, w, 3), dtype=np.uint8)

        cv2.putText(
            banner_left,
            f"[ORIGINAL] {img_path.name}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        status_text = (
            f"[COCO InSeg] {len(anns)} instance(s)"
            if anns
            else "[COCO InSeg] Background (0 instance)"
        )
        cv2.putText(
            banner_right,
            status_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 128) if anns else (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        left_composed = np.vstack([banner_left, raw_bgr])
        right_composed = np.vstack([banner_right, vis_overlay])

        # 左右拼接
        divider = np.full((h + header_h, 4, 3), 60, dtype=np.uint8)
        combined = np.hstack([left_composed, divider, right_composed])

        # 保存可视化
        out_vis_file = vis_dir / f"{idx + 1:02d}_{img_path.stem}_vis.png"
        cv2.imwrite(str(out_vis_file), combined)

    print(f"[OK] 30 张抽检对比可视化图片已保存至: {vis_dir}")
    return vis_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert FPC_QK_140 masks to COCO instance segmentation with configurable split ratios."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=next((p for p in Path(__file__).resolve().parents if p.name == "tools"), Path(__file__).resolve().parents[1]) / "DataSet" / "FPC_QK_140",
        help="Path to source FPC_QK_140 dataset",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=next((p for p in Path(__file__).resolve().parents if p.name == "tools"), Path(__file__).resolve().parents[1]) / "DataSet" / "FPC_QK_140_InSeg_coco",
        help="Path to output COCO instance segmentation dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="7:2:1",
        help="Split ratio for train:valid:test (e.g. '7:2:1' or '8:1:1', default: '7:2:1')",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=16,
        help="Min connected component area threshold (default: 16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset split (default: 42)",
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=30,
        help="Number of random visualization images to generate (default: 30)",
    )

    args = parser.parse_args()

    train_r, val_r, test_r = parse_split_ratios(args.split)

    stats = convert_and_export(
        src_dir=args.src_dir,
        out_dir=args.out_dir,
        train_ratio=train_r,
        val_ratio=val_r,
        test_ratio=test_r,
        min_area=args.min_area,
        seed=args.seed,
    )

    # 生成 30 张抽检可视化图片
    vis_dir = generate_visualizations(
        out_dir=args.out_dir,
        num_samples=args.num_vis,
        seed=args.seed,
    )

    print("\n" + "=" * 70)
    print("[SUCCESS] 转换与可视化生成全部完成！统计汇总报表如下：")
    print("=" * 70)
    for split_name, s in stats.items():
        print(f"[{split_name.upper()}]")
        print(f"  - 图像总数: {s['image_count']} 张")
        print(f"  - 实例总数: {s['instance_count']} 个")
        print(f"  - 纯背景负样本图: {s['empty_image_count']} 张")
        print(f"  - 平均实例数/图: {s['avg_instances_per_image']}")
        print(f"  - COCO JSON 路径: {s['json_path']}")
    print(f"\n[可视化目录] {vis_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
