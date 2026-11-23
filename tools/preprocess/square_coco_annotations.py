# -*- coding: utf-8 -*-
"""
COCO 标注框正方形化处理脚本 (Square COCO Annotations - 四向卡尺与多截少补版)

功能说明：
1. 读取 COCO 实例分割数据集（train/valid/test）。
2. 四向自适应工业卡尺算法 (4-Direction Caliper Algorithm)：
   - 自动检测 FPC 引脚朝向（头朝左 / 头朝右 / 头朝上 / 头朝下）：
     * 头朝左 (Left) : 平行边为上下，基准边 = 垂直高度 H；以左端头部锚定向右延伸 [H x H]
     * 头朝右 (Right): 平行边为上下，基准边 = 垂直高度 H；以右端头部锚定向左延伸 [H x H]
     * 头朝下 (Down) : 平行边为左右，基准边 = 水平宽度 W；以底端头部锚定向上延伸 [W x W]
     * 头朝上 (Up)   : 平行边为左右，基准边 = 水平宽度 W；以顶端头部锚定向下延伸 [W x W]
3. Segmentation“多截少补”自适应对齐：
   - 保护圆角端头特征；
   - 若标注多边形沿身部长度短于正方形尺寸（少了）：自动沿两侧平行边向外延伸填充，填满正方形框；
   - 若标注多边形长于正方形尺寸（多了）：超出正方形框部分自动截断裁剪；
   - 重新计算调整后的多边形实际几何面积 area。
4. 严格保护原始数据：输出到独立的新数据集目录，自动复制图像。
5. 提供进度显示（tqdm）、详细统计汇总及抽样对比可视化（visualizations/）。

使用方法：
    python tools/square_coco_annotations.py
    python tools/square_coco_annotations.py --data-root tools/DataSet/FPC_Line5_seg_coco --out-root tools/DataSet/FPC_Line5_seg_coco_square
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 保证 Windows 命令行控制台输出 utf-8 编码兼容
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ===========================================================================
# 工业视觉数据集正方形化 - 全局参数配置区 (Global Configuration)
# 说明：在 IDE (VS Code / PyCharm) 中直接按 F5 运行即可生效；亦支持命令行传参覆盖。
# ===========================================================================
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists() or (p / ".git").exists()), Path(__file__).resolve().parents[2])

# 1. 源数据集路径（支持相对路径或绝对路径）
# 可选示例:
#   PROJECT_ROOT / "tools" / "DataSet" / "FPC_Line4_Seg_coco"
#   PROJECT_ROOT / "tools" / "DataSet" / "FPC_Line6_Seg_coco"
#   PROJECT_ROOT / "tools" / "DataSet" / "FPC_QK_140_InSeg_coco"
DATA_ROOT = PROJECT_ROOT / "tools" / "DataSet" / "L9Seg-data_coco1"

# 2. 目标输出路径
# 设置为 None 时，将自动基于 DATA_ROOT 目录名自动生成: <DATA_ROOT>_square
OUT_ROOT = None

# 3. 待处理子集列表 (默认: ["train", "valid", "test"])
SPLITS = ["train", "valid", "test"]

# 4. 每个 split 随机抽样生成的对比可视化图片数 (-1: 全部生成, 0: 不生成以加速处理, 推荐: 30)
NUM_VISUALIZE = 30

# 5. 是否将原图复制/链接到输出数据集目录 (通常为 True，保证新数据集独立自包含)
COPY_IMAGES = True

# 6. 预检模式 (True: 仅模拟计算与打印统计，不写入磁盘; False: 正常写盘产出新数据集)
DRY_RUN = False

# 7. 随机数种子 (保证抽检抽样一致可复现)
SEED = 42
# ===========================================================================


def resolve_out_root(data_root: Path, out_root: Path | None) -> Path:
    """自动推导输出根目录：若未指定则自动基于源目录命名为 <data_root>_square。"""
    if out_root is not None and str(out_root).strip():
        return Path(out_root).resolve()
    return data_root.parent / f"{data_root.name}_square"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于四向自适应卡尺测宽与多边形多截少补的 COCO 标注框正方形化处理工具。"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"源数据集根目录 (默认使用全局变量: {DATA_ROOT})",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="目标数据集输出根目录 (默认基于全局变量或自动推导: <DATA_ROOT>_square)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help=f"待处理的数据集子集 (默认使用全局变量: {' '.join(SPLITS)})",
    )
    parser.add_argument(
        "--num-visualize",
        type=int,
        default=None,
        help=f"每个 split 随机抽样对比图数量 (默认使用全局变量: {NUM_VISUALIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"随机抽样随机数种子 (默认使用全局变量: {SEED})",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        default=None,
        help="是否将原图复制到新数据集目录",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="预检模式，仅打印计划操作而不实际写入文件",
    )

    args = parser.parse_args()

    # 优先采用命令行传入值，若未传入则回退到顶部全局变量
    final_data_root = (Path(args.data_root) if args.data_root is not None else Path(DATA_ROOT)).resolve()
    final_out_root_raw = Path(args.out_root) if args.out_root is not None else (Path(OUT_ROOT) if OUT_ROOT is not None else None)
    final_out_root = resolve_out_root(final_data_root, final_out_root_raw)

    args.data_root = final_data_root
    args.out_root = final_out_root
    args.splits = args.splits if args.splits is not None else list(SPLITS)
    args.num_visualize = args.num_visualize if args.num_visualize is not None else NUM_VISUALIZE
    args.seed = args.seed if args.seed is not None else SEED
    args.copy_images = args.copy_images if args.copy_images is not None else COPY_IMAGES
    args.dry_run = args.dry_run if args.dry_run is not None else DRY_RUN

    return args


def detect_fpc_orientation_and_head(
    patch_mask: np.ndarray,
) -> str:
    """
    通过分析局部 Patch Mask 四个边缘的饱满度与角点缺角特征，
    精确识别 FPC 引脚朝向与端头位置：
    返回: "left" | "right" | "top" | "bottom" (表示圆角头部 Head 所在方向)
    """
    ph, pw = patch_mask.shape[:2]
    if ph < 4 or pw < 4:
        return "left" if pw >= ph else "bottom"

    depth_y = max(1, min(5, ph // 6))
    depth_x = max(1, min(5, pw // 6))

    top_band = patch_mask[:depth_y, :]
    bottom_band = patch_mask[-depth_y:, :]
    left_band = patch_mask[:, :depth_x]
    right_band = patch_mask[:, -depth_x:]

    fullness_top = float(np.mean(top_band > 0))
    fullness_bottom = float(np.mean(bottom_band > 0))
    fullness_left = float(np.mean(left_band > 0))
    fullness_right = float(np.mean(right_band > 0))

    corner_sz_y = max(2, min(12, ph // 4))
    corner_sz_x = max(2, min(12, pw // 4))

    tl_void = float(np.mean(patch_mask[:corner_sz_y, :corner_sz_x] == 0))
    tr_void = float(np.mean(patch_mask[:corner_sz_y, -corner_sz_x:] == 0))
    bl_void = float(np.mean(patch_mask[-corner_sz_y:, :corner_sz_x] == 0))
    br_void = float(np.mean(patch_mask[-corner_sz_y:, -corner_sz_x:] == 0))

    left_head_score = (tl_void + bl_void) - (tr_void + br_void) + (fullness_right - fullness_left)
    right_head_score = (tr_void + br_void) - (tl_void + bl_void) + (fullness_left - fullness_right)
    bottom_head_score = (bl_void + br_void) - (tl_void + tr_void) + (fullness_top - fullness_bottom)
    top_head_score = (tl_void + tr_void) - (bl_void + br_void) + (fullness_top - fullness_bottom)

    is_vertical_cut = max(fullness_top, fullness_bottom) > max(fullness_left, fullness_right) + 0.15
    is_horizontal_cut = max(fullness_left, fullness_right) > max(fullness_top, fullness_bottom) + 0.15

    if is_vertical_cut:
        return "bottom" if bottom_head_score >= top_head_score else "top"
    elif is_horizontal_cut:
        return "left" if left_head_score >= right_head_score else "right"
    else:
        scores = {
            "bottom": bottom_head_score,
            "left": left_head_score,
            "top": top_head_score,
            "right": right_head_score,
        }
        return max(scores, key=scores.get)


def measure_caliper_width_and_box(
    segmentation: List[List[float]],
    orig_bbox: List[float],
    img_width: int,
    img_height: int,
) -> Tuple[float, float, float, float, str]:
    """
    四向自适应工业卡尺测宽与头部基准定位。
    """
    bx, by, bw, bh = orig_bbox

    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for poly in segmentation:
        if len(poly) >= 6:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)

    x1 = max(0, int(np.floor(bx)))
    y1 = max(0, int(np.floor(by)))
    x2 = min(img_width, int(np.ceil(bx + bw)))
    y2 = min(img_height, int(np.ceil(by + bh)))

    patch_mask = mask[y1:y2, x1:x2]
    head_dir = detect_fpc_orientation_and_head(patch_mask)

    if head_dir == "left":
        target_size = float(bh)
        new_x = float(bx)
        new_y = float(by)
        new_w = target_size
        new_h = target_size

    elif head_dir == "right":
        target_size = float(bh)
        new_x = float(bx + bw - target_size)
        new_y = float(by)
        new_w = target_size
        new_h = target_size

    elif head_dir == "bottom":
        target_size = float(bw)
        new_x = float(bx)
        new_y = float(by + bh - target_size)
        new_w = target_size
        new_h = target_size

    else:  # "top"
        target_size = float(bw)
        new_x = float(bx)
        new_y = float(by)
        new_w = target_size
        new_h = target_size

    if new_x < 0:
        new_x = 0.0
    if new_y < 0:
        new_y = 0.0
    if new_x + new_w > img_width:
        new_x = max(0.0, float(img_width - new_w))
    if new_y + new_h > img_height:
        new_y = max(0.0, float(img_height - new_h))

    return round(new_x, 2), round(new_y, 2), round(new_w, 2), round(new_h, 2), head_dir


def adjust_segmentation_polygon(
    segmentation: List[List[float]],
    orig_bbox: List[float],
    new_bbox: Tuple[float, float, float, float],
    head_dir: str,
    img_width: int,
    img_height: int,
) -> Tuple[List[List[float]], float]:
    """
    Segmentation 多截少补核心处理：
    1. 保持圆角端头原貌不变；
    2. 如果沿身部方向长度不足正方形（少了）：在原平切端与新正方形边框之间补全填充矩形；
    3. 如果超出正方形（多了）：与新正方形框求交截断；
    4. 提取更新后的闭合轮廓与面积。
    """
    bx, by, bw, bh = orig_bbox
    nx, ny, nw, nh = new_bbox

    # 1. 创建原掩膜
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for poly in segmentation:
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)

    # 2. “少了填充”：若新框在引脚身部方向超出原标注，将空隙矩形补全
    if head_dir == "bottom":
        # 身部向顶端延伸：若 ny < by，在 [bx, ny, bx+bw, by] 补全
        fill_y1 = max(0, int(np.floor(ny)))
        fill_y2 = max(0, int(np.ceil(by)))
        fill_x1 = max(0, int(np.floor(bx)))
        fill_x2 = min(img_width, int(np.ceil(bx + bw)))
        if fill_y2 > fill_y1 and fill_x2 > fill_x1:
            mask[fill_y1:fill_y2, fill_x1:fill_x2] = 255

    elif head_dir == "top":
        # 身部向底端延伸：若 ny + nh > by + bh，在 [bx, by+bh, bx+bw, ny+nh] 补全
        fill_y1 = max(0, int(np.floor(by + bh)))
        fill_y2 = min(img_height, int(np.ceil(ny + nh)))
        fill_x1 = max(0, int(np.floor(bx)))
        fill_x2 = min(img_width, int(np.ceil(bx + bw)))
        if fill_y2 > fill_y1 and fill_x2 > fill_x1:
            mask[fill_y1:fill_y2, fill_x1:fill_x2] = 255

    elif head_dir == "left":
        # 身部向右端延伸：若 nx + nw > bx + bw，在 [bx+bw, by, nx+nw, by+bh] 补全
        fill_x1 = max(0, int(np.floor(bx + bw)))
        fill_x2 = min(img_width, int(np.ceil(nx + nw)))
        fill_y1 = max(0, int(np.floor(by)))
        fill_y2 = min(img_height, int(np.ceil(by + bh)))
        if fill_x2 > fill_x1 and fill_y2 > fill_y1:
            mask[fill_y1:fill_y2, fill_x1:fill_x2] = 255

    elif head_dir == "right":
        # 身部向左端延伸：若 nx < bx，在 [nx, by, bx, by+bh] 补全
        fill_x1 = max(0, int(np.floor(nx)))
        fill_x2 = max(0, int(np.ceil(bx)))
        fill_y1 = max(0, int(np.floor(by)))
        fill_y2 = min(img_height, int(np.ceil(by + bh)))
        if fill_x2 > fill_x1 and fill_y2 > fill_y1:
            mask[fill_y1:fill_y2, fill_x1:fill_x2] = 255

    # 3. “多了截断”：使用正方形矩形窗口求交截断超出部分
    rect_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    rx1 = max(0, int(np.floor(nx)))
    ry1 = max(0, int(np.floor(ny)))
    rx2 = min(img_width, int(np.ceil(nx + nw)))
    ry2 = min(img_height, int(np.ceil(ny + nh)))

    if rx2 > rx1 and ry2 > ry1:
        rect_mask[ry1:ry2, rx1:rx2] = 255

    final_mask = cv2.bitwise_and(mask, rect_mask)

    # 4. 提取更新后的多边形轮廓
    contours, _ = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    new_polys: List[List[float]] = []
    total_area = 0.0

    for cnt in contours:
        if len(cnt) < 3:
            continue
        area = float(cv2.contourArea(cnt))
        if area <= 1.0:
            continue

        total_area += area
        flat_pts = cnt.reshape(-1, 2).astype(np.float32).flatten().tolist()
        new_polys.append([round(float(v), 2) for v in flat_pts])

    # 回退保护
    if not new_polys and (rx2 > rx1 and ry2 > ry1):
        fallback_poly = [
            float(rx1), float(ry1),
            float(rx2), float(ry1),
            float(rx2), float(ry2),
            float(rx1), float(ry2),
        ]
        new_polys = [fallback_poly]
        total_area = float((rx2 - rx1) * (ry2 - ry1))

    return new_polys, round(total_area, 2)


def render_comparison(
    img_path: Path,
    orig_anns: List[Dict[str, Any]],
    new_anns: List[Dict[str, Any]],
    head_dirs: List[str],
    save_path: Path,
) -> None:
    """
    渲染处理前后的对比图：
    左图：原图 + 原始 BBox (黄色) + 原始 Mask (绿色半透明)
    右图：原图 + 卡尺正方形 BBox (青色) + 多截少补 Mask (洋红色半透明)
    """
    raw_bytes = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return

    # 左图：原始
    left_img = img.copy()
    left_overlay = left_img.copy()
    for ann in orig_anns:
        bbox = ann.get("bbox", [])
        if len(bbox) == 4:
            bx, by, bw, bh = [int(v) for v in bbox]
            cv2.rectangle(left_img, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            cv2.putText(
                left_img,
                f"Orig: {bw}x{bh}",
                (bx, max(15, by - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        for poly in ann.get("segmentation", []):
            if len(poly) >= 6:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(left_overlay, [pts], (0, 220, 0))
                cv2.polylines(left_img, [pts], True, (0, 255, 0), 1)

    left_result = cv2.addWeighted(left_overlay, 0.4, left_img, 0.6, 0)
    cv2.putText(
        left_result, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )

    # 右图：处理后
    right_img = img.copy()
    right_overlay = right_img.copy()
    for idx, ann in enumerate(new_anns):
        bbox = ann.get("bbox", [])
        h_dir = head_dirs[idx] if idx < len(head_dirs) else "auto"
        if len(bbox) == 4:
            bx, by, bw, bh = [int(v) for v in bbox]
            cv2.rectangle(right_img, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)
            cv2.putText(
                right_img,
                f"Caliper ({h_dir}): {bw}x{bh}",
                (bx, max(15, by - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        for poly in ann.get("segmentation", []):
            if len(poly) >= 6:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(right_overlay, [pts], (255, 0, 255))
                cv2.polylines(right_img, [pts], True, (255, 100, 255), 1)

    right_result = cv2.addWeighted(right_overlay, 0.4, right_img, 0.6, 0)
    cv2.putText(
        right_result, "Square Mask Filled", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
    )

    combined = np.hstack([left_result, right_result])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    _, buf = cv2.imencode(".png", combined)
    buf.tofile(save_path)


def process_split(
    split: str,
    data_root: Path,
    out_root: Path,
    copy_images: bool,
    num_visualize: int,
    seed: int,
    dry_run: bool,
) -> Dict[str, Any]:
    """
    处理单个 split 数据集（train / valid / test）。
    """
    in_dir = data_root / split
    out_dir = out_root / split
    json_path = in_dir / "_annotations.coco.json"

    if not json_path.exists():
        print(f"[-] 跳过子集 {split}: 未找到标注文件 {json_path}")
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])
    info = coco_data.get("info", {})
    licenses = coco_data.get("licenses", [])

    img_id_to_img = {img["id"]: img for img in images}
    img_id_to_orig_anns: Dict[int, List[Dict[str, Any]]] = {}
    for ann in annotations:
        img_id_to_orig_anns.setdefault(ann["image_id"], []).append(ann)

    print(f"\n==================================================")
    print(f"正在处理子集: {split}")
    print(f"图像数量: {len(images)}, 标注实例数量: {len(annotations)}")
    print(f"算法: 四向自适应卡尺测宽 + Segmentation 多截少补")
    print(f"==================================================")

    new_annotations: List[Dict[str, Any]] = []
    ann_id_to_head_dir: Dict[int, str] = {}
    dir_counter = {"left": 0, "right": 0, "top": 0, "bottom": 0}

    stats = {
        "split": split,
        "num_images": len(images),
        "num_anns": len(annotations),
        "orig_w_list": [],
        "orig_h_list": [],
        "new_size_list": [],
        "orig_area_list": [],
        "new_area_list": [],
    }

    iterator = tqdm(annotations, desc=f"处理标注 [{split}]") if tqdm else annotations

    for ann in iterator:
        image_id = ann["image_id"]
        img_meta = img_id_to_img.get(image_id)
        if not img_meta:
            continue

        img_w = img_meta["width"]
        img_h = img_meta["height"]

        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            continue

        x, y, w, h = bbox
        stats["orig_w_list"].append(w)
        stats["orig_h_list"].append(h)
        stats["orig_area_list"].append(ann.get("area", w * h))

        orig_seg = ann.get("segmentation", [])

        # 1. 四向自适应卡尺测宽与头部基准定位
        new_x, new_y, new_w, new_h, h_dir = measure_caliper_width_and_box(
            segmentation=orig_seg,
            orig_bbox=bbox,
            img_width=img_w,
            img_height=img_h,
        )

        dir_counter[h_dir] = dir_counter.get(h_dir, 0) + 1
        ann_id_to_head_dir[ann["id"]] = h_dir

        new_bbox_tuple = (new_x, new_y, new_w, new_h)

        # 2. 多边形多截少补自适应调整
        adjusted_seg, new_area = adjust_segmentation_polygon(
            segmentation=orig_seg,
            orig_bbox=bbox,
            new_bbox=new_bbox_tuple,
            head_dir=h_dir,
            img_width=img_w,
            img_height=img_h,
        )

        stats["new_size_list"].append(new_w)
        stats["new_area_list"].append(new_area)

        new_ann = dict(ann)
        new_ann["bbox"] = [new_x, new_y, new_w, new_h]
        new_ann["segmentation"] = adjusted_seg
        new_ann["area"] = new_area
        new_annotations.append(new_ann)

    print(f"朝向统计 [{split}]: 左(Left)={dir_counter.get('left',0)}, 底(Bottom)={dir_counter.get('bottom',0)}, 右(Right)={dir_counter.get('right',0)}, 顶(Top)={dir_counter.get('top',0)}")

    if dry_run:
        print(f"[Dry-run] 子集 {split} 预检完成，跳过实际写入。")
        return stats

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 复制图片
    if copy_images:
        print(f"正在复制图像到目标目录: {out_dir}")
        img_iter = tqdm(images, desc=f"复制图像 [{split}]") if tqdm else images
        for img_info in img_iter:
            file_name = img_info["file_name"]
            src_img = in_dir / file_name
            dst_img = out_dir / file_name
            if src_img.exists() and not dst_img.exists():
                shutil.copy2(src_img, dst_img)

    # 2. 写入新的 COCO JSON 标注文件
    new_coco_data = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": images,
        "annotations": new_annotations,
    }
    out_json_path = out_dir / "_annotations.coco.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(new_coco_data, f, ensure_ascii=False, indent=2)
    print(f"[+] 已保存新标注文件: {out_json_path}")

    # 3. 抽样生成对比可视化（按朝向多样性分层随机抽样）
    if num_visualize != 0 and annotations:
        vis_dir = out_root / "visualizations" / split
        vis_dir.mkdir(parents=True, exist_ok=True)
        img_id_to_new_anns: Dict[int, List[Dict[str, Any]]] = {}
        for ann in new_annotations:
            img_id_to_new_anns.setdefault(ann["image_id"], []).append(ann)

        rng = random.Random(seed)
        all_candidate_ids = list(img_id_to_orig_anns.keys())

        if num_visualize == -1 or num_visualize >= len(all_candidate_ids):
            sample_img_ids = all_candidate_ids
            rng.shuffle(sample_img_ids)
        else:
            # 1. 按识别朝向对图像分池
            dir_to_img_ids: Dict[str, List[int]] = {}
            for ann in new_annotations:
                img_id = ann["image_id"]
                h_dir = ann_id_to_head_dir.get(ann["id"], "other")
                dir_to_img_ids.setdefault(h_dir, []).append(img_id)

            for d in dir_to_img_ids:
                dir_to_img_ids[d] = list(set(dir_to_img_ids[d]))
                rng.shuffle(dir_to_img_ids[d])

            # 2. 轮询抽取，确保上下左右各方向均匀覆盖
            sample_img_ids: List[int] = []
            selected_set = set()
            while len(sample_img_ids) < num_visualize and any(dir_to_img_ids.values()):
                for d in list(dir_to_img_ids.keys()):
                    while dir_to_img_ids[d]:
                        cid = dir_to_img_ids[d].pop()
                        if cid not in selected_set:
                            selected_set.add(cid)
                            sample_img_ids.append(cid)
                            break
                    if len(sample_img_ids) >= num_visualize:
                        break

            # 若特定池耗尽仍不足配额，从剩余候选集随机补充
            if len(sample_img_ids) < num_visualize:
                remaining = [cid for cid in all_candidate_ids if cid not in selected_set]
                rng.shuffle(remaining)
                sample_img_ids.extend(remaining[: (num_visualize - len(sample_img_ids))])

        print(f"正在随机生成 {len(sample_img_ids)} 张抽样对比可视化图至: {vis_dir}")
        for img_id in sample_img_ids:
            img_meta = img_id_to_img.get(img_id)
            if not img_meta:
                continue
            f_name = img_meta["file_name"]
            src_file = in_dir / f_name
            out_vis_file = vis_dir / f"compare_{f_name}"
            anns_orig = img_id_to_orig_anns.get(img_id, [])
            anns_new = img_id_to_new_anns.get(img_id, [])
            head_list = [ann_id_to_head_dir.get(a["id"], "auto") for a in anns_new]
            render_comparison(
                src_file,
                anns_orig,
                anns_new,
                head_list,
                out_vis_file,
            )
        print(f"[+] 随机抽样对比图生成完成！")

    return stats


def print_summary(all_stats: List[Dict[str, Any]]) -> None:
    """打印处理后的汇总统计指标。"""
    print("\n" + "=" * 65)
    print("               数据处理汇总报表 (Summary Report)             ")
    print("=" * 65)
    print(f"{'Split':<8} | {'图片数':<6} | {'标注数':<6} | {'原平均尺寸(W×H)':<16} | {'正方形边长均值':<12} | {'面积比率':<10}")
    print("-" * 65)

    for st in all_stats:
        if not st:
            continue
        split = st["split"]
        n_img = st["num_images"]
        n_ann = st["num_anns"]
        if n_ann > 0:
            avg_ow = np.mean(st["orig_w_list"])
            avg_oh = np.mean(st["orig_h_list"])
            avg_nsize = np.mean(st["new_size_list"])
            orig_tot_area = sum(st["orig_area_list"])
            new_tot_area = sum(st["new_area_list"])
            ratio = (new_tot_area / orig_tot_area * 100) if orig_tot_area > 0 else 100.0
            print(
                f"{split:<8} | {n_img:<6} | {n_ann:<6} | {avg_ow:.1f}×{avg_oh:.1f}{'px':<8} | {avg_nsize:.1f}×{avg_nsize:.1f}{'px':<4} | {ratio:.1f}%"
            )
        else:
            print(f"{split:<8} | {n_img:<6} | {n_ann:<6} | N/A              | N/A          | N/A")

    print("=" * 65)


def main() -> None:
    args = parse_args()
    print(">>> 启动 COCO 标注框四向卡尺正方形化与多截少补工具")
    print(f"输入路径: {args.data_root}")
    print(f"输出路径: {args.out_root}")
    print(f"处理子集: {args.splits}")
    print(f"模式: 四向卡尺测宽 + Segmentation 多截少补 (Fill & Trim)")
    print(f"抽样可视化数量: {args.num_visualize}")
    print(f"预检模式: {args.dry_run}")

    if not args.data_root.exists():
        dataset_parent = args.data_root.parent
        available = [p.name for p in dataset_parent.iterdir() if p.is_dir()] if dataset_parent.is_dir() else []
        print(f"\n[错误] 数据集目录不存在: {args.data_root}")
        if available:
            print(f"当前目录下现存的数据集包含:\n  - " + "\n  - ".join(available[:10]))
            print("请检查并修改脚本顶部全局变量 DATA_ROOT 或通过命令行 --data-root 传入正确路径。")
        raise FileNotFoundError(f"输入数据集根目录不存在: {args.data_root}")

    all_stats: List[Dict[str, Any]] = []
    for split in args.splits:
        stats = process_split(
            split=split,
            data_root=args.data_root,
            out_root=args.out_root,
            copy_images=args.copy_images,
            num_visualize=args.num_visualize,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        if stats:
            all_stats.append(stats)

    print_summary(all_stats)
    print("\n[✔] 四向卡尺正方形化与多截少补数据处理全部完成！")


if __name__ == "__main__":
    main()
