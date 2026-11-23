# -*- coding: utf-8 -*-
"""
labeljson 数据格式转实例分割训练集脚本 (LabelJSON to Instance Segmentation Dataset Converter)

功能特性：
1. 解析 RL02-data 及同类标注格式中的 images 和 labels (labeljson)；
2. 提取 shapes 中的 points 多边形轮廓，支持多类别动态统计或指定类别映射；
3. 计算精准多边形面积 (Shoelace 算法) 与 COCO bbox [xmin, ymin, width, height]；
4. 支持按比例划分训练集 (train) 与验证集 (val / valid)，可设置随机种子复现；
5. 保护原始数据：严禁就地覆盖，输出到全新独立数据集目录；
6. 格式支持：
   - coco (默认): train/ 和 valid/ 目录下包含图片及 _annotations.coco.json (与 train_l8_fpc_inst.py 对齐)；
   - yolo: images/{train,val} 与 labels/{train,val} 归一化多边形 txt；
   - both: 同时生成 COCO 与 YOLO 格式；
7. 具备全面的异常处理、多边形合法性过滤与 tqdm 进度显示；
8. 质检可视化：转换的同时在 visualizations/{train,valid} 自动生成左右双屏高对比度质检对比图，
   支持 --num-visualize 抽样数量与 --vis-max-size 分辨率自适应控制。

使用示例：
    # 默认转换并生成 30 张质检对比图
    python tools/convert_labeljson_to_inseg.py

    # 自定义输入输出路径、划分比例与全量生成可视化数据 (-1)
    python tools/convert_labeljson_to_inseg.py --input-dir tools/DataSet/RL02-data --output-dir tools/DataSet/RL02_Seg_coco --num-visualize -1

    # 仅转换数据，不生成可视化对比图
    python tools/convert_labeljson_to_inseg.py --num-visualize 0
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

try:
    from tqdm import tqdm
except ImportError:
    # 降级备用伪进度条
    def tqdm(iterable, desc="", **kwargs):
        print(f"[处理中] {desc}...")
        return iterable


# 默认基础路径解析
SCRIPT_DIR = next((p for p in Path(__file__).resolve().parents if p.name == "tools"), Path(__file__).resolve().parents[1])
DEFAULT_INPUT_DIR = SCRIPT_DIR / "data" / "NP_BZ-data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "NP_BZ_coco"
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
BACKGROUND_GROUP_KEY = "__background__"


def read_image_size(image_path: Path) -> Optional[Tuple[int, int]]:
    """安全读取图片真实宽高 (width, height)，兼容 Windows 中文路径。

    优先使用 PIL 仅解析图像文件头（避免解码大尺寸工业图），失败时降级使用 OpenCV。
    """
    if PILImage is not None:
        try:
            with PILImage.open(image_path) as img:
                w, h = img.size
                if w > 0 and h > 0:
                    return int(w), int(h)
        except Exception:
            pass

    try:
        raw_bytes = np.fromfile(str(image_path), dtype=np.uint8)
        img_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
        if img_bgr is not None and img_bgr.ndim >= 2:
            h, w = img_bgr.shape[:2]
            if w > 0 and h > 0:
                return int(w), int(h)
    except Exception:
        pass

    return None


def calculate_polygon_area(points: List[List[float]]) -> float:
    """使用鞋带公式 (Shoelace Formula) 计算多边形真实几何面积。"""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += (x1 * y2) - (x2 * y1)
    return float(abs(area) * 0.5)


def calculate_bbox(points: List[List[float]]) -> List[float]:
    """计算 COCO 格式边界框 [xmin, ymin, width, height]。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return [float(xmin), float(ymin), float(max(0.0, xmax - xmin)), float(max(0.0, ymax - ymin))]


def clean_polygon_points(
    raw_points: Any,
    img_width: int,
    img_height: int,
    min_points: int = 3,
) -> Optional[List[List[float]]]:
    """过滤并规范化多边形点集，去除连续重复点与越界点。"""
    if not isinstance(raw_points, (list, tuple)) or len(raw_points) < min_points:
        return None

    cleaned: List[List[float]] = []
    for pt in raw_points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            x = float(pt[0])
            y = float(pt[1])
        except (ValueError, TypeError):
            continue

        # 边界保护与截断 (工业相机偶尔出现微小越界)
        x = max(0.0, min(float(img_width), x))
        y = max(0.0, min(float(img_height), y))

        # 去除连续完全重合的点
        if not cleaned or (abs(cleaned[-1][0] - x) > 1e-4 or abs(cleaned[-1][1] - y) > 1e-4):
            cleaned.append([x, y])

    if len(cleaned) < min_points:
        return None

    return cleaned


def get_class_color(category_id: int) -> Tuple[int, int, int]:
    """根据类别 ID 生成鲜明区分的高对比度 BGR 颜色。

    内置经典工业视觉鲜艳高对比调色板，超出时使用黄金分割比在 HSV 色彩空间自动派生。
    """
    preset_palette = [
        (0, 255, 128),   # 荧光亮绿
        (0, 165, 255),   # 亮橙色
        (255, 144, 30),  # 鲜天蓝
        (0, 235, 255),   # 亮金黄
        (204, 50, 255),  # 霓虹紫
        (71, 99, 255),   # 鲜亮红
        (255, 0, 128),   # 冰青蓝
        (50, 205, 50),   # 酸橙绿
        (180, 105, 255), # 亮粉紫
        (0, 215, 255),   # 琥珀黄
    ]
    if 1 <= category_id <= len(preset_palette):
        return preset_palette[category_id - 1]

    # 超出范围时按 HSV 黄金比例均匀派生
    golden_ratio = 0.618033988749895
    h = ((category_id * golden_ratio) % 1.0) * 180.0
    s = 210 + ((category_id * 37) % 45)
    v = 225 + ((category_id * 19) % 30)
    hsv_pixel = np.uint8([[[int(h), int(s), int(v)]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))


def render_visual_comparison(
    raw_bgr: np.ndarray,
    instances: List[Dict[str, Any]],
    split_name: str,
    img_name: str,
    max_size: int = 0,
) -> np.ndarray:
    """渲染左右双屏高对比度质检对比图。

    左侧：原图
    右侧：叠加半透明掩码 (35%) + 高对比轮廓线 (2px) + 边界框 (1px) + 类别与面积标签徽章
    """
    orig_h, orig_w = raw_bgr.shape[:2]

    # 右侧图初始化
    vis_right = raw_bgr.copy()
    mask_overlay = np.zeros_like(raw_bgr, dtype=np.uint8)

    # 1. 绘制半透明 Mask 多边形
    for inst in instances:
        color = inst["color"]
        pts = np.array(inst["points"], dtype=np.int32)
        cv2.fillPoly(mask_overlay, [pts], color)

    # 35% 掩码半透明融合
    mask_bool = np.any(mask_overlay > 0, axis=-1)
    if np.any(mask_bool):
        vis_right[mask_bool] = cv2.addWeighted(
            raw_bgr, 0.65, mask_overlay, 0.35, 0
        )[mask_bool]

    # 2. 绘制轮廓线、边界框与文字标签徽章 (确保线条与文字保持 100% 不透明、清晰锐利)
    font_scale = max(0.40, min(1.0, orig_w / 2500.0))
    line_thick = max(1, int(round(font_scale * 2.0)))

    for inst in instances:
        color = inst["color"]
        pts = np.array(inst["points"], dtype=np.int32)
        bbox = inst["bbox"]
        label = inst["label"]
        area = inst.get("area", 0.0)

        # 轮廓线 (2px)
        cv2.polylines(vis_right, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # BBox 边框 [xmin, ymin, width, height]
        if len(bbox) == 4:
            bx, by, bw, bh = [int(v) for v in bbox]
            cv2.rectangle(vis_right, (bx, by), (bx + bw, by + bh), color, thickness=line_thick, lineType=cv2.LINE_AA)

            # 文字徽章
            tag_text = f"{label} (area={int(round(area))})"
            (tw, th), baseline = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            # 顶部防出界保护
            tag_y = max(by, th + 6)
            tag_x = max(0, min(bx, orig_w - tw - 8))
            # 黑色高对比背景底色
            cv2.rectangle(
                vis_right,
                (tag_x, tag_y - th - 5),
                (tag_x + tw + 6, tag_y + baseline + 2),
                (0, 0, 0),
                thickness=-1,
            )
            # 高亮彩色文字
            cv2.putText(
                vis_right,
                tag_text,
                (tag_x + 3, tag_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    # 3. 尺寸自适应缩放 (如设置 max_size > 0 则等比缩放)
    left_view = raw_bgr
    right_view = vis_right
    if max_size > 0 and max(orig_h, orig_w) > max_size:
        scale = float(max_size) / float(max(orig_h, orig_w))
        target_w = max(1, int(round(orig_w * scale)))
        target_h = max(1, int(round(orig_h * scale)))
        left_view = cv2.resize(left_view, (target_w, target_h), interpolation=cv2.INTER_AREA)
        right_view = cv2.resize(right_view, (target_w, target_h), interpolation=cv2.INTER_AREA)

    cur_h, cur_w = left_view.shape[:2]

    # 4. 构建顶部对比标识横幅 (Banner)
    banner_h = max(34, int(cur_h * 0.045))
    banner_font_scale = max(0.45, min(0.9, banner_h / 50.0))
    banner_thick = max(1, int(round(banner_font_scale * 1.5)))

    banner_left = np.zeros((banner_h, cur_w, 3), dtype=np.uint8)
    banner_right = np.zeros((banner_h, cur_w, 3), dtype=np.uint8)

    # 左侧横幅：原图信息
    orig_info = f"[ORIGINAL] {img_name} ({orig_w}x{orig_h})"
    cv2.putText(
        banner_left,
        orig_info,
        (10, int(banner_h * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        banner_font_scale,
        (220, 220, 220),
        thickness=banner_thick,
        lineType=cv2.LINE_AA,
    )

    # 右侧横幅：划分与实例数量
    inst_count = len(instances)
    status_str = f"[{split_name.upper()}] {inst_count} instance(s)" if inst_count > 0 else f"[{split_name.upper()}] Background (0 instance)"
    status_color = (0, 255, 128) if inst_count > 0 else (160, 160, 160)
    cv2.putText(
        banner_right,
        status_str,
        (10, int(banner_h * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        banner_font_scale,
        status_color,
        thickness=banner_thick,
        lineType=cv2.LINE_AA,
    )

    # 5. 上下拼接横幅，再左右拼合双屏
    left_block = np.vstack([banner_left, left_view])
    right_block = np.vstack([banner_right, right_view])
    combined = np.hstack([left_block, right_block])

    return combined


class LabelJsonConverter:
    """labeljson 数据集转实例分割格式处理器"""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        train_ratio: float = 0.8,
        seed: int = 42,
        export_format: str = "coco",
        copy_images: bool = True,
        num_visualize: int = 30,
        vis_max_size: int = 0,
    ):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.train_ratio = max(0.01, min(0.99, train_ratio))
        self.seed = seed
        self.export_format = export_format.lower()
        self.copy_images = copy_images
        self.num_visualize = num_visualize
        self.vis_max_size = max(0, vis_max_size)

        # 路径校验
        self.images_src_dir = self.input_dir / "images"
        self.labels_src_dir = self.input_dir / "labels"

        # 运行时状态统计
        self.categories: Dict[str, int] = {}
        self.train_files: List[str] = []
        self.val_files: List[str] = []
        # 缓存每个有效样本的元数据，避免重复磁盘 IO 并统一支持无 JSON 的隐式背景图
        self._sample_records: Dict[str, Dict[str, Any]] = {}
        self.stats = {
            "total_images": 0,
            "total_json": 0,
            "paired_images": 0,
            "no_json_background_images": 0,
            "empty_json_background_images": 0,
            "missing_images": 0,
            "corrupted_json": 0,
            "unreadable_images": 0,
            "total_shapes": 0,
            "valid_shapes": 0,
            "filtered_shapes": 0,
        }

    def _validate_inputs(self) -> bool:
        """检查源数据目录完整性与数据文件。"""
        if not self.input_dir.exists():
            print(f"[错误] 输入目录不存在: {self.input_dir}")
            return False

        if not self.images_src_dir.exists():
            print(f"[错误] 未找到 images 目录: {self.images_src_dir}")
            return False

        if not self.labels_src_dir.exists():
            print(f"[警告] 未找到 labels 目录: {self.labels_src_dir}，所有图片将作为隐式背景样本处理。")

        # 原始数据保护安全检查：禁止将输出目录指定在源目录内部或源目录本身
        try:
            if self.output_dir == self.input_dir or self.input_dir in self.output_dir.parents:
                print(f"[错误] 安全保护违规: 输出目录不能是输入目录或其子目录！")
                print(f"       输入目录: {self.input_dir}")
                print(f"       输出目录: {self.output_dir}")
                return False
        except Exception:
            pass

        return True

    def _find_image_file(self, json_stem: str, image_path_in_json: Optional[str]) -> Optional[Path]:
        """按 json 内部记录与文件名多策略查找配对的原始图片。"""
        # 1. 优先按 json 中的 imagePath 查找
        if image_path_in_json:
            candidate = self.images_src_dir / Path(image_path_in_json).name
            if candidate.is_file():
                return candidate

        # 2. 按 json_stem + 常见图像后缀查找
        for ext in SUPPORTED_IMAGE_EXTS:
            candidate = self.images_src_dir / f"{json_stem}{ext}"
            if candidate.is_file():
                return candidate

        return None

    def scan_and_split(self) -> bool:
        """以 images/ 为主入口扫描图片与标注，执行稀有类别优先的迭代分层划分。"""
        print("=" * 70)
        print(" [SCAN] 开始扫描 labeljson 数据集 (以 images/ 为主入口)...")
        print(f"    输入路径: {self.input_dir}")
        print(f"    输出路径: {self.output_dir}")
        print("=" * 70)

        # 1. 扫描 images/ 目录下所有支持格式的图片（按文件名排序保证确定性）
        image_paths = sorted(
            [
                p
                for p in self.images_src_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
            ],
            key=lambda p: p.name,
        )
        self.stats["total_images"] = len(image_paths)

        if not image_paths:
            print(f"[错误] 在 {self.images_src_dir} 中未扫描到任何支持的图像文件！")
            return False

        # 检测是否存在同 stem 不同后缀的图片，避免输出文件同名覆盖
        stem_to_image: Dict[str, Path] = {}
        unique_image_paths: List[Path] = []
        for img_p in image_paths:
            if img_p.stem in stem_to_image:
                print(
                    f"[警告] 检测到同名不同后缀图片，跳过重复项: {img_p.name} "
                    f"(已保留: {stem_to_image[img_p.stem].name})"
                )
                continue
            stem_to_image[img_p.stem] = img_p
            unique_image_paths.append(img_p)

        # 2. 扫描 labels/ 下的所有 JSON 并建立图片到 JSON 的映射
        img_to_json_info: Dict[Path, Tuple[Path, Dict[str, Any]]] = {}
        corrupted_json_stems: Set[str] = set()

        if self.labels_src_dir.exists():
            json_paths = sorted(list(self.labels_src_dir.glob("*.json")), key=lambda p: p.name)
            self.stats["total_json"] = len(json_paths)

            for jp in json_paths:
                stem = jp.stem
                try:
                    with open(jp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, dict):
                        raise ValueError("JSON 根节点不是字典对象")
                except Exception as e:
                    self.stats["corrupted_json"] += 1
                    corrupted_json_stems.add(stem)
                    print(f"[警告] 标注文件解析失败，将跳过对应图片以防误标为背景: {jp.name} (原因: {e})")
                    continue

                img_p = self._find_image_file(stem, data.get("imagePath"))
                if img_p is None:
                    self.stats["missing_images"] += 1
                    print(f"[警告] 标注文件 {jp.name} 未在 images 目录找到对应图像，跳过！")
                    continue

                img_resolved = img_p.resolve()
                if img_resolved in img_to_json_info:
                    print(
                        f"[警告] 图像 {img_p.name} 存在多个对应标注文件，保留首个: "
                        f"{img_to_json_info[img_resolved][0].name}，忽略: {jp.name}"
                    )
                    continue
                img_to_json_info[img_resolved] = (jp, data)

        # 3. 统一构建样本记录（含前景标注样本、空 JSON 背景样本、无 JSON 背景样本）
        group_to_stems: Dict[str, Set[str]] = defaultdict(set)
        valid_stems: List[str] = []
        self._sample_records.clear()

        for img_p in unique_image_paths:
            stem = img_p.stem
            img_resolved = img_p.resolve()

            # 若该图片存在同名损坏 JSON 且未匹配到其他有效 JSON，跳过以避免将损坏 NG 标为背景
            if img_resolved not in img_to_json_info and stem in corrupted_json_stems:
                continue

            if img_resolved in img_to_json_info:
                jp, data = img_to_json_info[img_resolved]
                img_width = int(data.get("imageWidth") or 0)
                img_height = int(data.get("imageHeight") or 0)
                if img_width <= 0 or img_height <= 0:
                    size = read_image_size(img_p)
                    if size is None:
                        self.stats["unreadable_images"] += 1
                        print(f"[警告] 无法读取图片尺寸，跳过损坏图像: {img_p.name}")
                        continue
                    img_width, img_height = size

                shapes = data.get("shapes", [])
                if not isinstance(shapes, list):
                    shapes = []

                # 预检有效多边形类别，避免仅有无效多边形的图错误归入前景类别
                sample_cats: Set[str] = set()
                for s in shapes:
                    if not isinstance(s, dict):
                        continue
                    cleaned_pts = clean_polygon_points(s.get("points", []), img_width, img_height, min_points=3)
                    if cleaned_pts is None or calculate_polygon_area(cleaned_pts) <= 0.0:
                        continue
                    lbl = str(s.get("label", "")).strip() or "unlabeled"
                    if lbl not in self.categories:
                        # COCO category_id 从 1 开始编号
                        self.categories[lbl] = len(self.categories) + 1
                    sample_cats.add(lbl)

                self.stats["paired_images"] += 1
                if not sample_cats:
                    self.stats["empty_json_background_images"] += 1

                self._sample_records[stem] = {
                    "image_path": img_p,
                    "json_path": jp,
                    "width": img_width,
                    "height": img_height,
                    "shapes": shapes,
                    "categories": sample_cats,
                }
            else:
                # 无对应 JSON 文件：直接作为隐式背景负样本（0 实例）
                size = read_image_size(img_p)
                if size is None:
                    self.stats["unreadable_images"] += 1
                    print(f"[警告] 无标注图片无法读取尺寸，跳过损坏图像: {img_p.name}")
                    continue
                img_width, img_height = size
                sample_cats = set()
                self.stats["no_json_background_images"] += 1

                self._sample_records[stem] = {
                    "image_path": img_p,
                    "json_path": None,
                    "width": img_width,
                    "height": img_height,
                    "shapes": [],
                    "categories": sample_cats,
                }

            valid_stems.append(stem)
            if sample_cats:
                for cat in sample_cats:
                    group_to_stems[cat].add(stem)
            else:
                group_to_stems[BACKGROUND_GROUP_KEY].add(stem)

        if not valid_stems:
            print("[错误] 扫描完成但没有可用样本，无法继续构建数据集！")
            return False

        bg_total = self.stats["no_json_background_images"] + self.stats["empty_json_background_images"]
        print(
            f" -> 扫描完成: 共 {len(valid_stems)} 张可用图像 "
            f"(有JSON: {self.stats['paired_images']}, 无JSON背景: {self.stats['no_json_background_images']}, "
            f"空JSON背景: {self.stats['empty_json_background_images']}, 隐式背景合计: {bg_total})"
        )
        print(f" -> 发现前景类别 ({len(self.categories)}): {self.categories}")

        # 4. 迭代分层划分 Train / Val（稀有类别优先，保证多标签场景下少数类比例准确且完全可复现）
        rng = random.Random(self.seed)
        train_set: Set[str] = set()
        val_set: Set[str] = set()

        # 按每组包含的图片数从少到多排序，样本数相同时按名称排序保证确定性
        ordered_groups = sorted(
            ((grp, sorted(stems)) for grp, stems in group_to_stems.items()),
            key=lambda item: (len(item[1]), item[0]),
        )

        for grp, stems_sorted in ordered_groups:
            total_n = len(stems_sorted)
            already_train = sum(1 for s in stems_sorted if s in train_set)
            unassigned = [s for s in stems_sorted if s not in train_set and s not in val_set]
            if not unassigned:
                continue

            if total_n == 1:
                # 极少样本（仅 1 张图）优先进入训练集，避免训练阶段完全未见过该类别
                target_train = 1
            else:
                target_train = int(round(total_n * self.train_ratio))
                # 总数 >= 2 时尽量保证 train 与 val 各至少 1 张
                target_train = max(1, min(total_n - 1, target_train))

            need_train = max(0, min(len(unassigned), target_train - already_train))
            rng.shuffle(unassigned)

            for s in unassigned[:need_train]:
                train_set.add(s)
            for s in unassigned[need_train:]:
                val_set.add(s)

        # 兜底补全未划分样本（正常情况下不会触发）
        for s in valid_stems:
            if s not in train_set and s not in val_set:
                if len(train_set) <= len(val_set) * (self.train_ratio / (1.0 - self.train_ratio)):
                    train_set.add(s)
                else:
                    val_set.add(s)

        self.train_files = sorted(list(train_set))
        self.val_files = sorted(list(val_set))

        total_valid = len(valid_stems)
        train_pct = (len(self.train_files) / total_valid * 100.0) if total_valid > 0 else 0.0
        val_pct = (len(self.val_files) / total_valid * 100.0) if total_valid > 0 else 0.0
        print(f" -> 划分统计 (目标 train_ratio={self.train_ratio:.2f}, seed={self.seed}):")
        print(f"    训练集 (train): {len(self.train_files)} 张 ({train_pct:.1f}%)")
        print(f"    验证集 (valid): {len(self.val_files)} 张 ({val_pct:.1f}%)")

        # 打印各类别（含隐式背景）在 train / valid 中的明细分布表
        print(" -> 各类别分层划分明细 (按图片数统计，多标签图片会在对应多个类别中分别计数):")
        print(f"    {'类别名称':<24} {'总数':>6} {'Train':>6} {'Valid':>6} {'Train占比':>10}")
        print("    " + "-" * 58)
        display_groups = [c for c in self.categories.keys()]
        if BACKGROUND_GROUP_KEY in group_to_stems:
            display_groups.append(BACKGROUND_GROUP_KEY)
        for grp in display_groups:
            stems_in_grp = group_to_stems.get(grp, set())
            n_total = len(stems_in_grp)
            n_tr = sum(1 for s in stems_in_grp if s in train_set)
            n_va = sum(1 for s in stems_in_grp if s in val_set)
            ratio_str = f"{(n_tr / n_total * 100.0):.1f}%" if n_total > 0 else "0.0%"
            disp_name = "background (隐式背景)" if grp == BACKGROUND_GROUP_KEY else grp
            print(f"    {disp_name:<24} {n_total:>6} {n_tr:>6} {n_va:>6} {ratio_str:>10}")

        return True

    def _convert_coco_split(
        self,
        stems: List[str],
        split_name: str,
        split_dir: Path,
        annotation_filename: str = "_annotations.coco.json",
    ) -> Dict[str, Any]:
        """构建单个划分的 COCO 数据集并拷贝图像。"""
        split_dir.mkdir(parents=True, exist_ok=True)

        coco_data: Dict[str, Any] = {
            "info": {
                "description": f"Instance Segmentation Dataset - {split_name}",
                "version": "1.0",
                "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "licenses": [],
            "categories": [
                {"id": cid, "name": cname, "supercategory": "object"}
                for cname, cid in self.categories.items()
            ],
            "images": [],
            "annotations": [],
        }

        ann_id = 1
        for img_id, stem in enumerate(tqdm(stems, desc=f"构建 COCO [{split_name}]"), start=1):
            rec = self._sample_records.get(stem)
            if rec is None:
                continue

            src_img_p: Path = rec["image_path"]
            if not src_img_p.is_file():
                continue

            img_name = src_img_p.name
            img_width = int(rec["width"])
            img_height = int(rec["height"])

            # 拷贝图片到划分目录
            dst_img_p = split_dir / img_name
            if self.copy_images:
                shutil.copy2(src_img_p, dst_img_p)

            # 添加 image 项（无 JSON 的背景图也写入 images，但 annotations 为 0 条）
            coco_data["images"].append(
                {
                    "id": img_id,
                    "file_name": img_name,
                    "width": img_width,
                    "height": img_height,
                }
            )

            # 处理 shapes 标注
            for shape in rec["shapes"]:
                if not isinstance(shape, dict):
                    continue
                self.stats["total_shapes"] += 1
                lbl = str(shape.get("label", "")).strip() or "unlabeled"
                cat_id = self.categories.get(lbl, 1)

                pts = shape.get("points", [])
                cleaned_pts = clean_polygon_points(pts, img_width, img_height, min_points=3)
                if cleaned_pts is None:
                    self.stats["filtered_shapes"] += 1
                    continue

                area = calculate_polygon_area(cleaned_pts)
                if area <= 0.0:
                    self.stats["filtered_shapes"] += 1
                    continue

                bbox = calculate_bbox(cleaned_pts)
                # COCO 多边形格式为平铺一维浮点数组: [x1, y1, x2, y2, ...]
                segmentation_flat: List[float] = []
                for pt in cleaned_pts:
                    segmentation_flat.extend([float(pt[0]), float(pt[1])])

                coco_data["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "segmentation": [segmentation_flat],
                        "area": round(area, 2),
                        "bbox": [round(b, 2) for b in bbox],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
                self.stats["valid_shapes"] += 1

        # 保存 COCO json
        save_path = split_dir / annotation_filename
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

        print(f"   [成功] 保存 COCO 标注: {save_path} (图片数: {len(coco_data['images'])}, 实例数: {len(coco_data['annotations'])})")
        return coco_data

    def _convert_yolo_split(
        self,
        stems: List[str],
        split_name: str,
        yolo_root: Path,
    ) -> None:
        """构建 YOLO-seg 格式 (images/{split}/*.png, labels/{split}/*.txt)。"""
        img_out_dir = yolo_root / "images" / split_name
        lbl_out_dir = yolo_root / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        # YOLO 类别索引从 0 起算
        sorted_cats = sorted(self.categories.keys(), key=lambda k: self.categories[k])
        cat_to_yolo_idx = {name: idx for idx, name in enumerate(sorted_cats)}

        for stem in tqdm(stems, desc=f"构建 YOLO-seg [{split_name}]"):
            rec = self._sample_records.get(stem)
            if rec is None:
                continue

            src_img_p: Path = rec["image_path"]
            if not src_img_p.is_file():
                continue

            img_name = src_img_p.name
            img_width = float(rec["width"] or 1.0)
            img_height = float(rec["height"] or 1.0)

            # 拷贝图片
            if self.copy_images:
                shutil.copy2(src_img_p, img_out_dir / img_name)

            lines: List[str] = []
            for shape in rec["shapes"]:
                if not isinstance(shape, dict):
                    continue
                lbl = str(shape.get("label", "")).strip() or "unlabeled"
                yolo_cls = cat_to_yolo_idx.get(lbl, 0)

                pts = shape.get("points", [])
                cleaned_pts = clean_polygon_points(pts, int(img_width), int(img_height), min_points=3)
                if cleaned_pts is None:
                    continue

                if calculate_polygon_area(cleaned_pts) <= 0.0:
                    continue

                # YOLO 归一化坐标 x / width, y / height
                norm_coords: List[str] = []
                for pt in cleaned_pts:
                    nx = max(0.0, min(1.0, pt[0] / img_width))
                    ny = max(0.0, min(1.0, pt[1] / img_height))
                    norm_coords.extend([f"{nx:.6f}", f"{ny:.6f}"])

                lines.append(f"{yolo_cls} " + " ".join(norm_coords))

            txt_path = lbl_out_dir / f"{stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                # 背景样本写入标准空 .txt 文件
                f.write("\n".join(lines) + ("\n" if lines else ""))

    def run(self) -> bool:
        """主执行逻辑"""
        if not self._validate_inputs():
            return False

        if not self.scan_and_split():
            return False

        print("\n" + "=" * 70)
        print(" [BUILD] 正在生成实例分割数据集产物...")
        print("=" * 70)

        # 1. 导出 COCO 格式
        # 项目约定结构：train/_annotations.coco.json, valid/_annotations.coco.json
        if self.export_format in ("coco", "both"):
            train_dir = self.output_dir / "train"
            val_dir = self.output_dir / "valid"

            self._convert_coco_split(self.train_files, "train", train_dir)
            self._convert_coco_split(self.val_files, "valid", val_dir)

            # 补充保存通用 categories 与 class_names 配置供训练脚本直接读取
            class_names_path = self.output_dir / "class_names.json"
            reversed_mapping = {cid: cname for cname, cid in self.categories.items()}
            with open(class_names_path, "w", encoding="utf-8") as f:
                json.dump(reversed_mapping, f, indent=2, ensure_ascii=False)
            print(f"   [成功] 保存类别配置文件: {class_names_path}")

        # 2. 导出 YOLO 格式
        if self.export_format in ("yolo", "both"):
            yolo_root = self.output_dir if self.export_format == "yolo" else self.output_dir / "yolo_format"
            self._convert_yolo_split(self.train_files, "train", yolo_root)
            self._convert_yolo_split(self.val_files, "val", yolo_root)

            # 生成 data.yaml 配置
            data_yaml = [
                f"path: {yolo_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                f"names: {[c for c in sorted(self.categories.keys(), key=lambda k: self.categories[k])]}",
            ]
            with open(yolo_root / "data.yaml", "w", encoding="utf-8") as f:
                f.write("\n".join(data_yaml) + "\n")
            print(f"   [成功] 保存 YOLO 配置: {yolo_root / 'data.yaml'}")

        # 3. 抽样生成质检可视化对比图
        if self.num_visualize != 0:
            self.generate_visualizations()

        print("\n" + "=" * 70)
        print(" [DONE] 数据集转换圆满完成！")
        print(f"    输出根目录: {self.output_dir}")
        print(f"    图像总数: {len(self._sample_records)} (无JSON背景: {self.stats['no_json_background_images']}, 空JSON背景: {self.stats['empty_json_background_images']})")
        print(f"    有效形状总数: {self.stats['valid_shapes']} / {self.stats['total_shapes']}")
        print(f"    过滤无效形状: {self.stats['filtered_shapes']}")
        print(f"    前景类别列表: {list(self.categories.keys())}")
        if self.num_visualize != 0:
            print(f"    可视化目录: {self.output_dir / 'visualizations'}")
        print("=" * 70)
        return True

    def generate_visualizations(self) -> None:
        """从划分后的数据集中抽样生成左右双屏高对比度质检对比图。"""
        if self.num_visualize == 0:
            return

        vis_root = self.output_dir / "visualizations"
        print("\n" + "=" * 70)
        print(" [VIS] 正在生成质检抽样可视化数据...")
        print(f"    可视化保存根路径: {vis_root}")
        print(f"    设定抽样数: {self.num_visualize} {'(全量生成)' if self.num_visualize < 0 else ''}")
        if self.vis_max_size > 0:
            print(f"    分辨率限制: 单边最大 {self.vis_max_size}px")
        else:
            print("    分辨率设置: 保持 1:1 原始工业相机分辨率")
        print("=" * 70)

        splits_info = [
            ("train", self.train_files),
            ("valid", self.val_files),
        ]

        total_files = len(self.train_files) + len(self.val_files)
        if total_files == 0:
            print("[警告] 划分中无可用样本，跳过可视化生成！")
            return

        # 计算各个 split 的抽样配额
        if self.num_visualize < 0:
            quota_map = {"train": len(self.train_files), "valid": len(self.val_files)}
        else:
            train_target = int(round(self.num_visualize * (len(self.train_files) / max(1, total_files))))
            if len(self.train_files) > 0 and len(self.val_files) > 0 and self.num_visualize >= 2:
                train_target = max(1, min(len(self.train_files), train_target))
                val_target = max(1, min(len(self.val_files), self.num_visualize - train_target))
            else:
                train_target = min(len(self.train_files), train_target)
                val_target = min(len(self.val_files), self.num_visualize - train_target)
            quota_map = {"train": train_target, "valid": val_target}

        total_generated = 0

        for split_name, stems in splits_info:
            quota = quota_map.get(split_name, 0)
            if quota <= 0 or not stems:
                continue

            split_vis_dir = vis_root / split_name
            split_vis_dir.mkdir(parents=True, exist_ok=True)

            # 区分含有效标注的正样本与无标注/无 JSON 的隐式背景负样本
            pos_stems: List[str] = []
            neg_stems: List[str] = []

            for s in stems:
                rec = self._sample_records.get(s)
                if rec is None:
                    continue
                if len(rec["categories"]) > 0:
                    pos_stems.append(s)
                else:
                    neg_stems.append(s)

            rnd = random.Random(self.seed + (10 if split_name == "train" else 20))
            rnd.shuffle(pos_stems)
            rnd.shuffle(neg_stems)

            # 抽样分配：优先选择含实例的正样本，保留约 10% 负样本用于验证背景
            if self.num_visualize < 0 or quota >= len(stems):
                selected = pos_stems + neg_stems
            else:
                num_neg = min(len(neg_stems), max(0, int(quota * 0.1)))
                num_pos = min(len(pos_stems), quota - num_neg)
                # 若某一类别不足则相互补齐
                if num_pos + num_neg < quota:
                    rem = quota - (num_pos + num_neg)
                    if len(pos_stems) > num_pos:
                        num_pos = min(len(pos_stems), num_pos + rem)
                    elif len(neg_stems) > num_neg:
                        num_neg = min(len(neg_stems), num_neg + rem)
                selected = pos_stems[:num_pos] + neg_stems[:num_neg]
                rnd.shuffle(selected)
                selected = selected[:quota]

            desc = f"生成可视化 [{split_name}]"
            split_count = 0
            for s in tqdm(selected, desc=desc):
                rec = self._sample_records.get(s)
                if rec is None:
                    continue

                src_img_p: Path = rec["image_path"]
                if not src_img_p.is_file():
                    continue

                try:
                    # Windows 中文路径安全二进制读取
                    raw_bytes = np.fromfile(str(src_img_p), dtype=np.uint8)
                    img_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        continue
                except Exception as e:
                    print(f"[警告] 读取图片失败，跳过: {src_img_p.name} (原因: {e})")
                    continue

                img_h, img_w = img_bgr.shape[:2]

                # 收集当前图片的有效实例（无 JSON 背景图 shapes 为空，渲染为 Background (0 instance)）
                instances: List[Dict[str, Any]] = []
                for shape in rec["shapes"]:
                    if not isinstance(shape, dict):
                        continue
                    lbl = str(shape.get("label", "")).strip() or "unlabeled"
                    cat_id = self.categories.get(lbl, 1)
                    pts = shape.get("points", [])
                    cleaned_pts = clean_polygon_points(pts, img_w, img_h, min_points=3)
                    if cleaned_pts is None:
                        continue
                    area = calculate_polygon_area(cleaned_pts)
                    if area <= 0.0:
                        continue
                    bbox = calculate_bbox(cleaned_pts)
                    color = get_class_color(cat_id)

                    instances.append({
                        "label": lbl,
                        "category_id": cat_id,
                        "points": cleaned_pts,
                        "bbox": bbox,
                        "area": area,
                        "color": color,
                    })

                try:
                    vis_image = render_visual_comparison(
                        raw_bgr=img_bgr,
                        instances=instances,
                        split_name=split_name,
                        img_name=src_img_p.name,
                        max_size=self.vis_max_size,
                    )

                    save_path = split_vis_dir / f"{s}_vis.jpg"
                    # 安全写入 JPEG 95% 兼顾高画质与紧凑存储
                    success, buf = cv2.imencode(".jpg", vis_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    if success:
                        buf.tofile(str(save_path))
                        split_count += 1
                        total_generated += 1
                except Exception as e:
                    print(f"[警告] 可视化渲染保存失败，跳过: {s} (原因: {e})")
                    continue

            print(f"   [成功] [{split_name}] 可视化图生成完成: {split_count} 张 -> {split_vis_dir}")

        print(f"   [汇总] 可视化质检数据全部生成完毕，共计: {total_generated} 张")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 labeljson 标注格式转换为实例分割训练集 (支持 COCO 与 YOLO 格式划分 train/val，并自动生成质检可视化对比图)"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"源数据根目录 (包含 images/ 和 labels/，默认: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"目标数据集输出目录 (默认: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集所占比例 (范围 0.01~0.99，默认: 0.8 即 8:2 划分)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="coco",
        choices=["coco", "yolo", "both"],
        help="输出的实例分割格式：coco (默认，对齐轻训库 train_l8_fpc_inst.py), yolo, 或 both",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="仅生成标注文件，不实际拷贝图片文件",
    )
    parser.add_argument(
        "--num-visualize",
        type=int,
        default=-1,
        help="生成的可视化抽样图片数量 (默认: 30；设为 -1 时全量生成，设为 0 时跳过生成)",
    )
    parser.add_argument(
        "--vis-max-size",
        type=int,
        default=0,
        help="可视化图片单边最大分辨率限制 (默认: 0 保持 1:1 原图；如设为 1920 则等比缩放)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    converter = LabelJsonConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        export_format=args.format,
        copy_images=not args.no_copy_images,
        num_visualize=args.num_visualize,
        vis_max_size=args.vis_max_size,
    )
    success = converter.run()
    if not success:
        sys.exit(1)
