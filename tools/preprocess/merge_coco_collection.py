"""COCO 实例分割多数据集集合（Dataset Collection）合并与构建工具。

功能特性：
1. 支持多路径与通配符（Glob）输入，自动识别多个 COCO 子数据集；
2. 类别按类别名称 (name) 自动对齐统一，解决不同标注批次中 category_id 不一致的问题；
3. 图像 ID 与标注 ID 全局重编码（自增），解决多数据集间 ID 冲突；
4. 图像文件名防重：自动附加子数据集别名前缀，杜绝同名图片覆盖；
5. Windows 存储优化：优先采用硬链接（os.link）避免占用双倍磁盘空间与漫长的复制等待，若跨盘则自动回退为安全拷贝；
6. 自动聚合子集的 train、valid、test 划分，生成标准 COCO 结构；
7. 产出详细合并统计报告（collection_summary.json）与控制台打印；
8. 既可作为独立 CLI 运行，亦可被训练脚本（如 train_l8_fpc_inst_vitb.py）导入作为前置处理器。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# 保证 Windows 命令行控制台输出 utf-8 编码兼容
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tqdm import tqdm


# ===========================================================================
# COCO 多数据集集合合并 - 全局参数配置区 (Global Configuration)
# 说明：在 IDE (VS Code / PyCharm) 中直接按 F5 运行即可生效；亦支持命令行传参覆盖。
# ===========================================================================
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists() or (p / ".git").exists()), Path(__file__).resolve().parents[2])
DATASET_ROOT = PROJECT_ROOT / "tools" / "DataSet"

# 1. 待合并的子数据集列表（支持目录路径、字符串或通配符）
# 示例：
#   [DATASET_ROOT / "FPC_Line4_Seg_coco", DATASET_ROOT / "FPC_Line9_Seg_coco"]
#   [str(DATASET_ROOT / "FPC_Line*")]
DATASETS = [
    DATASET_ROOT / "L9Seg-data_coco1_square",
    DATASET_ROOT / "L9Seg-data_coco_square",
]

# 2. 合并产物的自定义集合目录名 (设置为 None 时，根据所选数据集名字自动生成)
# 示例: "FPC_Lines_Aggregated_coco"
COLLECTION_NAME = "FPC_Lines_Aggregated_coco"

# 3. 合并产物的输出根目录 (设置为 None 时，默认自动存放在 tools/DataSet/<COLLECTION_NAME>)
OUT_DIR = None

# 4. 图像链接策略 ("auto": 优先 Windows NTFS 硬链接避免双倍空间占用, "link": 强制硬链接, "copy": 强制物理拷贝)
LINK_MODE = "auto"
# ===========================================================================


SUPPORTED_SPLITS = ("train", "valid", "test")
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def sanitize_tag(name: str) -> str:
    """清洗数据集目录名，生成合法的简明文件名前缀。"""
    cleaned = re.sub(r"[^\w\-]", "_", name)
    return cleaned.strip("_")


def resolve_dataset_paths(input_paths: List[str | Path]) -> List[Path]:
    """解析并展开命令行传入的多路径与通配符，去重并过滤有效目录。"""
    resolved_paths: List[Path] = []
    seen: Set[str] = set()

    for item in input_paths:
        item_str = str(item).strip()
        if not item_str:
            continue

        # 支持包含通配符的路径 (如 tools/DataSet/FPC_Line*)
        if any(char in item_str for char in ["*", "?", "["]):
            matched = glob.glob(item_str)
            candidate_paths = [Path(m) for m in matched]
        else:
            candidate_paths = [Path(item_str)]

        for path in candidate_paths:
            resolved = path.resolve()
            if str(resolved) not in seen and resolved.is_dir():
                # 检查是否包含至少一个标准的 split 目录
                has_split = any((resolved / s).is_dir() for s in SUPPORTED_SPLITS)
                if has_split:
                    seen.add(str(resolved))
                    resolved_paths.append(resolved)

    return sorted(resolved_paths, key=lambda p: p.name)


def link_or_copy_file(src: Path, dst: Path, link_mode: str = "auto") -> str:
    """创建文件链接或复制。
    
    link_mode:
        - "link": 仅硬链接（若失败抛异常）
        - "copy": 强制复制
        - "auto": 优先硬链接，失败时自动降级为复制
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "skipped"

    if link_mode in ("auto", "link"):
        try:
            os.link(src, dst)
            return "linked"
        except (OSError, NotImplementedError):
            if link_mode == "link":
                raise
            # auto 模式下失败回退到复制

    shutil.copy2(src, dst)
    return "copied"


class CocoCollectionMerger:
    """COCO 数据集集合合并引擎。"""

    def __init__(
        self,
        dataset_paths: List[Path],
        output_dir: Path,
        link_mode: str = "auto",
        verbose: bool = True,
    ) -> None:
        if not dataset_paths:
            raise ValueError("传入的数据集路径列表为空，无法进行合并。")
        self.dataset_paths = dataset_paths
        self.output_dir = output_dir.resolve()
        self.link_mode = link_mode
        self.verbose = verbose

    def _discover_categories(self) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """第一遍扫描：按类别名称对齐所有数据集中的 categories。"""
        all_cat_names: List[str] = []
        cat_name_to_supercat: Dict[str, str] = {}

        for ds_path in self.dataset_paths:
            for split in SUPPORTED_SPLITS:
                anno_file = ds_path / split / "_annotations.coco.json"
                if not anno_file.is_file():
                    continue

                with open(anno_file, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except Exception as e:
                        print(f"[警告] 无法读取 {anno_file}: {e}", file=sys.stderr)
                        continue

                for cat in data.get("categories", []):
                    cname = str(cat.get("name", "")).strip()
                    if not cname:
                        continue
                    if cname not in all_cat_names:
                        all_cat_names.append(cname)
                        cat_name_to_supercat[cname] = cat.get("supercategory", "none")

        # 统一排序，分配 1-indexed 的 category_id
        unified_categories: List[Dict[str, Any]] = []
        name_to_unified_id: Dict[str, int] = {}

        for idx, cname in enumerate(sorted(all_cat_names), start=1):
            name_to_unified_id[cname] = idx
            unified_categories.append({
                "id": idx,
                "name": cname,
                "supercategory": cat_name_to_supercat.get(cname, "none"),
            })

        return unified_categories, name_to_unified_id

    def merge(self) -> Dict[str, Any]:
        """执行完整的多数据集合并流程。"""
        if self.verbose:
            print("=" * 70)
            print("【COCO 多数据集集合合并引擎 (COCO Dataset Collection Merger)】")
            print("=" * 70)
            print(f"参与合并的子数据集总数: {len(self.dataset_paths)} 个")
            for idx, p in enumerate(self.dataset_paths, start=1):
                print(f"  [{idx}] {p.name} ({p})")
            print(f"目标集合输出目录: {self.output_dir}")
            print(f"图片引用策略: {self.link_mode} (优先硬链接以节约磁盘)")
            print("-" * 70)

        unified_categories, name_to_unified_id = self._discover_categories()

        if not unified_categories:
            raise RuntimeError("在提供的数据集子目录中未发现有效的 categories，请检查标注文件格式。")

        if self.verbose:
            print("统一对齐后的类别体系:")
            for cat in unified_categories:
                print(f"  - ID: {cat['id']} | 类别名: '{cat['name']}' (Supercategory: {cat['supercategory']})")
            print("-" * 70)

        # 准备输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        global_image_id = 1
        global_anno_id = 1

        overall_stats: Dict[str, Any] = {
            "source_datasets": [str(p) for p in self.dataset_paths],
            "categories": unified_categories,
            "splits": {},
            "link_mode": self.link_mode,
        }

        # 逐 split 进行聚合 (train, valid, test)
        for split in SUPPORTED_SPLITS:
            split_out_dir = self.output_dir / split
            split_out_dir.mkdir(parents=True, exist_ok=True)

            merged_images: List[Dict[str, Any]] = []
            merged_annotations: List[Dict[str, Any]] = []
            dataset_image_counts: Dict[str, int] = defaultdict(int)
            dataset_anno_counts: Dict[str, int] = defaultdict(int)
            linked_count = 0
            copied_count = 0

            for ds_path in self.dataset_paths:
                split_in_dir = ds_path / split
                anno_file = split_in_dir / "_annotations.coco.json"
                if not anno_file.is_file():
                    continue

                ds_tag = sanitize_tag(ds_path.name)

                with open(anno_file, "r", encoding="utf-8") as f:
                    try:
                        coco_data = json.load(f)
                    except Exception as e:
                        print(f"[警告] 解析 {anno_file} 失败，已跳过: {e}", file=sys.stderr)
                        continue

                # 构建该数据集的本地 category_id 到 name 映射
                local_cat_map: Dict[int, str] = {
                    c["id"]: str(c["name"]).strip()
                    for c in coco_data.get("categories", [])
                    if "id" in c and "name" in c
                }

                # 收集原始 annotations，按 image_id 归类
                image_id_to_annos: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
                for anno in coco_data.get("annotations", []):
                    image_id_to_annos[anno.get("image_id")].append(anno)

                # 处理 images
                orig_images = coco_data.get("images", [])
                desc = f"合并 [{ds_tag} -> {split}]"
                iterable = tqdm(orig_images, desc=desc, leave=False) if self.verbose else orig_images

                for img_info in iterable:
                    old_img_id = img_info["id"]
                    old_file_name = img_info["file_name"]

                    # 寻找源图片文件
                    src_img_file = split_in_dir / old_file_name
                    if not src_img_file.is_file():
                        # 有些数据集图片放在 split/images/ 目录下
                        sub_img_file = split_in_dir / "images" / old_file_name
                        if sub_img_file.is_file():
                            src_img_file = sub_img_file
                        else:
                            # 跳过找不到文件的图片记录
                            continue

                    # 前缀防重名策略: <ds_tag>_<orig_filename>
                    new_file_name = f"{ds_tag}_{old_file_name}"
                    dst_img_file = split_out_dir / new_file_name

                    # 链接或拷贝
                    action = link_or_copy_file(src_img_file, dst_img_file, link_mode=self.link_mode)
                    if action == "linked":
                        linked_count += 1
                    elif action == "copied":
                        copied_count += 1

                    # 登记新图片
                    current_img_id = global_image_id
                    global_image_id += 1

                    new_img_item = dict(img_info)
                    new_img_item["id"] = current_img_id
                    new_img_item["file_name"] = new_file_name
                    merged_images.append(new_img_item)
                    dataset_image_counts[ds_path.name] += 1

                    # 处理对应的标注
                    for anno in image_id_to_annos.get(old_img_id, []):
                        old_cat_id = anno.get("category_id")
                        cat_name = local_cat_map.get(old_cat_id)
                        if not cat_name or cat_name not in name_to_unified_id:
                            continue

                        current_anno_id = global_anno_id
                        global_anno_id += 1

                        new_anno_item = dict(anno)
                        new_anno_item["id"] = current_anno_id
                        new_anno_item["image_id"] = current_img_id
                        new_anno_item["category_id"] = name_to_unified_id[cat_name]
                        merged_annotations.append(new_anno_item)
                        dataset_anno_counts[ds_path.name] += 1

            if merged_images:
                # 写入该 split 的 _annotations.coco.json
                out_anno_json = split_out_dir / "_annotations.coco.json"
                split_coco_dict = {
                    "info": {
                        "description": f"Aggregated COCO Dataset Collection - {split}",
                        "source_datasets": [p.name for p in self.dataset_paths],
                        "version": "1.0",
                    },
                    "licenses": [],
                    "categories": unified_categories,
                    "images": merged_images,
                    "annotations": merged_annotations,
                }
                with open(out_anno_json, "w", encoding="utf-8") as f:
                    json.dump(split_coco_dict, f, ensure_ascii=False, indent=2)

                overall_stats["splits"][split] = {
                    "images_count": len(merged_images),
                    "annotations_count": len(merged_annotations),
                    "linked_files": linked_count,
                    "copied_files": copied_count,
                    "details_by_dataset": {
                        p.name: {
                            "images": dataset_image_counts.get(p.name, 0),
                            "annotations": dataset_anno_counts.get(p.name, 0),
                        }
                        for p in self.dataset_paths
                    },
                }

                if self.verbose:
                    print(
                        f"  [✓] {split.upper()} 子集聚合完成: {len(merged_images)} 张图片, "
                        f"{len(merged_annotations)} 个标注 (硬链接: {linked_count}, 复制: {copied_count})"
                    )

        # 保存全局元数据与统计报告
        summary_file = self.output_dir / "collection_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(overall_stats, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print("-" * 70)
            print(f"[OK] 数据集集合合并完毕，已保存至: {self.output_dir}")
            print(f"  - 汇总报告: {summary_file}")
            print("=" * 70)

        return overall_stats


def merge_coco_datasets(
    datasets: List[str | Path],
    output_dir: Optional[str | Path] = None,
    collection_name: Optional[str] = None,
    link_mode: str = "auto",
    verbose: bool = True,
) -> Path:
    """供外部 Python 脚本（如训练脚本）直接调用的便捷入口。
    
    返回合并完成后的集合根目录 Path。
    """
    resolved_paths = resolve_dataset_paths(datasets)
    if not resolved_paths:
        raise ValueError(f"未找到任何有效的数据集目录，传入参数为: {datasets}")

    # 若仅传入单个数据集，直接返回其本身，无需重复组装
    if len(resolved_paths) == 1 and not output_dir and not collection_name:
        return resolved_paths[0]

    if output_dir is None:
        tools_dataset_root = resolved_paths[0].parent
        if collection_name is None:
            tags = [sanitize_tag(p.name) for p in resolved_paths]
            if len(tags) <= 3:
                collection_name = "Collection_" + "_".join(tags)
            else:
                collection_name = f"Collection_{tags[0]}_{tags[1]}_plus_{len(tags)-2}more"
        out_path = tools_dataset_root / collection_name
    else:
        out_path = Path(output_dir).resolve()

    merger = CocoCollectionMerger(
        dataset_paths=resolved_paths,
        output_dir=out_path,
        link_mode=link_mode,
        verbose=verbose,
    )
    merger.merge()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="COCO 实例分割多数据集集合（Dataset Collection）合并工具"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="待合并的 COCO 数据集路径列表（默认使用全局变量 DATASETS，支持空格分隔多个路径或通配符）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="合并产物的输出目录（默认使用全局变量 OUT_DIR，若为 None 则自动生成在 DataSet/<collection_name>）",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help=f"自定义集合目录名 (默认使用全局变量: {COLLECTION_NAME})",
    )
    parser.add_argument(
        "--link-mode",
        choices=["auto", "link", "copy"],
        default=None,
        help=f"图片链接模式：auto=优先硬链接失败转拷贝, link=仅硬链接, copy=强制复制 (默认使用全局变量: {LINK_MODE})",
    )

    args = parser.parse_args()

    # 优先采用命令行传入值，若未传入则回退到顶部全局变量
    effective_datasets = args.datasets if args.datasets is not None else list(DATASETS)
    effective_out = args.out if args.out is not None else OUT_DIR
    effective_collection_name = args.collection_name if args.collection_name is not None else COLLECTION_NAME
    effective_link_mode = args.link_mode if args.link_mode is not None else LINK_MODE

    try:
        final_dir = merge_coco_datasets(
            datasets=effective_datasets,
            output_dir=effective_out,
            collection_name=effective_collection_name,
            link_mode=effective_link_mode,
            verbose=True,
        )
        print(f"\n[成功] 数据集集合就绪: {final_dir}")
    except Exception as e:
        print(f"\n[错误] 合并数据集失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
