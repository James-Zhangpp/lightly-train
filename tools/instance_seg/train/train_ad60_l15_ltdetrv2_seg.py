# -*- coding: utf-8 -*-
"""AD60_L15 FPC 实例分割训练脚本（ltdetrv2-seg-m-coco · EdgeCrafter 架构）。

功能特性：
  1. 架构特性：基于 LTDETRv2 (EdgeCrafter ECViT-TinyPlus + HybridEncoder + ECSegTransformer)，
     轻量级、高精度、高推理吞吐，兼顾工业部署友好性；
  2. 样本量自适应调度：自动读取 train/_annotations.coco.json 的样本数量，智能推算
     - 单轮步数（StepsPerEpoch）
     - 总训练步数（STEPS = StepsPerEpoch * target_epochs）
     - 学习率预热区间（LR_WARMUP，前 5%~10% 步数）
     - 验证与 Checkpoint 保存间隔（VAL_EVERY_NUM_STEPS，自适应对齐 1~2 个整轮 Epoch 周期）
  3. 类别自适应提取：自动解析 COCO categories 映射；
  4. 训练模式自适应：若检测到产物目录下已存在 exported_best.pt，自动降低学习率进入微调（Fine-tune）模式；
  5. 工业级数据增强：水平/垂直翻转、90度旋转、小角度 ±15° 旋转、微幅色彩对比度微调；
  6. 参数可配置：支持 CLI 传入 --dataset, --out, --batch-size, --epochs, --size, --steps, --dry-run 等。

用法示例：
  # 自动推算并启动训练（默认 80 个 Epoch，Batch Size=6，输入 512x512）
  python tools/train_ad60_l15_ltdetrv2_seg.py

  # 演练模式：仅推算超参数与调度面板，不启动真实 GPU 训练
  python tools/train_ad60_l15_ltdetrv2_seg.py --dry-run

  # 自定义目标轮数、批大小与图像尺寸
  python tools/train_ad60_l15_ltdetrv2_seg.py --epochs 100 --batch-size 8 --size 512
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# 保证 Windows 命令行控制台输出 utf-8 编码兼容
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 关闭 oneDNN 浮点警告与降低 TF 日志等级
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists() or (p / ".git").exists()), Path(__file__).resolve().parents[3])
TOOLS_DIR = next((p for p in Path(__file__).resolve().parents if p.name == "tools"), Path(__file__).resolve().parents[2])
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import lightly_train

# ---------------------------------------------------------------------------
# 默认常量配置
# ---------------------------------------------------------------------------
DEFAULT_DATASET_ROOT = TOOLS_DIR / "DataSet" / "AD60_L15_coco"
DEFAULT_OUT_DIR = PROJECT_ROOT / "out" / "AD60_L15_InSeg_ltdetrv2_m"
DEFAULT_MODEL = "ltdetrv2-seg-m-coco"

DEFAULT_IMAGE_SIZE = 512
DEFAULT_BATCH_SIZE = 6
DEFAULT_EPOCHS = 80


def inspect_dataset_and_calculate_schedule(
    train_json_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    target_epochs: int = DEFAULT_EPOCHS,
    fine_tune: bool = False,
    custom_steps: int | None = None,
    custom_val_interval: int | None = None,
) -> Dict[str, Any]:
    """读取训练集 COCO 标注并根据样本量自适应计算训练周期、步数与验证频次。

    返回:
        字典包含:
            - num_train_images: int
            - steps_per_epoch: int
            - total_steps: int
            - actual_epochs: float
            - lr_warmup_steps: Tuple[int, int]
            - val_every_steps: int
            - class_names: Dict[int, str]
    """
    if not train_json_path.is_file():
        raise FileNotFoundError(f"[错误] 找不到训练集标注文件: {train_json_path}")

    with open(train_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    num_train_images = len(images)
    if num_train_images == 0:
        raise ValueError(f"[错误] 训练集中未检测到任何有效图像记录: {train_json_path}")

    # 自动解析类别列表
    categories = coco_data.get("categories", [])
    if categories:
        class_names = {cat["id"]: cat["name"] for cat in categories}
    else:
        class_names = {1: "FPC"}

    # 1. 计算单个 Epoch 步数
    steps_per_epoch = max(1, math.ceil(num_train_images / batch_size))

    # 2. 计算总步数 STEPS
    if custom_steps is not None and custom_steps > 0:
        total_steps = custom_steps
        actual_epochs = round(total_steps / steps_per_epoch, 2)
    else:
        raw_steps = steps_per_epoch * target_epochs
        if fine_tune:
            total_steps = max(600, int(raw_steps * 0.75))
        else:
            total_steps = max(800, int(raw_steps))
        actual_epochs = round(total_steps / steps_per_epoch, 2)

    # 3. 计算学习率预热 LR_WARMUP
    if fine_tune:
        warmup_start = max(20, int(total_steps * 0.04))
        warmup_end = max(50, int(total_steps * 0.08))
    else:
        warmup_start = max(50, int(total_steps * 0.06))
        warmup_end = max(100, int(total_steps * 0.12))
    lr_warmup_steps = (warmup_start, warmup_end)

    # 4. 计算验证与保存间隔 VAL_EVERY_NUM_STEPS（优先对齐整轮 Epoch 周期）
    if custom_val_interval is not None and custom_val_interval > 0:
        val_every_steps = custom_val_interval
    else:
        if steps_per_epoch >= 40:
            val_steps = steps_per_epoch  # 数据量适中/偏大时，每 1 个 Epoch 验证一次
        else:
            val_steps = steps_per_epoch * 2  # 小样本（<40步/轮）时，每 2 个 Epoch 验证一次
        # 限制在合理区间 [40, 200]
        val_every_steps = max(40, min(val_steps, 200))

    return {
        "num_train_images": num_train_images,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "actual_epochs": actual_epochs,
        "lr_warmup_steps": lr_warmup_steps,
        "val_every_steps": val_every_steps,
        "class_names": class_names,
    }


def validate_dataset_exists(
    train_json: Path,
    train_img_dir: Path,
    val_json: Path,
    val_img_dir: Path,
) -> None:
    """检查数据集标注和图片目录是否存在。"""
    if not train_json.is_file():
        raise FileNotFoundError(f"[错误] 训练集 COCO 标注文件不存在: {train_json}")
    if not val_json.is_file():
        raise FileNotFoundError(f"[错误] 验证集 COCO 标注文件不存在: {val_json}")
    if not train_img_dir.is_dir():
        raise FileNotFoundError(f"[错误] 训练集图像目录不存在: {train_img_dir}")
    if not val_img_dir.is_dir():
        raise FileNotFoundError(f"[错误] 验证集图像目录不存在: {val_img_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AD60_L15 FPC 实例分割训练脚本 (LTDETRv2-seg-m-coco 架构)"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"COCO 数据集根目录 (默认: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"实验与产物保存目录 (默认: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"模型架构名称 (默认: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"训练批大小 Batch Size (默认: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"目标训练轮数 (默认: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"输入图像尺寸 (默认: {DEFAULT_IMAGE_SIZE})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="总训练步数覆盖值（若指定则覆盖自适应计算结果）",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=None,
        help="评估与保存间隔步数（若指定则覆盖自适应计算结果）",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="冻结 ECViT 骨干网络，仅训练检测与分割头 (默认: False 全量微调)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅演练查看推算出的超参数与调度面板，不启动真实 GPU 训练",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset.resolve()
    train_json = dataset_root / "train" / "_annotations.coco.json"
    train_img_dir = dataset_root / "train"
    val_json = dataset_root / "valid" / "_annotations.coco.json"
    val_img_dir = dataset_root / "valid"

    validate_dataset_exists(train_json, train_img_dir, val_json, val_img_dir)

    out_dir = args.out.resolve()
    exported_best = out_dir / "exported_models" / "exported_best.pt"
    fine_tune_from_best = exported_best.is_file()

    # 动态推算超参数
    sched = inspect_dataset_and_calculate_schedule(
        train_json_path=train_json,
        batch_size=args.batch_size,
        target_epochs=args.epochs,
        fine_tune=fine_tune_from_best,
        custom_steps=args.steps,
        custom_val_interval=args.val_interval,
    )

    steps = sched["total_steps"]
    lr_warmup = sched["lr_warmup_steps"]
    val_every_steps = sched["val_every_steps"]
    class_names = sched["class_names"]
    image_size = (args.size, args.size)

    # 基础学习率配置：微调时稍降低
    base_lr = 2e-4 if not fine_tune_from_best else 1e-4

    print("=" * 70)
    print("【LTDETRv2-seg-m 工业实例分割自适应训练调度面板】")
    print("=" * 70)
    print(f"  - 模型架构: {args.model}")
    print(f"  - 数据集路径: {dataset_root}")
    print(f"  - 训练集样本数: {sched['num_train_images']} 张")
    print(f"  - 图像输入分辨率: {image_size[0]} x {image_size[1]}")
    print(f"  - 批大小 (Batch Size): {args.batch_size}")
    print(f"  - 单轮步数 (Steps Per Epoch): {sched['steps_per_epoch']} 步/轮")
    print(f"  - 目标轮数 (Actual Epochs): {sched['actual_epochs']} 轮")
    print(f"  - 自适应总训练步数 (STEPS): {steps} 步")
    print(f"  - 学习率基础值 (Base LR): {base_lr}")
    print(f"  - 自适应学习率预热 (LR_WARMUP): {lr_warmup[0]} ~ {lr_warmup[1]} 步")
    print(f"  - 自适应验证与保存间隔 (VAL_EVERY): 每 {val_every_steps} 步评估一次")
    print(f"  - Backbone 状态: {'冻结 (Freeze)' if args.freeze_backbone else '全量训练 (End-to-End)'}")
    print(f"  - 类别映射 (Class Names): {class_names}")
    print(f"  - 微调状态: {'从已导出的 best 继续微调' if fine_tune_from_best else '全新训练'}")
    print(f"  - 产物输出路径: {out_dir}")
    print("=" * 70)

    if args.dry_run:
        print("\n[OK] 演练完成 (--dry-run 开启，不启动真实 GPU 训练)。")
        return

    print("\n>>> 开始调用 lightly_train 启动 LTDETRv2-seg-m 实例分割模型训练...")

    lightly_train.train_instance_segmentation(
        out=out_dir,
        model=args.model,
        data={
            "format": "coco",
            "train": {
                "annotations": str(train_json),
                "images": str(train_img_dir),
            },
            "val": {
                "annotations": str(val_json),
                "images": str(val_img_dir),
            },
            "skip_if_annotations_missing": False,
        },
        steps=steps,
        batch_size=args.batch_size,
        num_workers=6,
        devices="auto",
        accelerator="auto",
        strategy="auto",
        precision="bf16-mixed",
        float32_matmul_precision="auto",
        num_nodes=1,
        seed=42,
        overwrite=not fine_tune_from_best,
        resume_interrupted=False,
        checkpoint=exported_best if fine_tune_from_best else None,
        model_args={
            "lr": base_lr,
            "weight_decay": 1e-4,
            "lr_warmup_steps": lr_warmup[1],
            "backbone_freeze": args.freeze_backbone,
            "use_ema_model": True,
        },
        transform_args={
            "image_size": image_size,
            "random_flip": {
                "horizontal_prob": 0.5,
                "vertical_prob": 0.5,
            },
            "random_rotate_90": {"prob": 0.5},
            "random_rotate": {"prob": 0.5, "degrees": 15.0},
        },
        logger_args={
            "val_every_num_steps": val_every_steps,
        },
        metric_args={
            "classwise": True,
        },
        save_checkpoint_args={
            "save_every_num_steps": val_every_steps,
            "save_last": True,
            "save_best": True,
        },
    )

    print(f"\n=== 训练完成 ===")
    print(f"最优导出模型: {exported_best}")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
