"""L8 FPC 实例分割训练脚本 v2（DINOv3-ViT-B/16 + EoMT-Inst，适配 RTX 5070 12GB）。

相比 v1 (train_l8_fpc_inst_vitb.py) 的核心优化：
  1. 修复 backbone_freeze 硬编码问题，默认开启全量微调（解冻骨干网络提取工业纹理特征）；
     若显式传入 --freeze-backbone 且未指定 --lr，自动将学习率提升至 2e-4 保证 Head 收敛；
  2. 适配 RTX 5070 12GB 显存：默认 batch_size=2 + gradient_accumulation_steps=4（等效批量 8），
     兼顾 12GB 显存安全线与平稳的梯度更新；
  3. 针对单图 1~3 个目标的工业场景，将 num_queries 从 50 降为 15，减少冗余低分重叠框；
  4. 提高小样本兜底步数至 1500 步、收窄 Warmup 占比（5%~12%），避免前期低学习率空转；
  5. 平衡损失权重（loss_dice_coefficient=5.0, loss_class_coefficient=4.0, loss_no_object_coefficient=0.2），
     在保持边缘贴合的同时增强对冗余重复 query 的抑制；
  6. 输出目录独立至 out/NP_data_vitb16_v2，避免误加载旧的未收敛权重。

用法：
  # 演练检查超参数与步数推算（推荐先跑一次）
  python tools/train_l8_fpc_inst_vitb_v2.py --dry-run

  # 启动全新训练（RTX 5070 12GB 默认全量微调）
  python tools/train_l8_fpc_inst_vitb_v2.py

  # 若需极省显存模式，可开启骨干冻结（脚本会自动使用 2e-4 学习率）
  python tools/train_l8_fpc_inst_vitb_v2.py --freeze-backbone
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

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
# 默认常量配置（针对 RTX 5070 12GB 与小样本工业 OK/NG 实例分割优化）
# ---------------------------------------------------------------------------
DEFAULT_DATASET_ROOT = TOOLS_DIR / "data" / "NP_BZ_coco"
DEFAULT_OUT_DIR = PROJECT_ROOT / "out" / "NP_BZ_vitb16_v2"

IMAGE_SIZE = (512, 512)
# 12GB 显存全量微调 ViT-B (512x512, bf16) 推荐单步 batch_size=2，配合梯度累积保证更新平稳
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 4
DEFAULT_EPOCHS = 50
# 全量微调基础学习率（对应等效 batch_size=8）；冻结骨干时默认回退至 2e-4
DEFAULT_LR_FULL = 8e-5
DEFAULT_LR_FROZEN = 2e-4
DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_NUM_QUERIES = 15


def inspect_dataset_and_calculate_schedule(
    train_json_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    target_epochs: int = DEFAULT_EPOCHS,
    fine_tune: bool = False,
    custom_steps: int | None = None,
    custom_val_interval: int | None = None,
) -> Dict[str, Any]:
    """读取训练集 COCO 标注并自适应推算训练步数与预热周期。"""
    if not train_json_path.is_file():
        raise FileNotFoundError(f"找不到训练集标注文件: {train_json_path}")

    with open(train_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    num_train_images = len(images)
    if num_train_images == 0:
        raise ValueError(f"训练集中未检测到任何有效图像记录: {train_json_path}")

    categories = coco_data.get("categories", [])
    if categories:
        class_names = {cat["id"]: cat["name"] for cat in categories}
    else:
        class_names = {1: "FPC"}

    # 1. 计算单轮步数（按实际 dataloader batch_size 计算）
    steps_per_epoch = max(1, math.ceil(num_train_images / batch_size))

    # 2. 计算总步数 STEPS，确保小样本下也有充足步数让二分图匹配收敛
    if custom_steps is not None and custom_steps > 0:
        total_steps = custom_steps
    else:
        raw_steps = steps_per_epoch * target_epochs
        if fine_tune:
            total_steps = max(800, int(raw_steps * 0.6))
        else:
            total_steps = max(1500, int(raw_steps))
    actual_epochs = round(total_steps / steps_per_epoch, 2)

    # 3. 收窄 Warmup 占比（5%~12%），避免前期长时间低学习率空转
    if fine_tune:
        warmup_start = max(20, int(total_steps * 0.04))
        warmup_end = max(60, int(total_steps * 0.10))
    else:
        warmup_start = max(40, int(total_steps * 0.05))
        warmup_end = max(100, int(total_steps * 0.12))
    lr_warmup_steps = (warmup_start, warmup_end)

    # 4. 验证与保存间隔
    if custom_val_interval is not None and custom_val_interval > 0:
        val_every_steps = custom_val_interval
    else:
        val_steps = steps_per_epoch if steps_per_epoch >= 50 else steps_per_epoch * 2
        val_every_steps = max(50, min(val_steps, 200))

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
        raise FileNotFoundError(f"训练集 COCO 标注文件不存在: {train_json}")
    if not val_json.is_file():
        raise FileNotFoundError(f"验证集 COCO 标注文件不存在: {val_json}")
    if not train_img_dir.is_dir():
        raise FileNotFoundError(f"训练集图像目录不存在: {train_img_dir}")
    if not val_img_dir.is_dir():
        raise FileNotFoundError(f"验证集图像目录不存在: {val_img_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DINOv3 ViT-Base + EoMT Instance Segmentation v2 (RTX 5070 12GB Optimized)."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to single dataset root folder",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Path(s) to multiple COCO dataset root folders or glob pattern for collection training",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Optional custom folder name for the aggregated dataset collection",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Path to output artifacts folder (default: {DEFAULT_OUT_DIR.name})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Per-step training batch size (default: {DEFAULT_BATCH_SIZE} for 12GB GPU)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=DEFAULT_GRAD_ACCUM,
        help=f"Gradient accumulation steps (default: {DEFAULT_GRAD_ACCUM})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Base learning rate (auto-selected based on --freeze-backbone if omitted)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Target number of epochs for adaptive step calculation (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Manual override for total training steps (optional)",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=None,
        help="Manual override for evaluation step interval (optional)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=DEFAULT_NUM_QUERIES,
        help=f"Number of object queries in EoMT head (default: {DEFAULT_NUM_QUERIES})",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze DINOv3 ViT-B backbone and only train mask/class head",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only inspect dataset and print calculated schedule without starting GPU training",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(f"--batch-size 必须大于 0，当前为: {args.batch_size}")
    if args.grad_accum <= 0:
        raise ValueError(f"--grad-accum 必须大于 0，当前为: {args.grad_accum}")
    if args.num_queries <= 0:
        raise ValueError(f"--num-queries 必须大于 0，当前为: {args.num_queries}")

    # 处理多数据集集合与单数据集兼容
    if args.datasets:
        from merge_coco_collection import merge_coco_datasets

        dataset_root = merge_coco_datasets(
            datasets=args.datasets,
            collection_name=args.collection_name,
            verbose=True,
        )
    else:
        dataset_root = args.dataset

    train_json = dataset_root / "train" / "_annotations.coco.json"
    train_img_dir = dataset_root / "train"
    val_json = dataset_root / "valid" / "_annotations.coco.json"
    val_img_dir = dataset_root / "valid"

    validate_dataset_exists(train_json, train_img_dir, val_json, val_img_dir)

    out_dir = args.out
    exported_best = out_dir / "exported_models" / "exported_best.pt"
    fine_tune_from_best = exported_best.is_file()

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

    # 根据是否冻结骨干网络自动匹配默认学习率，避免冻结后 lr 过小不收敛
    base_lr = args.lr if args.lr is not None else (
        DEFAULT_LR_FROZEN if args.freeze_backbone else DEFAULT_LR_FULL
    )
    effective_lr = base_lr if not fine_tune_from_best else base_lr * 0.5
    effective_batch = args.batch_size * args.grad_accum

    print("=" * 70)
    print("【DINOv3 ViT-Base v2 工业视觉训练调度 (RTX 5070 12GB 优化版)】")
    print("=" * 70)
    print("  - 模型架构: dinov3/vitb16-eomt-inst-coco (ViT-Base, 768维)")
    print(f"  - 数据集路径: {dataset_root}")
    print(f"  - 训练集样本数: {sched['num_train_images']} 张")
    print(f"  - 批大小 (Batch Size): {args.batch_size} × 梯度累积 {args.grad_accum} = 等效 {effective_batch}")
    print(f"  - 学习率 (Base LR): {effective_lr} (预热步数: {lr_warmup[0]} ~ {lr_warmup[1]})")
    print(f"  - 单轮步数 (Steps Per Epoch): {sched['steps_per_epoch']} 步/轮")
    print(f"  - 目标轮数 (Actual Epochs): {sched['actual_epochs']} 轮")
    print(f"  - 总训练步数 (STEPS): {steps} 步")
    print(f"  - 验证与保存间隔 (VAL_EVERY): 每 {val_every_steps} 步评估一次")
    print(f"  - Query 数量 (num_queries): {args.num_queries}")
    print(f"  - 类别映射 (Class Names): {class_names}")
    print(f"  - Backbone 状态: {'冻结 (Freeze)' if args.freeze_backbone else '全量微调 (End-to-End)'}")
    print(f"  - 微调状态: {'从已导出的 best 继续微调' if fine_tune_from_best else '全新训练 (独立输出目录)'}")
    print("  - 损失权重: Dice = 5.0, Class = 4.0, No-Object = 0.2 (强化重复 query 抑制)")
    print(f"  - 产物输出路径: {out_dir}")
    print("=" * 70)

    if args.dry_run:
        print("\n[OK] 演练完成 (--dry-run 开启，不启动真实 GPU 训练)。")
        return

    print("\n>>> 开始调用 lightly_train 启动 ViT-Base v2 实例分割模型训练...")

    lightly_train.train_instance_segmentation(
        out=out_dir,
        model="dinov3/vitb16-eomt-inst-coco",
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
        gradient_accumulation_steps=args.grad_accum,
        num_workers=4,
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
            "lr": effective_lr,
            "weight_decay": DEFAULT_WEIGHT_DECAY,
            "lr_warmup_steps": lr_warmup,
            "num_queries": args.num_queries,
            "backbone_freeze": args.freeze_backbone,
            "loss_dice_coefficient": 5.0,
            "loss_class_coefficient": 4.0,
            "loss_no_object_coefficient": 0.2,
        },
        transform_args={
            "image_size": IMAGE_SIZE,
            "random_flip": {
                "horizontal_prob": 0.5,
                "vertical_prob": 0.5,
            },
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
                "prob": 0.5,
                "strength": 1.0,
            },
            "scale_jitter": None,
            "normalize": {
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
            },
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

    print("\n=== 训练完成 ===")
    print(f"最优导出模型: {exported_best}")
    print("提示：可使用以下命令导出为 ONNX 模型:")
    print(f"  python tools/toOnnx_l8_fpc_inst.py --checkpoint {exported_best}")


if __name__ == "__main__":
    main()

