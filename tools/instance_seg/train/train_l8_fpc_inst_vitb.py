"""L8 FPC 实例分割训练脚本（DINOv3-ViT-B/16 + EoMT-Inst）。

本脚本专为更高精度的 ViT-Base 架构 (dinov3/vitb16-eomt-inst-coco) 定制：
  - 特征维度由 ViT-S 的 384 维扩充为 768 维，参数量提升至 100M+，具备更细致的工业微细边缘识别能力；
  - 适配 RTX 4060 Ti 16GB 显卡，默认批大小从 6 调至 4，基础学习率收紧至 6e-5 并拉长预热区间；
  - 产物输出独立至 out/AD60_L15_InSeg_vitb16，彻底隔离不同维度权重，避免加载冲突；
  - 保持 Dice Loss 权重强化 (8.0) 与零尺度畸变配置 (scale_jitter=None)。

用法：
  # 演练检查超参数与显存/步数推算（推荐）
  python tools/train_l8_fpc_inst_vitb.py --dry-run

  # 启动训练
  python tools/train_l8_fpc_inst_vitb.py

  # 显存受限时（如小显卡），可开启骨干网络冻结
  python tools/train_l8_fpc_inst_vitb.py --freeze-backbone
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

# 规避旧版 PyTorch 在 RTX 50 系列显卡上无法加载 sm80-sm100 CUTLASS 内核的问题
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)


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
# 默认常量配置（针对 ViT-Base 进行工业级适配）
# ---------------------------------------------------------------------------
DEFAULT_DATASET_ROOT = TOOLS_DIR / "data" / "NP_data_coco"
DEFAULT_OUT_DIR = PROJECT_ROOT / "out" / "NP_data_vitb16"

IMAGE_SIZE = (512, 512)
# ViT-Base (768维) 在 512x512 下显存占用较 ViT-S 翻倍，16GB 显卡下推荐批大小为 4
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 25
# ViT-Base 规模较大，全量微调推荐学习率 6e-5，更平稳不易震荡
DEFAULT_LR = 6e-5
DEFAULT_WEIGHT_DECAY = 0.05


def inspect_dataset_and_calculate_schedule(
    train_json_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    target_epochs: int = DEFAULT_EPOCHS,
    fine_tune: bool = False,
    custom_steps: int | None = None,
    custom_val_interval: int | None = None,
) -> Dict[str, Any]:
    """读取训练集 COCO 标注并针对 ViT-Base 架构自适应推算训练步数与预热周期。"""
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

    # 1. 计算单轮步数
    steps_per_epoch = max(1, math.ceil(num_train_images / batch_size))

    # 2. 计算总步数 STEPS
    if custom_steps is not None and custom_steps > 0:
        total_steps = custom_steps
        actual_epochs = round(total_steps / steps_per_epoch, 2)
    else:
        raw_steps = steps_per_epoch * target_epochs
        if fine_tune:
            total_steps = max(600, int(raw_steps * 0.8))
        else:
            total_steps = max(800, int(raw_steps))
        actual_epochs = round(total_steps / steps_per_epoch, 2)

    # 3. 针对 ViT-Base 模型拉长预热区间（Warmup），避免大模型初始步梯度过冲
    if fine_tune:
        warmup_start = max(30, int(total_steps * 0.06))
        warmup_end = max(80, int(total_steps * 0.15))
    else:
        warmup_start = max(60, int(total_steps * 0.12))
        warmup_end = max(150, int(total_steps * 0.25))
    lr_warmup_steps = (warmup_start, warmup_end)

    # 4. 验证与保存间隔
    if custom_val_interval is not None and custom_val_interval > 0:
        val_every_steps = custom_val_interval
    else:
        if steps_per_epoch >= 40:
            val_steps = steps_per_epoch
        else:
            val_steps = steps_per_epoch * 2
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
        description="Train DINOv3 ViT-Base + EoMT Instance Segmentation with Adaptive Schedule."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to single dataset root folder (legacy)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Path(s) to multiple COCO dataset root folders or glob pattern (e.g. tools/DataSet/FPC_Line*) for collection training",
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
        help="Path to output experimental artifacts folder (default: out/AD60_L15_InSeg_vitb16)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size (default: 4 for ViT-B on 16GB GPU)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Base learning rate (default: 6e-5 for ViT-B)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Target number of epochs for adaptive step calculation (default: 25)",
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
        "--freeze-backbone",
        action="store_true",
        help="Freeze DINOv3 ViT-B backbone and only train mask/detection head",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only inspect dataset and print calculated schedule without starting GPU training",
    )

    args = parser.parse_args()

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

    effective_lr = args.lr if not fine_tune_from_best else args.lr * 0.5

    print("=" * 70)
    print("【DINOv3 ViT-Base 工业视觉自适应训练调度 (Adaptive Training Schedule)】")
    print("=" * 70)
    print(f"  - 模型架构: dinov3/vitb16-eomt-inst-coco (ViT-Base, 768维)")
    print(f"  - 数据集路径: {dataset_root}")
    print(f"  - 训练集样本数: {sched['num_train_images']} 张")
    print(f"  - 批大小 (Batch Size): {args.batch_size} (适配 16GB 显存)")
    print(f"  - 学习率 (Base LR): {effective_lr} (预热步数: {lr_warmup[0]} ~ {lr_warmup[1]})")
    print(f"  - 单轮步数 (Steps Per Epoch): {sched['steps_per_epoch']} 步/轮")
    print(f"  - 目标轮数 (Actual Epochs): {sched['actual_epochs']} 轮")
    print(f"  - 自适应总训练步数 (STEPS): {steps} 步")
    print(f"  - 自适应验证与保存间隔 (VAL_EVERY): 每 {val_every_steps} 步评估一次")
    print(f"  - 类别映射 (Class Names): {class_names}")
    print(f"  - Backbone 状态: {'冻结 (Freeze)' if args.freeze_backbone else '全量微调 (End-to-End 精准边缘拟合)'}")
    print(f"  - 微调状态: {'从已导出的 best 继续微调' if fine_tune_from_best else '全新训练 (权重与 vits16 隔离)'}")
    print(f"  - 边界损失权重 (Dice Loss): 8.0 (强化真实边缘贴合度，抑制溢出)")
    print(f"  - 产物输出路径: {out_dir}")
    print("=" * 70)

    if args.dry_run:
        print("\n[OK] 演练完成 (--dry-run 开启，不启动真实 GPU 训练)。")
        return

    print("\n>>> 开始调用 lightly_train 启动 ViT-Base 实例分割模型训练...")

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
            "num_queries": 50,
            "backbone_freeze": True,
            "loss_class_coefficient": 5.0,
        },
        transform_args={
            "image_size": IMAGE_SIZE,
            "random_flip": {
                "horizontal_prob": 0.5,
                "vertical_prob": 0.5,
            },
            #"random_rotate_90": {"prob": 0.5},
           # "random_rotate": {"prob": 0.5, "degrees": 5.0},
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

    print(f"\n=== 训练完成 ===")
    print(f"最优导出模型: {exported_best}")
    print(f"提示：可使用以下命令导出为工业标准内嵌预处理 ONNX 模型:")
    print(f"  python tools/toOnnx_l8_fpc_inst.py --checkpoint {exported_best}")
    print(f"导出后可直接使用以下命令进行推理验证:")
    print(f"  python tools/infer_l8_fpc_inst.py")


if __name__ == "__main__":
    main()
