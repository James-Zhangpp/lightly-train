"""L8 FPC 实例分割训练脚本（DINOv3 + EoMT-Inst）。

数据集路径：
  tools/DataSet/L8_FPC_InSeg_coco/
    train/
      L8_xxx.png
      _annotations.coco.json
    valid/
      L8_xxx.png
      _annotations.coco.json
    test/
      L8_xxx.png
      _annotations.coco.json

输出模型目录：
  out/l8_fpc_dinov3_vits16_eomt_inst/
    checkpoints/last.ckpt, best.ckpt
    exported_models/exported_best.pt, exported_last.pt

用法：
  python tools/train_l8_fpc_inst.py
"""

from __future__ import annotations

import os
from pathlib import Path

# 关闭 oneDNN 浮点警告与降低 TF 日志等级（若环境中安装了 TensorFlow 依赖）
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import lightly_train

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent

# 数据集根目录
DATASET_ROOT = TOOLS_DIR / "DataSet" / "FPC_Line10_Seg_coco_square"
TRAIN_JSON = DATASET_ROOT / "train" / "_annotations.coco.json"
TRAIN_IMG_DIR = DATASET_ROOT / "train"
VAL_JSON = DATASET_ROOT / "valid" / "_annotations.coco.json"
VAL_IMG_DIR = DATASET_ROOT / "valid"

# 实验产物输出目录
OUT_DIR = PROJECT_ROOT / "out" / "l10_fpc_square"
EXPORTED_BEST = OUT_DIR / "exported_models" / "exported_best.pt"

# 类别定义（FPC_Line4_Seg_coco 数据集中的 categories id: 1 -> FPC）
CLASS_NAMES = {
    1: "FPC",
}

# ---------------------------------------------------------------------------
# 训练超参数配置
# ---------------------------------------------------------------------------
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 8
FINE_TUNE_FROM_BEST = EXPORTED_BEST.is_file()
STEPS = 3000 if FINE_TUNE_FROM_BEST else 5000
LR_WARMUP = (300, 600) if FINE_TUNE_FROM_BEST else (800, 1200)


def validate_dataset_exists() -> None:
    """检查数据集标注和图片目录是否存在。"""
    if not TRAIN_JSON.is_file():
        raise FileNotFoundError(f"训练集 COCO 标注文件不存在: {TRAIN_JSON}")
    if not VAL_JSON.is_file():
        raise FileNotFoundError(f"验证集 COCO 标注文件不存在: {VAL_JSON}")
    if not TRAIN_IMG_DIR.is_dir():
        raise FileNotFoundError(f"训练集图像目录不存在: {TRAIN_IMG_DIR}")
    if not VAL_IMG_DIR.is_dir():
        raise FileNotFoundError(f"验证集图像目录不存在: {VAL_IMG_DIR}")


def main() -> None:
    validate_dataset_exists()

    print(f"=== 开始 L8 FPC 实例分割模型训练 ===")
    print(f"数据集目录: {DATASET_ROOT}")
    print(f"训练集 JSON: {TRAIN_JSON}")
    print(f"验证集 JSON: {VAL_JSON}")
    print(f"输出路径: {OUT_DIR}")
    print(f"输入图像尺寸: {IMAGE_SIZE}")
    print(f"微调状态: {'从已导出的 best 继续微调' if FINE_TUNE_FROM_BEST else '全新训练'}")
    print(f"总训练步数: {STEPS}, Batch Size: {BATCH_SIZE}")

    lightly_train.train_instance_segmentation(
        out=OUT_DIR,
        model="dinov3/vits16-eomt-inst-coco",
        data={
            "format": "coco",
            "train": {
                "annotations": str(TRAIN_JSON),
                "images": str(TRAIN_IMG_DIR),
            },
            "val": {
                "annotations": str(VAL_JSON),
                "images": str(VAL_IMG_DIR),
            },
            "skip_if_annotations_missing": False,
        },
        steps=STEPS,
        batch_size=BATCH_SIZE,
        num_workers=4,
        devices="auto",
        accelerator="auto",
        strategy="auto",
        precision="bf16-mixed",
        float32_matmul_precision="auto",
        num_nodes=1,
        seed=42,
        overwrite=not FINE_TUNE_FROM_BEST,
        resume_interrupted=False,
        checkpoint=EXPORTED_BEST if FINE_TUNE_FROM_BEST else None,
        model_args={
            "lr": 1e-4 if not FINE_TUNE_FROM_BEST else 5e-5,
            "weight_decay": 0.05,
            "lr_warmup_steps": LR_WARMUP,
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
            "random_rotate_90": {"prob": 0.5},
            "random_rotate": {"prob": 0.5, "degrees": 15.0},
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
                "prob": 0.5,
                "strength": 1.0,
            },
            "scale_jitter": {
                "sizes": None,
                "min_scale": 0.8,
                "max_scale": 1.2,
                "num_scales": 5,
                "prob": 0.8,
                "divisible_by": None,
            },
            "normalize": {
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
            },
        },
        metric_args={
            "classwise": True,
        },
        save_checkpoint_args={
            "save_every_num_steps": 500,
            "save_last": True,
            "save_best": True,
        },
    )

    print(f"=== 训练完成 ===")
    print(f"最优导出模型: {EXPORTED_BEST}")
    print(f"提示：可使用以下命令导出为工业标准内嵌预处理 ONNX 模型:")
    print(f"  python tools/toOnnx_l8_fpc_inst.py --checkpoint {EXPORTED_BEST}")
    print(f"导出后可直接使用以下命令进行推理验证:")
    print(f"  python tools/infer_l8_fpc_inst.py")


if __name__ == "__main__":
    main()
