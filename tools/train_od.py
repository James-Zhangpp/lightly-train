"""焊接 FPC 目标检测训练脚本（DINOv3 + LT-DETR，COCO 格式）。

数据由 tools/mask_to_coco.py 生成，目录示例：
  weldingseg-coco/
    images/{train,val}/*.png
    annotations/instances_{train,val}.json   # category_id=1 -> fpc

训练产物：
  out/.../exported_models/exported_best.pt
  out/.../exported_models/exported_last.pt
  out/.../checkpoints/last.ckpt

默认超参见 DINOv3LTDETRObjectDetectionTrainArgs（dinov3_ltdetr_object_detection/train_model.py）。
注释中「默认」指框架默认，非本脚本当前取值。
"""

import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

import lightly_train

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COCO_ROOT = Path(__file__).resolve().parent / "weldingseg-coco"
OUT_DIR = PROJECT_ROOT / "out" / "welding_dinov3_vits16_ltdetr_od_v1"
EXPORTED_BEST = OUT_DIR / "exported_models" / "exported_best.pt"

TRAIN_ANN = COCO_ROOT / "annotations" / "instances_train.json"
VAL_ANN = COCO_ROOT / "annotations" / "instances_val.json"
TRAIN_IMAGES = COCO_ROOT / "images" / "train"
VAL_IMAGES = COCO_ROOT / "images" / "val"

# 训练图像约 1336 张有标注（batch=10 时约 134 step/epoch）
BATCH_SIZE = 8

# ---------------------------------------------------------------------------
# 训练阶段（存在 exported_best 时微调，否则全量）
# ---------------------------------------------------------------------------
FINE_TUNE_FROM_BEST = EXPORTED_BEST.is_file()
# 全量约 90 epoch；微调约 30 epoch（按有标注样本估算）
STEPS = 4000 if FINE_TUNE_FROM_BEST else 12000
LR_WARMUP = 500 if FINE_TUNE_FROM_BEST else 1000

if __name__ == "__main__":
    if not TRAIN_ANN.is_file() or not VAL_ANN.is_file():
        raise FileNotFoundError(
            f"COCO 标注不存在，请先运行: python tools/mask_to_coco.py\n"
            f"  期望: {TRAIN_ANN}\n  期望: {VAL_ANN}"
        )

    lightly_train.train_object_detection(
        out=OUT_DIR,
        model="dinov3/vits16-ltdetr-coco",
        data={
            "format": "coco",
            "train": {
                "annotations": TRAIN_ANN,
                "images": TRAIN_IMAGES,
            },
            "val": {
                "annotations": VAL_ANN,
                "images": VAL_IMAGES,
            },
            # 跳过无 bbox 的图（mask 全背景）；默认 False 会报错
            "skip_if_annotations_missing": True,
        },
        # 默认 "auto" -> 约 450k step，焊接小数据集必须显式指定
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
            # 学习率。默认 1e-4
            "lr": 5e-5 if FINE_TUNE_FROM_BEST else 1e-4,
            # 权重衰减。默认 1e-4
            "weight_decay": 1e-4,
            # 预热步数。默认 2000
            "lr_warmup_steps": LR_WARMUP,
            # 学习率调度。默认 "linear"，可选 "flat-cosine"
            "scheduler_name": "linear",
        },
        transform_args={
            # 与焊接裁剪图一致；默认 "auto" -> 640×640（patch 对齐后）
            "image_size": (512, 512),
            "normalize": {
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
            },
        },
        metric_args={
            # 按验证 mAP 保存 best。默认 watch_metric=val_metric/map
            "classwise": True,
        },
        loader_args=None,
        logger_args=None,
        save_checkpoint_args={
            "save_every_num_steps": 1000,
            "save_last": True,
            "save_best": True,
        },
        torch_compile_args=None,
        gradient_accumulation_steps="auto",
    )
