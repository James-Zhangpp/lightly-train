"""焊接 FPC 实例分割训练（DINOv3 + EoMT-Inst）。

数据由 tools/mask_to_coco.py 生成（YOLO-seg 格式）：
  weldingseg-coco/
    images/{train,val}/*.png
    labels/{train,val}/*.txt      # 每行一个实例 polygon，class 0 = fpc
    annotations/instances_*.json  # 可选，COCO 格式备份

训练产物：
  out/.../exported_models/exported_best.pt
  out/.../exported_models/exported_last.pt

用法：
  python tools/mask_to_coco.py
  python tools/train_inst.py
"""

import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

import lightly_train

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = Path(__file__).resolve().parent

# YOLO 数据根目录（优先 DataSet/）
def _resolve_coco_root() -> Path:
    for candidate in (
        _TOOLS_DIR / "DataSet" / "weldingseg-yolo",
        _TOOLS_DIR / "weldingseg-yolo",
    ):
        if (candidate / "labels" / "train").is_dir():
            return candidate
    return _TOOLS_DIR / "DataSet" / "weldingseg-yolo"


COCO_ROOT = _resolve_coco_root()
OUT_DIR = PROJECT_ROOT / "out" / "welding_dinov3_vits16_eomt_inst_v1"
EXPORTED_BEST = OUT_DIR / "exported_models" / "exported_best.pt"

# YOLO txt 中的类别
CLASS_NAMES = {
    0: "ignore",
    1: "OK",
    2: "NG"
}

BATCH_SIZE = 4
FINE_TUNE_FROM_BEST = EXPORTED_BEST.is_file()
STEPS = 4000 if FINE_TUNE_FROM_BEST else 12000
LR_WARMUP = (500, 750) if FINE_TUNE_FROM_BEST else (1000, 1500)

if __name__ == "__main__":
    train_labels = COCO_ROOT / "labels" / "train"
    val_labels = COCO_ROOT / "labels" / "val"
    if not train_labels.is_dir() or not val_labels.is_dir():
        raise FileNotFoundError(
            f"YOLO 标注不存在，请先运行: python tools/mask_to_coco.py\n"
            f"  期望: {train_labels}\n  期望: {val_labels}"
        )

    lightly_train.train_instance_segmentation(
        out=OUT_DIR,
        model="dinov3/vits16-eomt-inst-coco",
        data={
            "format": "yolo",
            "path": str(COCO_ROOT),
            "train": "images/train",
            "val": "images/val",
            "names": CLASS_NAMES,
            "ignore_classes": {0},  # 训练时自动忽略 ignore 类别
            "skip_if_label_file_missing": False,
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
            "backbone_freeze": False,
        },
        transform_args={
            "image_size": (512, 512),
            "normalize": {
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
            },
        },
        metric_args={
            "classwise": True,
        },
        save_checkpoint_args={
            "save_every_num_steps": 1000,
            "save_last": True,
            "save_best": True,
        },
    )
