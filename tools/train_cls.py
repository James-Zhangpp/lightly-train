"""焊接图像分类训练（DINOv3 ViT + 线性头，multiclass）。

数据目录（由 tools/mask_to_cls.py 生成）：
  weldingcls-data/
    train/{no_fpc,fpc}/*.png
    val/{no_fpc,fpc}/*.png

训练产物：
  out/.../exported_models/exported_best.pt   # 验证 top-1 最高
  out/.../exported_models/exported_last.pt
  out/.../checkpoints/last.ckpt

用法：
  python tools/mask_to_cls.py
  python tools/train_cls.py
"""

import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

import lightly_train

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(__file__).resolve().parent / "weldingcls-data"
OUT_DIR = PROJECT_ROOT / "out" / "welding_dinov3_vitt16_cls_v1"
EXPORTED_BEST = OUT_DIR / "exported_models" / "exported_best.pt"

CLASSES = {
    0: "no_fpc",
    1: "fpc",
}

FINE_TUNE_FROM_BEST = EXPORTED_BEST.is_file()
STEPS = 3000 if FINE_TUNE_FROM_BEST else 8000
LR_WARMUP = (250, 500) if FINE_TUNE_FROM_BEST else (500, 1000)

if __name__ == "__main__":
    train_dir = DATA_ROOT / "train"
    val_dir = DATA_ROOT / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"分类数据不存在，请先运行: python tools/mask_to_cls.py\n"
            f"  期望: {train_dir}\n  期望: {val_dir}"
        )

    lightly_train.train_image_classification(
        out=OUT_DIR,
        model="dinov3/vitt16",
        classification_task="multiclass",
        data={
            "train": train_dir,
            "val": val_dir,
            "classes": CLASSES,
        },
        steps=STEPS,
        batch_size=32,
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
            "weight_decay": 0.01,
            "lr_warmup_steps": LR_WARMUP,
            "backbone_freeze": False,
        },
        transform_args={
            "image_size": (384, 384),
            "normalize": {
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
            },
            "color_jitter": {
                "prob": 0.5,
                "strength": 1.0,
                "brightness": 32.0 / 255.0,
                "contrast": 0.5,
                "saturation": 0.5,
                "hue": 18.0 / 360.0,
            },
            "random_flip": {
                "horizontal_prob": 0.5,
                "vertical_prob": 0.0,
            },
        },
        save_checkpoint_args={
            "save_every_num_steps": 1000,
            "save_last": True,
            "save_best": True,
        },
    )
