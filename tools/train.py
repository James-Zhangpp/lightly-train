import os

# Suppress TensorFlow oneDNN info log (often pulled in via TensorBoard, etc.).
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

import lightly_train


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(__file__).resolve().parent / "weldingseg-data"
OUT_DIR = PROJECT_ROOT / "out" / "welding_dinov3_vits16_eomt_lr1e4_q50"
EXPORTED_BEST = OUT_DIR / "exported_models" / "exported_best.pt"

# Resume modes (pick one):
# - Fresh / new lr fine-tune: resume_interrupted=False, checkpoint=EXPORTED_BEST or None
# - Crash resume (needs checkpoints/last.ckpt): resume_interrupted=True, checkpoint=None


if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out=OUT_DIR,
        model="dinov3/vits16-eomt",
        data={
            "train": {
                "images": DATA_ROOT / "img_dir" / "train",
                "masks": DATA_ROOT / "ann_dir" / "train",
            },
            "val": {
                "images": DATA_ROOT / "img_dir" / "val",
                "masks": DATA_ROOT / "ann_dir" / "val",
            },
            "classes": {
                0: "background",
                1: "fpc",
            },
        },
        # ~127 steps/epoch (1271 train images, batch 10) -> 20000 steps ~ 157 epochs.
        # Use exported_best.pt if val mIoU plateaus before 20k.
        steps=20000,
        batch_size=10,
        num_workers=6,
        devices="auto",
        accelerator="auto",
        strategy="auto",
        precision="bf16-mixed",
        float32_matmul_precision="auto",
        num_nodes=1,
        seed=42,
        overwrite=True,
        resume_interrupted=False,
        checkpoint=EXPORTED_BEST if EXPORTED_BEST.is_file() else None,
        model_args={
            # Lower lr if val mIoU drops while train loss still falls (overfitting).
            "lr": 5e-5,
            "weight_decay": 0.01,
            # The task is binary and usually has one dominant foreground region.
            "num_queries": 50,
            "lr_warmup_steps": (250, 500),
        },
        transform_args={
            "image_size": (512, 512),
            "num_channels": 3,
            "scale_jitter": {
                "sizes": None,
                "min_scale": 0.75,
                "max_scale": 1.25,
                "num_scales": 10,
                "prob": 1.0,
                "divisible_by": None
            },
            "random_crop": {
                "height": 512,
                "width": 512,
                "pad_if_needed": True,
                "pad_position": "center",
                "fill": 0,
                "prob": 1.0,
            },
            "random_flip": {
                "horizontal_prob": 0.5,
                "vertical_prob": 0.0,
            },
            "random_rotate": {
                "prob": 0.2,
                "degrees": 10,
            },
            "normalize": {
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
            },
        },
        loader_args=None,
        logger_args=None,
        save_checkpoint_args={
            "save_every_num_steps": 1000,
            "save_last": True,
            "save_best": True,
        },
    )
