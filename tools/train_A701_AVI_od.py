"""焊接 FPC 目标检测训练脚本（ltdetrv2-l-coco · A701_AVI_wpoint_detect 专用）。

数据目录示例：
  DataSet/A701_AVI_wpoint_detect/
    images/{train,val}/*.png
    labels/{train,val}/*.txt
    data.yaml
"""

import os
import glob
import argparse
from pathlib import Path

# 抑制 TensorFlow oneDNN 提示
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import lightly_train

def parse_args():
    parser = argparse.ArgumentParser(description="工业视觉目标检测训练脚本 - A701_AVI 专用")
    
    parser.add_argument("--model", type=str, default="ltdetrv2-l-coco", help="模型架构名称")
    parser.add_argument("--data-dir", type=str, default="DataSet/A701_AVI_wpoint_detect", help="数据集目录名称（需放在 tools 目录下）")
    parser.add_argument("--batch-size", type=int, default=4, help="全局 batch size")
    parser.add_argument("--size", type=int, default=640, help="训练与裁剪图像的目标尺寸")
    parser.add_argument("--epochs", type=int, default=60, help="首轮 Scratch 目标训练 Epoch 数")
    parser.add_argument("--mode", type=str, default="scratch", choices=["scratch", "finetune", "resume"], 
                        help="训练模式: scratch(从头训练), finetune(微调), resume(断点续训)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # ---------------------------------------------------------------------------
    # 1. 路径自动解析与生成
    # ---------------------------------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    TOOLS_DIR = Path(__file__).resolve().parent
    DATA_ROOT = TOOLS_DIR / args.data_dir
    
    model_suffix = args.model.split('/')[-1].replace('-', '_')
    data_suffix = args.data_dir.split('/')[-1]
    OUT_DIR = PROJECT_ROOT / "out" / f"{data_suffix}_{model_suffix}"
    EXPORTED_BEST = OUT_DIR / "exported_models" / "exported_best.pt"
    
    # ---------------------------------------------------------------------------
    # 2. 训练模式控制
    # ---------------------------------------------------------------------------
    TRAIN_MODE = args.mode.lower()
    FINE_TUNE_FROM_BEST = (TRAIN_MODE == "finetune") and EXPORTED_BEST.is_file()
    RESUME_INTERRUPTED = (TRAIN_MODE == "resume")
    
    # ---------------------------------------------------------------------------
    # 3. 动态扫描数据集并计算计算 Steps / Warmup
    # ---------------------------------------------------------------------------
    train_img_dir = DATA_ROOT / "images" / "train"
    train_images = glob.glob(str(train_img_dir / "*.*"))
    num_images = len([f for f in train_images if Path(f).is_file()])
    
    if num_images == 0:
        raise ValueError(f"❌ 找不到训练图片！请检查路径: {train_img_dir}")
        
    steps_per_epoch = max(1, num_images / args.batch_size)
    
    # 微调时缩短 Epoch
    target_epochs = 30 if FINE_TUNE_FROM_BEST else args.epochs
    STEPS = int(steps_per_epoch * target_epochs)
    
    warmup_epochs = max(1, int(target_epochs * 0.05)) 
    LR_WARMUP = int(warmup_epochs * steps_per_epoch)
    
    LR = 5e-5 if FINE_TUNE_FROM_BEST else 1e-4
    
    val_interval = 50  # 根据您的要求，固定为 50 步验证一次

    # ---------------------------------------------------------------------------
    # 4. 打印当前任务配置面板
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print(f"🚀 开始动态自适应训练 - A701_AVI 专用 (目标检测)")
    print(f"📦 数据集:   {args.data_dir} (共找到 {num_images} 张样本)")
    print(f"🤖 模型:     {args.model}")
    print(f"📏 图像尺寸: {args.size} x {args.size}")
    print(f"🧮 训练模式: {TRAIN_MODE.upper()} | Batch: {args.batch_size}")
    print(f"⏱️ 训练跨度: {steps_per_epoch:.1f} steps/epoch * {target_epochs} epochs = 核心执行 {STEPS} 步")
    print(f"🔥 热身区间: {LR_WARMUP} steps | 每 {val_interval} 步验证一次")
    print(f"📂 输出目录: {OUT_DIR}")
    print("=" * 60)

    # ---------------------------------------------------------------------------
    # 5. 启动底层训练框架
    # ---------------------------------------------------------------------------
    lightly_train.train_object_detection(
        out=OUT_DIR,
        model=args.model,
        data={
            "format": "yolo",
            "path": str(DATA_ROOT),
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "HD",
            },
            "skip_if_label_file_missing": True,
        },
        steps=STEPS,
        batch_size=args.batch_size,
        num_workers=4,
        devices="auto",
        accelerator="auto",
        strategy="auto",
        precision="bf16-mixed",
        float32_matmul_precision="auto",
        num_nodes=1,
        seed=42,
        overwrite=(not FINE_TUNE_FROM_BEST) and (not RESUME_INTERRUPTED),
        resume_interrupted=RESUME_INTERRUPTED,
        checkpoint=EXPORTED_BEST if FINE_TUNE_FROM_BEST else None,
        model_args={
            "lr": LR,
            "weight_decay": 1e-4,
            "lr_warmup_steps": LR_WARMUP,
            "scheduler_name": "linear",
        },
        transform_args={
            "image_size": (args.size, args.size),
            "resize": {
                "height": args.size,
                "width": args.size,
                "keep_aspect_ratio": True,
                "pad_value": (114, 114, 114),
            },
            "normalize": {
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
            },
        },
        metric_args={
            "classwise": True,
        },
        loader_args=None,
        logger_args={
            "val_every_num_steps": val_interval,
        },
        save_checkpoint_args={
            "save_every_num_steps": val_interval,
            "save_last": True,
            "save_best": True,
        },
        torch_compile_args=None,
        gradient_accumulation_steps="auto",
    )

if __name__ == "__main__":
    main()
