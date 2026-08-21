"""焊接语义分割训练脚本（DINOv3 + EoMT · A07_01_AVI 专用）。

已配置针对 A07_01_AVI 的参数，支持命令行传参、自动统计图片数量并计算 Steps。
"""

import os
import glob
import argparse
from pathlib import Path

# 抑制 TensorFlow oneDNN 提示（TensorBoard 等间接依赖可能拉起 TF）
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import lightly_train

def parse_args():
    parser = argparse.ArgumentParser(description="工业视觉语义分割训练脚本 - A07_01_AVI 专用")
    
    # 核心可调参数
    parser.add_argument("--model", type=str, default="dinov3/vitb16-eupe-eomt", help="模型架构名称")
    parser.add_argument("--data-dir", type=str, default="DataSet/A07_01_AVI", help="数据集目录名称（需放在 tools 目录下）")
    parser.add_argument("--batch-size", type=int, default=4, help="全局 batch size (大模型 86M 建议 4~8)")
    parser.add_argument("--size", type=int, default=512, help="训练与裁剪图像的目标尺寸 (基于 1190x611 推荐使用 512)")
    parser.add_argument("--epochs", type=int, default=60, help="首轮 Scratch 目标训练 Epoch 数 (默认 60)")
    parser.add_argument("--mode", type=str, default="scratch", choices=["scratch", "finetune", "resume"], 
                        help="训练模式: scratch(从头训练), finetune(微调), resume(断点续训)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # ---------------------------------------------------------------------------
    # 1. 路径自动解析与生成
    # ---------------------------------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_ROOT = Path(__file__).resolve().parent / args.data_dir
    
    # 动态生成输出目录
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
    train_img_dir = DATA_ROOT / "img_dir" / "train"
    train_images = glob.glob(str(train_img_dir / "*.*"))
    num_images = len([f for f in train_images if Path(f).is_file()])
    
    if num_images == 0:
        raise ValueError(f"❌ 找不到训练图片！请检查路径: {train_img_dir}")
        
    steps_per_epoch = max(1, num_images / args.batch_size)
    
    # 微调时缩短 Epoch
    target_epochs = 20 if FINE_TUNE_FROM_BEST else args.epochs
    STEPS = int(steps_per_epoch * target_epochs)
    
    # 动态计算热身步数 (设定预热期约占总 Epoch 的 5%~10%)
    warmup_epochs = max(1, int(target_epochs * 0.08)) 
    warmup_steps_start = int(warmup_epochs * 0.5 * steps_per_epoch)
    warmup_steps_end = int(warmup_epochs * 2.0 * steps_per_epoch)
    LR_WARMUP = (max(10, warmup_steps_start), max(20, warmup_steps_end))
    
    LR = 3e-5 if FINE_TUNE_FROM_BEST else 6e-5
    
    # 动态计算验证/保存频率 (每 2 个 Epoch 执行一次)
    val_interval = max(50, int(steps_per_epoch * 2.0))

    # ---------------------------------------------------------------------------
    # 4. 打印当前任务配置面板
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print(f"🚀 开始动态自适应训练 - A07_01_AVI 专用")
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
    lightly_train.train_semantic_segmentation(
        out=OUT_DIR,
        model=args.model,
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
                1: "target", # 修改为目标类别
            },
        },
        steps=STEPS,
        batch_size=args.batch_size,
        num_workers=2,
        devices="auto",
        accelerator="auto",
        strategy="auto",
        precision="bf16-mixed",
        float32_matmul_precision="high",
        num_nodes=1,
        seed=42,
        overwrite=(not FINE_TUNE_FROM_BEST) and (not RESUME_INTERRUPTED),
        resume_interrupted=RESUME_INTERRUPTED,
        checkpoint=EXPORTED_BEST if FINE_TUNE_FROM_BEST else None,
        
        model_args={
            "lr": LR,
            "llrd": 0.72,
            "weight_decay": 0.08,
            "num_queries": 18,
            "lr_warmup_steps": LR_WARMUP,
            "fix_num_upscale_blocks": False,
            "loss_num_points": 16384,
            "loss_oversample_ratio": 3.0,
            "loss_importance_sample_ratio": 0.85,
            "loss_class_coefficient": 3.0,
            "loss_mask_coefficient": 6.0,
            "loss_dice_coefficient": 12.0,
            "loss_no_object_coefficient": 0.05,
        },
        
        transform_args={
            "image_size": (args.size, args.size),
            "num_channels": 3,
            "color_jitter": {
                "prob": 0.5,
                "strength": 0.8,
                "brightness": 15.0 / 255.0,
                "contrast": 0.15,
                "saturation": 0.2,
                "hue": 6.0 / 360.0,
            },
            "scale_jitter": {
                "sizes": None,
                "min_scale": 0.88,
                "max_scale": 1.12,
                "num_scales": 20,
                "prob": 0.6,
                "divisible_by": None,
            },
            "random_crop": {
                "height": args.size,
                "width": args.size,
                "pad_if_needed": True,
                "pad_position": "random",  
                "fill": 0,
                "prob": 1.0,
            },
            "random_flip": {
                "horizontal_prob": 0.5,
                "vertical_prob": 0.5,
            },
            "random_rotate_90": {
                "prob": 0.75,
            },
            "random_rotate": None,
            "normalize": {
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
            },
        },
        
        loader_args=None,
        logger_args={
            "val_every_num_steps": val_interval,  
        },
        metric_args={
            "classwise": True,
            "watch_metric": "val_metric/miou",
        },
        save_checkpoint_args={
            "save_every_num_steps": val_interval,
            "save_last": True,
            "save_best": True,
        },
    )

if __name__ == "__main__":
    main()
