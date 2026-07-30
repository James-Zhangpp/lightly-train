"""工业缺陷语义分割专项训练脚本 (全参数暴露与详细调优注释版)。

本脚本已将模型选型、分辨率、学习率、Loss 权重、数据增强、解码头 Query 等所有关键参数
完全暴露为命令行参数，并附带有工业实战调优效果注释。
"""

import os
import sys
import glob
import logging
import argparse
from pathlib import Path
from PIL import Image

# 抑制 TensorFlow oneDNN 提示
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import lightly_train

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_breakage")

def get_original_full_image_size(image_paths):
    """自动分析数据集的原图真实宽高，并向上微调对齐至 8 的倍数。"""
    sizes = []
    for path in image_paths[:15]:
        try:
            with Image.open(path) as img:
                sizes.append(img.size)  # (width, height)
        except Exception as e:
            logger.warning(f"读取图片尺寸失败 {path}: {e}")
            continue
            
    if not sizes:
        raise ValueError("无法读取训练图像尺寸，请检查 img_dir/train 下的图片是否损坏或格式不支持")
        
    widths, heights = zip(*sizes)
    max_w = int(max(widths))
    max_h = int(max(heights))
    
    aligned_h = ((max_h + 7) // 8) * 8
    aligned_w = ((max_w + 7) // 8) * 8
    
    reason = f"原图实际最大尺寸为 ({max_h}x{max_w})，已向上微调补齐对齐为 8 的倍数: ({aligned_h}x{aligned_w})"
    return (aligned_h, aligned_w), reason

def parse_args():
    parser = argparse.ArgumentParser(
        description="工业缺陷多类别语义分割可调参数训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ---------------------------------------------------------------------------
    # 1. 基础配置与模型选型参数
    # ---------------------------------------------------------------------------
    parser.add_argument("--model", type=str, default="dinov3/vitb16-eomt",
                        help="目标分割模型架构。\n"
                             "可选: 'dinov3/vitb16-eomt' (Base特征，平衡好), \n"
                             "      'dinov3/vitl16-eomt' (Large大模型，特征提取能力极强), \n"
                             "      'segformer/b2' (快速微小缺陷重叠切片), \n"
                             "      'segformer/b5' (高精旗舰重叠切片模型)")
    parser.add_argument("--data-dir", type=str, default="brasn_seg",
                        help="数据集目录名称（需放在 tools/ 目录下，内含 img_dir 和 ann_dir）")
    parser.add_argument("--classes", nargs="+", default=["In", "Pa", "Sc"], 
                        help="异常缺陷类别名称列表 (默认: ['In', 'Pa', 'Sc'])，自动映射 0: background, 1: In, 2: Pa, 3: Sc")
    parser.add_argument("--mode", type=str, default="scratch", choices=["scratch", "finetune", "resume"],
                        help="训练模式: \n"
                             "'scratch'  - 从头开始首轮训练\n"
                             "'finetune' - 基于已导出的 Best 模型继续微调\n"
                             "'resume'   - 因意外中断后断点续训")

    # ---------------------------------------------------------------------------
    # 2. 图像分辨率与尺寸对齐参数
    # ---------------------------------------------------------------------------
    parser.add_argument("--image-width", type=int, default=400,
                        help="训练输入宽度。调大(如400)获得8像素更细物理感受野提升微小点检测；调小(如200)加快训练降低显存")
    parser.add_argument("--image-height", type=int, default=400,
                        help="训练输入高度。调大(如400)获得8像素更细物理感受野提升微小点检测；调小(如200)加快训练降低显存")
    parser.add_argument("--auto-image-size", action="store_true",
                        help="启用后自动按数据集最大原图尺寸对齐到 8/16 倍数，用于原图全尺寸训练")

    # ---------------------------------------------------------------------------
    # 3. 训练跨度与资源配置参数
    # ---------------------------------------------------------------------------
    parser.add_argument("--batch-size", type=int, default=4,
                        help="单次前向传播的 Batch Size。显存不足(OOM)时调小(如2或1)；显存充足时调大(如8)加速训练并稳定梯度")
    parser.add_argument("--epochs", type=int, default=300, 
                        help="总训练 Epoch 数。小数据集建议 150~300 次以充分收敛；数据量大(>2000张)可降至 60~100 次")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader 数据读取子进程数。Windows 推荐 0 或 2(防卡死爆内存)；Linux 可设为 CPU 核心数的一半")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="梯度累加步数 (Gradient Accumulation)。模拟更大 Batch：实际 Batch = batch_size * grad_accum")
    parser.add_argument("--precision", type=str, default="32", choices=["32", "bf16-mixed", "16-mixed"],
                        help="训练硬件精度类型。\n"
                             "  '32'         - FP32 全单精度计算（默认，精度最高最稳定，彻底规避梯度溢出/NaN风险，显存占用稍高）；\n"
                             "  'bf16-mixed' - Bfloat16 混合精度（推荐 RTX30/40系显卡，指数位大不易溢出，大幅省显存并加速训练 1.5~2倍）；\n"
                             "  '16-mixed'   - Float16 混合精度（适合老款 GTX10/20系显卡，省显存提速，但动态范围较小）")

    # ---------------------------------------------------------------------------
    # 4. 优化器与学习率策略参数 (Hyperparameters)
    # ---------------------------------------------------------------------------
    parser.add_argument("--lr", type=float, default=1.0e-4,
                        help="初始学习率 (Learning Rate)。调大(如2e-4)收敛快但可能震荡；微调finetune时建议调小(如4e-5)防止破坏权重")
    parser.add_argument("--llrd", type=float, default=0.85,
                        help="逐层学习率衰减率 (Layer-wise LR Decay)。值越小(如0.75)越保护 Backbone 浅层；值越大(如0.90)深浅层更新越接近")
    parser.add_argument("--weight-decay", type=float, default=0.02,
                        help="权重衰减/L2正则化。调大(如0.05)强力抑制小数据集过拟合；调小(如0.01)增强对复杂样品的拟合能力")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgdm", "adam"],
                        help="优化器算法选型。'adamw' (默认，适合 Transformer 模型的自适应梯度衰减), 'sgdm' (带动量的 SGD，收敛平滑)")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "linear", "polynomial"],
                        help="学习率衰减调度策略。'cosine' (默认，余弦退火，后期平滑收敛最佳), 'linear' (线性衰减), 'polynomial' (多项式衰减)")
    parser.add_argument("--backbone-lr-mult", type=float, default=1.0,
                        help="Backbone 骨干网络独立学习率乘子倍率。设为 0.1 可使 Backbone 以小学习率微调，防止破坏预训练通用特征")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="Mask 概率二值化判定阈值 (0.0~1.0)。调低(如0.3)可提高极微小缺陷的召回率(防止漏检)；调高(如0.7)可降低假阳性误检率")

    # ---------------------------------------------------------------------------
    # 5. 解码头与 Loss 损失函数权重参数 (Loss Coefficients)
    # ---------------------------------------------------------------------------
    parser.add_argument("--num-queries", type=int, default=10,
                        help="EoMT 解码头 Mask Query 数量。工业图像缺陷少时设为 10~20(防止空查询稀释注意力)；密集多缺陷设为 50~100")
    parser.add_argument("--loss-num-points", type=int, default=25000,
                        help="点采样 Loss 采样点数 (Point Sampling)。调高(如25000~50000)显著提高微小白点和细划痕边缘的分割精细度")
    parser.add_argument("--loss-class-weight", type=float, default=5.0,
                        help="分类损失系数 (Class Coefficient)。调高增加类别分类判定的准确率")
    parser.add_argument("--loss-mask-weight", type=float, default=5.0,
                        help="掩码 Focal 损失系数 (Mask Coefficient)。调高强化像素掩码整体形态轮廓学习")
    parser.add_argument("--loss-dice-weight", type=float, default=5.0,
                        help="Dice 边界重叠损失系数 (Dice Coefficient)。调高强力关注预测与真实 Mask 的交并集，显著提升 mIoU")
    parser.add_argument("--loss-no-object-weight", type=float, default=0.10,
                        help="无目标/背景 Query 惩罚权重。调低(如0.05)减少背景对缺陷查询干扰；调高(如0.20)减少假阳性误检")
    parser.add_argument("--loss-oversample-ratio", type=float, default=5.0,
                        help="难点过采样比例 (Oversample Ratio)。调高(如5.0~8.0)在候选点中专门挑选最难区分的边缘/微小缺陷点计算 Loss，专治微小白点")
    parser.add_argument("--loss-importance-ratio", type=float, default=0.85,
                        help="重要性采样比例 (Importance Sample Ratio)。0.85 表示 85%% 的采样点聚焦困难边缘点，15%% 均匀分布全图全局背景")

    # ---------------------------------------------------------------------------
    # 6. 数据增强参数 (Data Augmentations)
    # ---------------------------------------------------------------------------
    parser.add_argument("--pad-position", type=str, default="center", choices=["center", "random"],
                        help="随机裁剪平移模式。'random' 在训练时随机平移窗口，可在多个 Epoch 累加下实现 100% 重叠采样 (Overlap)；'center' 居中对齐")
    parser.add_argument("--color-jitter-prob", type=float, default=0.3,
                        help="颜色色彩抖动触发概率。工业微小亮度缺陷建议调低(如0.3)，防止色彩过扰破坏白点/划痕与背景的对比度差")
    parser.add_argument("--rotate-90-prob", type=float, default=0.0,
                        help="90/180/270度正交随机旋转概率 (0.0~1.0)。若工件表面缺陷方向无固定规律可设为 0.5，增加旋转泛化")
    parser.add_argument("--gaussian-blur-prob", type=float, default=0.0,
                        help="高斯模糊触发概率 (0.0~1.0)。设为 0.2 可以模拟现场镜头灰尘、油污或焦距微漂抖动，增强鲁棒性")
    parser.add_argument("--scale-jitter-prob", type=float, default=0.0,
                        help="多尺度随机缩放抖动概率 (0.0~1.0)。0.0 表示禁用(保持1:1物理切片精确对齐)；0.5 表示启用随机缩放")

    return parser.parse_args()

def main():
    try:
        args = parse_args()
        
        # ---------------------------------------------------------------------------
        # 1. 路径自动解析
        # ---------------------------------------------------------------------------
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        DATA_ROOT = Path(__file__).resolve().parent / args.data_dir
        
        model_suffix = args.model.split('/')[-1].replace('-', '_')
        OUT_DIR = PROJECT_ROOT / "out" / f"{args.data_dir}_{args.image_height}x{args.image_width}_{model_suffix}"
        EXPORTED_BEST = OUT_DIR / "exported_models" / "exported_best.pt"
        
        # ---------------------------------------------------------------------------
        # 2. 训练模式控制
        # ---------------------------------------------------------------------------
        TRAIN_MODE = args.mode.lower()
        FINE_TUNE_FROM_BEST = (TRAIN_MODE == "finetune") and EXPORTED_BEST.is_file()
        RESUME_INTERRUPTED = (TRAIN_MODE == "resume")
        
        # ---------------------------------------------------------------------------
        # 3. 扫描图片并自动获取原图尺寸
        # ---------------------------------------------------------------------------
        train_img_dir = DATA_ROOT / "img_dir" / "train"
        train_images = [Path(f) for f in glob.glob(str(train_img_dir / "*.*")) if Path(f).is_file()]
        num_images = len(train_images)
        
        if num_images == 0:
            raise ValueError(f"❌ 找不到训练图片！请检查数据集路径是否存在: {train_img_dir}")
            
        if args.auto_image_size:
            # 自动原图尺寸模式：自动向上对齐到 8 的倍数
            train_image_size, size_reason = get_original_full_image_size(train_images)
        else:
            # 自动将输入宽高对齐到 8 的倍数 (200=8*25, 400=8*50)
            aligned_h = ((args.image_height + 7) // 8) * 8
            aligned_w = ((args.image_width + 7) // 8) * 8
            train_image_size = (aligned_h, aligned_w)
            if (aligned_h, aligned_w) != (args.image_height, args.image_width):
                size_reason = (
                    f"设定的图像尺寸为 {args.image_height}x{args.image_width}，"
                    f"已自动对齐至 8 的倍数: ({aligned_h}x{aligned_w})"
                )
            else:
                size_reason = (
                    f"使用固定训练尺寸 ({aligned_h}x{aligned_w}) [8x8 步长完全对齐]"
                )

        # 构建多类别字典 (0: background, 1: In, 2: Pa, 3: Sc ...)
        classes_dict = {0: "background"}
        for idx, cls_name in enumerate(args.classes, start=1):
            classes_dict[idx] = cls_name
            
        classes_desc = ", ".join([f"{k}: {v}" for k, v in classes_dict.items()])

        # ---------------------------------------------------------------------------
        # 4. 动态计算 Step 与 Warmup 步数
        # ---------------------------------------------------------------------------
        steps_per_epoch = max(1, num_images / args.batch_size)
        target_epochs = 30 if FINE_TUNE_FROM_BEST else args.epochs
        STEPS = int(steps_per_epoch * target_epochs)
        
        warmup_epochs = max(1, int(target_epochs * 0.10)) 
        warmup_steps_start = int(warmup_epochs * 0.5 * steps_per_epoch)
        warmup_steps_end = int(warmup_epochs * 2.0 * steps_per_epoch)
        LR_WARMUP = (max(10, warmup_steps_start), max(20, warmup_steps_end))
        
        # 动态决定学习率
        current_lr = 4.0e-5 if FINE_TUNE_FROM_BEST else args.lr
        # 每 10 个 Epoch 验证评估一次
        val_interval = max(10, int(steps_per_epoch * 10.0))

        # ---------------------------------------------------------------------------
        # 5. 打印详细配置面板
        # ---------------------------------------------------------------------------
        print("=" * 68)
        print(f"🚀 开始【{args.data_dir} {train_image_size[0]}x{train_image_size[1]}】多类别缺陷语义分割训练")
        print(f"📦 数据集路径: {DATA_ROOT} (共扫描到 {num_images} 张原图)")
        print(f"🔍 尺寸分辨率: {size_reason}")
        print(f"🏷️ 缺陷类别:   {classes_desc}")
        print(f"🤖 目标模型:   {args.model}")
        print(f"📏 训练输入:   {train_image_size[0]} x {train_image_size[1]} (H x W)")
        print(f"🧮 训练模式:   {TRAIN_MODE.upper()} | Batch Size: {args.batch_size} | Grad Accum: {args.grad_accum} | Workers: {args.num_workers}")
        print(f"⚡ 硬件精度:   {args.precision} | 初始 LR: {current_lr} | LLRD: {args.llrd} | Weight Decay: {args.weight_decay}")
        print(f"🎯 损失配置:   Queries: {args.num_queries} | Loss Points: {args.loss_num_points} | Dice Weight: {args.loss_dice_weight}")
        print(f"⏱️ 训练跨度:   {steps_per_epoch:.1f} steps/epoch * {target_epochs} epochs = 核心执行 {STEPS} 步")
        print(f"🔥 Warmup区间: {LR_WARMUP} steps | 每 {val_interval} 步验证评估一次")
        print(f"📂 结果保存:   {OUT_DIR}")
        print("=" * 68)

        # ---------------------------------------------------------------------------
        # 6. 启动 semantic segmentation 训练
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
                "classes": classes_dict,
            },
            steps=STEPS,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            devices="auto",
            accelerator="auto",
            strategy="auto",
            precision=args.precision,
            float32_matmul_precision="high",
            num_nodes=1,
            seed=42,
            overwrite=(not FINE_TUNE_FROM_BEST) and (not RESUME_INTERRUPTED),
            resume_interrupted=RESUME_INTERRUPTED,
            checkpoint=EXPORTED_BEST if FINE_TUNE_FROM_BEST else None,
            
            gradient_accumulation_steps=args.grad_accum,
            debug_args=None,
            
            # ---------------------------------------------------------------------------
            # 6. 模型网络与损失函数超参数 (Model & Loss Arguments) - 全参数详细效果注释
            # ---------------------------------------------------------------------------
            model_args={
                # 💡【学习率与优化器控制参数】
                "lr": current_lr,                            # 基础学习率 (Learning Rate)。控制网络权重的更新步长，过大易震荡，过小收敛慢
                "llrd": args.llrd,                           # 逐层学习率衰减率 (Layer-wise LR Decay)。如 0.85，深层学习率大，浅层 Backbone 学习率指数级衰减以保护预训练通用特征
                "weight_decay": args.weight_decay,           # L2 权重衰减 (Regularization)。调大(如0.05)抑制小数据集过拟合；调小(如0.01)提高样品拟合度
                "lr_warmup_steps": LR_WARMUP,                # Warmup 热身步数区间 (start_step, end_step)。在训练初期用极小学习率预热，防止刚开始梯度爆炸
                "optimizer": args.optimizer,                 # 优化器算法。'adamw' (适合 Transformer 自适应梯度), 'sgdm' (带动量 SGD 适合平滑收敛)
                "scheduler": args.scheduler,                 # 学习率退火调度策略。'cosine' (余弦退火，末期平滑收敛最佳), 'linear', 'polynomial'
                "backbone_lr_multiplier": args.backbone_lr_mult, # Backbone 骨干网独立学习率倍率。设为 0.1 可使 Backbone 以极小学习率微调，防止破坏预训练特征
                
                # 💡【解码头架构与 Query 查询参数】
                "num_queries": args.num_queries,             # Mask Query 缺陷查询向量数量。每个 Query 负责预测一个潜在缺陷对象；工业单图缺陷少时设为 10~20，防止过多空 Query 稀释注意力
                "fix_num_upscale_blocks": False,             # 是否固定上采样模块数量。False 表示网络会根据图像输入尺寸 (H, W) 和 Patch Size 自动计算最优的上采样层数
                "mask_threshold": args.mask_threshold,       # Mask 概率二值化阈值 (0.0~1.0)。调低(如0.3)提高极微小缺陷召回率(防漏检)；调高(如0.7)降低假阳性误检
                
                # 💡【微小缺陷点采样 Loss 参数 (Point Sampling for Micro Defects)】
                "loss_num_points": args.loss_num_points,     # 计算 Mask Loss 时随机/重点采样的像素点数量 (默认 25000)。采样点越多，对极细划痕(Sc)和微小白点(Pa)边缘分割越精准
                "loss_oversample_ratio": args.loss_oversample_ratio, # 难点过采样比例 (Oversample Ratio)。在 5*N 个候选点中优先挑选分类难度最大的 N 个点计算 Loss，专治微小白点和锯齿边缘
                "loss_importance_sample_ratio": args.loss_importance_ratio, # 重要性采样混合比例 (0.85)。表示 85% 的采样点集中在困难/边缘点，15% 均匀散布在全局背景点上，平衡局部与全局
                
                # 💡【各损失函数组件的联合加权系数 (Loss Coefficients)】
                "loss_class_coefficient": args.loss_class_weight, # 类别分类 Focal Loss 权重。控制分类头判别 In/Pa/Sc 缺陷类别的正确率权重
                "loss_mask_coefficient": args.loss_mask_weight,   # 像素 Mask 掩码 Focal Loss 权重。控制预测像素二值掩码形态轮廓与真实标注的吻合度
                "loss_dice_coefficient": args.loss_dice_weight,   # Dice 重叠度损失权重。强力约束预测 Mask 与真实 Mask 的交并集 (IoU)，调高可直接提升 mIoU 指标
                "loss_no_object_coefficient": args.loss_no_object_weight, # 无目标/背景 Query 惩罚权重。控制未匹配到缺陷的空 Query 的惩罚力度；调低(如0.05)减少背景干扰，调高(如0.20)减少假阳性误检
            },
            
            # ---------------------------------------------------------------------------
            # 7. 数据增强超参数 (Data Augmentations) - 详细效果注释
            # ---------------------------------------------------------------------------
            transform_args={
                "image_size": train_image_size,  # (Height, Width) 统一缩放后的输入分辨率，决定网络的观察尺寸
                "num_channels": 3,               # 输入图像通道数 (RGB 3通道)
                
                # 💡【色彩/亮度抖动 (Color Jitter)】
                # 工业应用提示：白点、微小划痕核心依赖局部亮度/灰度差。
                # 调高 strength/brightness 会增加环境光变化鲁棒性，但过高会模糊白点与背景的灰度差；
                # 工业微小缺陷场景建议将 prob 控制在 0.3 左右，strength 控制在 0.3 左右。
                "color_jitter": {
                    "prob": args.color_jitter_prob,  # 颜色抖动触发概率。工业镜头光源固定时建议 0.3；如果现场光照不稳定可调高至 0.5
                    "strength": 0.3,                 # 颜色扰动整体强度。建议 0.3，过强会破坏划痕的灰度边缘
                    "brightness": 5.0 / 255.0,       # 亮度变化上限。约 2% 的微小亮度浮动，防止全图过度增亮淹没微小白点
                    "contrast": 0.05,                # 对比度扰动范围。保护低对比度微小凹陷(In)与背景的边缘对比度
                    "saturation": 0.05,              # 饱和度扰动范围。工业灰度/单色相机建议保持极小(0.05)
                    "hue": 2.0 / 360.0,              # 色调微调范围。防止过大幅度色调偏移破坏工件表面真实材质特征
                },
                
                # 💡【多尺度随机缩放抖动 (Scale Jitter)】
                # 工业应用提示：对于固定分辨率(200x200 / 400x400)精细对齐数据集，
                # 禁用尺度抖动(prob=0.0)可以保证模型学习到 1:1 的真实物理缺陷尺寸；
                # 如果相机安装高度不一导致缺陷大小变化，可将 prob 设为 0.5，min_scale=0.8, max_scale=1.2 增强尺寸泛化。
                "scale_jitter": {
                    "sizes": None,
                    "min_scale": 0.8,               # 最小缩放比例 (0.8 表示缩小至 80%)
                    "max_scale": 1.2,               # 最大缩放比例 (1.2 表示放大至 120%)
                    "num_scales": 5,
                    "prob": args.scale_jitter_prob, # 触发概率。0.0 表示禁用，保证 8x8 物理切片精准对齐；0.5 表示开启尺度多变性
                    "divisible_by": 8,
                },
                
                # 💡【随机平移裁剪与填充 (Random Crop & Padding)】
                # 工业应用提示：pad_position 参数极其关键！
                # - 'center': 居中对齐，适合工件永远在图像正中央的场景；
                # - 'random': 随机位置平移，在多个 Epoch 训练累加下可实现 100% 重叠采样 (Overlap)，大幅提升边界识别能力。
                "random_crop": {
                    "height": train_image_size[0],   # 目标裁剪高度
                    "width": train_image_size[1],    # 目标裁剪宽度
                    "pad_if_needed": True,           # 当图像小于目标尺寸时自动填充
                    "pad_position": args.pad_position, # 'center' 居中 / 'random' 随机平移实现滑动重叠采样
                    "fill": 0,                       # 填充像素值，0 代表纯黑背景
                    "prob": 1.0,                     # 确保每张图片都经过尺寸对齐裁剪
                },
                
                # 💡【随机镜像翻转 (Random Flip)】
                # 工业应用提示：无方向限制的表面缺陷（如划痕 Sc、颗粒 Pa、凹陷 In）建议开启 0.5 概率翻转；
                # 翻转可以使有效训练样本量翻倍 (4倍)，极大地增强模型对不同方向缺陷的泛化能力。
                "random_flip": {
                    "horizontal_prob": 0.5,          # 50% 概率水平左右镜像翻转
                    "vertical_prob": 0.5,            # 50% 概率垂直上下镜像翻转
                },
                
                # 💡【正交 90度/180度/270度 随机旋转 (Random Rotate 90)】
                # 工业应用提示：若工件表面缺陷方向无固定规律，设为 rotate_90_prob>0 可使数据集增强 4 倍正交旋转不变性；
                # 若缺陷具有固定方向性（如螺纹、刻字、二维码），请保持 0.0 禁用。
                "random_rotate_90": {
                    "prob": args.rotate_90_prob,     # 0.0 表示禁用正交旋转；设为 0.5 开启正交旋转
                } if args.rotate_90_prob > 0 else None,
                
                # 💡【小角度微旋转 (Random Rotate)】
                # 模拟工件放置微小角度偏转（如 ±15 度）
                "random_rotate": {
                    "degrees": (-15.0, 15.0),        # 旋转角度范围
                    "prob": 0.2,                     # 20% 概率微小偏转
                } if args.rotate_90_prob > 0 else None,
                
                # 💡【高斯模糊/磨砂失焦模拟 (Gaussian Blur)】
                # 工业应用提示：设为 0.2 时，训练时随机对 20% 图片做轻微高斯模糊，
                # 能够模拟生产线上工业相机镜头偶发灰尘、水汽、油污或机械震动带来的失焦模糊，增强模型鲁棒性。
                "gaussian_blur": {
                    "kernel_size": 5,                # 模糊核窗口大小 (5x5 像素)
                    "sigma": (0.1, 2.0),             # 高斯分布标准差范围
                    "prob": args.gaussian_blur_prob, # 触发概率 (0.0 表示禁用，0.2 表示开启防失焦增强)
                } if args.gaussian_blur_prob > 0 else None,
                
                # 💡【ImageNet 标准像素归一化 (Normalize)】
                # 将 0~255 像素缩放归一化为标准的 0~1 正态分布，加速神经网络收敛
                "normalize": {
                    "mean": (0.485, 0.456, 0.406),   # RGB 三通道均值
                    "std": (0.229, 0.224, 0.225),    # RGB 三通道标准差
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
        logger.info("🎉 训练任务顺利完成！")
    except Exception as e:
        logger.error(f"❌ 训练过程发生异常错误: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
