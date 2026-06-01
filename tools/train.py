"""焊接语义分割训练脚本（DINOv3 + EoMT）。

数据目录结构见 tools/weldingseg-data/：
  img_dir/{train,val}  原图
  ann_dir/{train,val}  掩码（类别 id：0=background, 1=fpc）

训练产物：
  out/.../exported_models/exported_best.pt  验证 mIoU 最高
  out/.../exported_models/exported_last.pt   最后一次保存
  out/.../checkpoints/last.ckpt              中断续训用（需 resume_interrupted=True）

默认超参来源：lightly_train DINOv3EoMTSemanticSegmentation（见库内 train_model.py / transforms.py）。
本脚本未写的项使用框架默认值；下方注释中的「默认」均指框架默认，非本文件当前取值。
"""

import os

# 抑制 TensorFlow oneDNN 提示（TensorBoard 等间接依赖可能拉起 TF）
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

import lightly_train

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(__file__).resolve().parent / "weldingseg-data"
OUT_DIR = PROJECT_ROOT / "out" / "welding_dinov3_vits16_eomt_v2"
EXPORTED_BEST = OUT_DIR / "exported_models" / "exported_best.pt"

# ---------------------------------------------------------------------------
# 训练阶段切换（根据是否存在 exported_best 自动选择）
# ---------------------------------------------------------------------------
# 续训模式说明：
#   - 从 best 微调（亮/暗增强等）：resume_interrupted=False，checkpoint=exported_best.pt
#   - 崩溃续跑（需 last.ckpt）：resume_interrupted=True，checkpoint=None
#
# 若存在 exported_best：5000 step 微调（约 40 epoch，按 batch=10、1271 张训练图估算）
# 否则：20000 step 全量训练（约 157 epoch）
FINE_TUNE_FROM_BEST = EXPORTED_BEST.is_file()
STEPS = 5000 if FINE_TUNE_FROM_BEST else 20000
# 默认 lr_warmup_steps=(500, 1000)；微调时按 step 数缩短
LR_WARMUP = (125, 250) if FINE_TUNE_FROM_BEST else (250, 500)

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        # 实验输出目录；默认无，必填
        out=OUT_DIR,
        # 预训练骨干；例如 dinov3/vits16-eomt，无单独默认值
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
        # 训练步数。默认 "auto" -> 40000（dinov3 语义分割）
        # 本脚本：有 best 则 5000，否则 20000
        steps=STEPS,
        # 全局 batch。默认 "auto" -> 16（每卡 batch 由 GPU 数量自动分摊）
        batch_size=10,
        # DataLoader 进程数/卡。默认 "auto"（按 CPU 核数）
        num_workers=6,
        # GPU 数量。默认 "auto"（用全部可见 GPU）
        devices="auto",
        # 加速器。默认 "auto"（有 GPU 用 cuda）
        accelerator="auto",
        # 分布式策略。默认 "auto"
        strategy="auto",
        # 混合精度。默认 "bf16-mixed"
        precision="bf16-mixed",
        # float32 矩阵乘精度。默认 "auto"
        float32_matmul_precision="auto",
        # 多机节点数。默认 1
        num_nodes=1,
        # 随机种子。默认 0
        seed=42,
        # 是否清空 out。默认 False；从 best 微调时改为 False 以免覆盖日志
        overwrite=not FINE_TUNE_FROM_BEST,
        # 从 checkpoints/last.ckpt 恢复完整训练状态。默认 False
        resume_interrupted=False,
        # 仅加载权重继续训（非 last.ckpt）。默认 None
        checkpoint=EXPORTED_BEST if FINE_TUNE_FROM_BEST else None,
        model_args={
            # 学习率。默认 1e-4；微调亮/暗鲁棒性时常用 5e-5
            "lr": 5e-5,
            # 权重衰减。默认 0.05
            "weight_decay": 0.01,
            # EoMT query 数量。默认 "auto" -> 100；二分类小目标可改为 50
            "num_queries": 50,
            # (非 ViT warmup, ViT warmup) 步数。默认 (500, 1000)
            "lr_warmup_steps": LR_WARMUP,
        },
        transform_args={
            # 训练裁剪/缩放目标尺寸。默认 "auto" -> 与模型 image_size 一致，通常 (512,512)
            "image_size": (512, 512),
            # 输入通道数。默认 "auto" -> 3
            "num_channels": 3,
            # 颜色扰动（仅训练）。默认：prob=0.5, strength=1.0,
            #   brightness=32/255, contrast=0.5, saturation=0.5, hue=18/360
            # 下列为加强版，缓解现场过亮/过暗与训练集曝光不一致
            "color_jitter": {
                "prob": 0.8,  # 默认 0.5：做扰动的概率
                "strength": 1.5,  # 默认 1.0：下面四项的整体倍率
                "brightness": 64.0 / 255.0,  # 默认 32/255：亮度随机幅度
                "contrast": 0.8,  # 默认 0.5：对比度
                "saturation": 0.5,  # 默认 0.5：饱和度
                "hue": 18.0 / 360.0,  # 默认 18/360：色调（弧度）
            },
            # 随机缩放后再 crop。默认：min=0.5, max=2.0, num_scales=20, prob=1.0
            "scale_jitter": {
                "sizes": None,  # 默认 None：用 min/max_scale 生成尺度列表
                "min_scale": 0.75,  # 默认 0.5：相对 image_size 最小缩放
                "max_scale": 1.25,  # 默认 2.0：最大缩放
                "num_scales": 10,  # 默认 20：在 [min,max] 间均匀取几个尺度
                "prob": 1.0,  # 默认 1.0：执行概率
                "divisible_by": None,  # 默认 None；当前实现未使用
            },
            # 随机裁 512x512。默认：height/width=auto, pad_if_needed=True,
            #   pad_position="center", fill=0, prob=1.0
            "random_crop": {
                "height": 512,
                "width": 512,
                "pad_if_needed": True,
                "pad_position": "center",
                "fill": 0,
                "prob": 1.0,
            },
            # 随机翻转。默认：horizontal_prob=0.5, vertical_prob=0.0
            "random_flip": {
                "horizontal_prob": 0.5,
                "vertical_prob": 0.0,
            },
            # 小角度旋转。默认：未启用（None）；启用时常见 prob=0.2, degrees=10
            "random_rotate": {
                "prob": 0.2,
                "degrees": 10,
            },
            # ImageNet 归一化（与推理/ONNX 一致）。默认即 ImageNet mean/std
            "normalize": {
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
            },
        },
        # DataLoader 额外参数。默认 None
        loader_args=None,
        # TensorBoard 等日志。默认 None（使用内置默认）
        logger_args=None,
        save_checkpoint_args={
            # 每隔多少 step 存一次中间 ckpt。默认 1000
            "save_every_num_steps": 1000,
            # 是否保存 last。默认 True
            "save_last": True,
            # 是否按验证 mIoU 保存 best。默认 True
            "save_best": True,
        },
    )
