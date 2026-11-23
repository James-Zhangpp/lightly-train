"""L8 FPC 实例分割模型专用 ONNX 导出脚本。

特性：
1. 默认对齐 tools/train_l8_fpc_inst.py 的产物路径。
2. 输入封装为 NHWC 格式：[Batch, 512, 512, 3]，float32，取值范围 [0, 255]。
3. 模型内部内嵌图像归一化（/255.0 后做 ImageNet mean/std），上位机直接传 0~255 RGB 像素矩阵。
4. 输出：
   - labels: [Batch, Q]  类别 ID (int64)
   - masks:  [Batch, Q, 512, 512]  二值掩膜 (uint8: 0 或 1)
   - scores: [Batch, Q]  置信度 (float32)
5. 自动导出配套的 export_meta.json 并执行 ONNXRuntime 冒烟测试。

用法：
  python tools/toOnnx_l8_fpc_inst.py
  python tools/toOnnx_l8_fpc_inst.py --checkpoint out/l4_fpc_dinov3_vits16_eomt_inst/exported_models/exported_best.pt
"""

import argparse
import copy
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import lightly_train
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 路径与默认参数
# ---------------------------------------------------------------------------
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists() or (p / ".git").exists()), Path(__file__).resolve().parents[3])
DEFAULT_EXPERIMENT = PROJECT_ROOT / "out" / "NP_BZ_vitb16_v2"
DEFAULT_CHECKPOINT = DEFAULT_EXPERIMENT / "exported_models" / "exported_best.pt"
META_FILENAME = "export_meta.json"

# ImageNet 标准归一化参数（与训练保持一致）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 默认类别映射
DEFAULT_CLASSES = {
    1: "FPC",
}

# ===========================================================================
# ⭐【单独直接运行配置区】（无需命令行，直接在 IDE 右键/F5 运行修改此处即可）
# ===========================================================================
CHECKPOINT_PATH: Path | None = None   # 权重路径，None 则默认使用 DEFAULT_CHECKPOINT (exported_best.pt)
ONNX_OUT_PATH: Path | None = None     # 输出 ONNX 路径，None 则默认保存在权重同目录下 model_l8_fpc_nhwc.onnx
IMAGE_HEIGHT: int = 512               # 导出尺寸高
IMAGE_WIDTH: int = 512                # 导出尺寸宽
BATCH_SIZE: int = 1                   # 导出 Batch 大小
STATIC_BATCH: bool = True            # 是否固定为静态 Batch 维度 (True: 静态 [BATCH_SIZE, 512, 512, 3]; False: 动态 Batch 维度)
OPSET_VERSION: int = 17               # ONNX Opset 版本
SIMPLIFY_ONNX: bool = True            # 是否使用 onnxslim 进行图优化瘦身
VERIFY_ONNX: bool = True              # 是否在导出后进行 ONNX 结构校验与冒烟测试
SMOKE_TEST_THRESHOLD: float = 0.5     # 冒烟测试过滤置信度阈值
# ===========================================================================


class ModelWithNHWCPreprocessing(nn.Module):
    """内嵌 NHWC 维度置换与 ImageNet 归一化的包装模块。

    输入：
      x: [Batch, H, W, 3]，float32 类型，像素值范围 [0.0, 255.0] (RGB)
    输出：
      labels: [Batch, Q]，类别 ID
      masks:  [Batch, Q, H, W]，uint8 类型 (取值为 0 或 1)
      scores: [Batch, Q]，float32 置信度
    """

    def __init__(
        self,
        model: nn.Module,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        super().__init__()
        self.model = model
        # 归一化 buffer：形状为 [1, 3, 1, 1]
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. 维度置换：[Batch, H, W, C] -> [Batch, C, H, W]
        x = x.permute(0, 3, 1, 2)

        # 2. 内嵌归一化：缩放至 [0, 1] 后进行标准 ImageNet (x - mean) / std
        x = x / 255.0
        x = (x - self.mean) / self.std

        # 3. 基础模型前向推理，获取 (labels, masks, scores)
        labels, masks, scores = self.model(x)

        # 4. masks 转为 uint8 (0 或 1)，对工业 C#/C++ 上位机和各种推理后端最友好
        masks_uint8 = masks.to(torch.uint8)

        return labels, masks_uint8, scores


class CachedRopeEmbedding(nn.Module):
    """静态缓存 DINOv3 RoPE 旋转位置编码，提升 ONNX 图整洁度与 TensorRT 兼容性。"""

    def __init__(self, sin: torch.Tensor, cos: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("sin", sin.detach().clone())
        self.register_buffer("cos", cos.detach().clone())

    def forward(self, *, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sin, self.cos


def _cache_rope_embed_for_export(model: nn.Module, height: int, width: int) -> None:
    """按导出图像尺寸静态缓存 RoPE 位置编码。"""
    backbone = getattr(model, "backbone", None)
    rope_embed = getattr(backbone, "rope_embed", None)
    patch_size = getattr(backbone, "patch_size", None)
    if backbone is None or rope_embed is None or patch_size is None:
        return

    grid_h = height // int(patch_size)
    grid_w = width // int(patch_size)
    with torch.no_grad():
        sin, cos = rope_embed(H=grid_h, W=grid_w)
    backbone.rope_embed = CachedRopeEmbedding(sin, cos)
    print(f"🔒 已为 DINOv3 静态缓存 RoPE 位置编码: 网格尺寸={grid_h}x{grid_w}")


def save_export_meta(
    model: nn.Module,
    meta_path: Path,
    classes: dict[int, str] | None = None,
    image_size: tuple[int, int] = (512, 512),
    batch_size: int = 1,
    dynamic_batch: bool = True,
) -> None:
    """保存模型元数据，便于推理脚本或上位机动态加载类别与配置。"""
    if classes is None:
        classes = getattr(model, "classes", None)
    if not classes:
        classes = DEFAULT_CLASSES

    batch_dim: int = -1 if dynamic_batch else batch_size
    num_q = int(getattr(model, "num_queries", 50))

    meta: dict[str, Any] = {
        "task": "instance_segmentation",
        "model_architecture": "dinov3_vits16_eomt_inst",
        "input_format": "NHWC_RGB",
        "input_name": "images",
        "input_shape": [batch_dim, int(image_size[0]), int(image_size[1]), 3],
        "input_dtype": "float32",
        "pixel_range": [0.0, 255.0],
        "embedded_preprocessing": {
            "permute": [0, 3, 1, 2],
            "scale": 1.0 / 255.0,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "output_names": ["labels", "masks", "scores"],
        "output_shapes": {
            "labels": [batch_dim, num_q],
            "masks": [batch_dim, num_q, int(image_size[0]), int(image_size[1])],
            "scores": [batch_dim, num_q],
        },
        "masks_dtype": "uint8 (0 or 1)",
        "classes": {str(k): v for k, v in classes.items()},
        "num_queries": num_q,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"📄 元数据已保存 -> {meta_path}")


def export_l8_fpc_onnx(
    checkpoint: Path,
    onnx_out: Path,
    *,
    classes: dict[int, str] | None = None,
    image_size: tuple[int, int] = (512, 512),
    batch_size: int = 1,
    dynamic_batch: bool = True,
    opset_version: int = 17,
    simplify: bool = True,
    verify: bool = True,
) -> None:
    """核心导出函数。"""
    print(f"📦 正在加载 Checkpoint: {checkpoint}")
    base_model = lightly_train.load_model(checkpoint)
    base_model.eval()

    h, w = image_size
    _cache_rope_embed_for_export(base_model, h, w)

    # 包装 NHWC 输入与内嵌预处理
    wrapped = ModelWithNHWCPreprocessing(
        base_model,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ).eval()

    device = next(base_model.parameters()).device
    wrapped = wrapped.to(device)

    # 构造 NHWC 形状的假数据 [Batch, 512, 512, 3]
    actual_batch = batch_size if batch_size > 0 else 1
    dummy_input = torch.rand(actual_batch, h, w, 3, device=device, dtype=torch.float32) * 255.0

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    batch_desc = "动态" if dynamic_batch else str(batch_size)
    print(
        f"🚀 开始导出 ONNX -> {onnx_out}\n"
        f"   输入规格: [Batch={batch_desc}, H={h}, W={w}, C=3], float32 [0~255]\n"
        f"   输出规格: labels, masks (uint8), scores\n"
        f"   Opset: {opset_version}"
    )

    # 动态 Batch 维度配置
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch_size"},
            "labels": {0: "batch_size"},
            "masks": {0: "batch_size"},
            "scores": {0: "batch_size"},
        }

    export_kwargs: dict[str, Any] = {
        "input_names": ["images"],
        "output_names": ["labels", "masks", "scores"],
        "opset_version": opset_version,
        "do_constant_folding": True,
        "dynamic_axes": dynamic_axes,
    }

    # 强制关闭 dynamo，使用经过工业验证的标准 TorchScript ONNX 导出后端
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        wrapped,
        (dummy_input,),
        str(onnx_out),
        **export_kwargs,
    )
    print(f"✅ ONNX 模型导出完成: {onnx_out}")

    # 保存元数据
    meta_path = onnx_out.parent / META_FILENAME
    save_export_meta(
        base_model,
        meta_path,
        classes=classes,
        image_size=image_size,
        batch_size=batch_size,
        dynamic_batch=dynamic_batch,
    )

    # 可选模型简化
    if simplify:
        try:
            import onnxslim

            print("⚡ 正在使用 onnxslim 进行 ONNX 模型优化瘦身...")
            onnxslim.slim(model=str(onnx_out), output_model=str(onnx_out))
            print("⚡ onnxslim 优化完成！")
        except ImportError:
            print("⚠️ 未安装 onnxslim，跳过模型简化步骤（可选：pip install onnxslim）")
        except Exception as e:
            print(f"⚠️ onnxslim 优化出现警告，保留原 ONNX 文件: {e}")

    # 校验模型
    if verify:
        try:
            import onnx

            print("🔍 正在检查 ONNX 模型结构有效性...")
            onnx.checker.check_model(str(onnx_out), full_check=True)
            print("✅ ONNX 结构校验通过！")
        except ImportError:
            pass


def smoke_test_onnx(onnx_path: Path, meta_path: Path, threshold: float = 0.5) -> None:
    """使用 ONNXRuntime 进行推理验证（冒烟测试）。"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️ 未安装 onnxruntime，跳过冒烟测试 (pip install onnxruntime)")
        return

    print(f"\n==================== ONNX 冒烟测试 ====================")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _, h, w, c = meta["input_shape"]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print(f"📌 模型输入节点:")
    for inp in inputs:
        print(f"   Name: '{inp.name}', Shape: {inp.shape}, Type: {inp.type}")

    print(f"📌 模型输出节点:")
    for out in outputs:
        print(f"   Name: '{out.name}', Shape: {out.shape}, Type: {out.type}")

    # 构造纯随机像素矩阵 [1, 512, 512, 3] (取值 0~255)
    test_input = np.random.uniform(0.0, 255.0, size=(1, h, w, c)).astype(np.float32)
    labels, masks, scores = session.run(None, {inputs[0].name: test_input})

    print(f"📌 推理输出张量信息:")
    print(f"   labels: shape={labels.shape}, dtype={labels.dtype}")
    print(f"   masks:  shape={masks.shape}, dtype={masks.dtype}, unique_values={np.unique(masks)}")
    print(f"   scores: shape={scores.shape}, dtype={scores.dtype}, max_score={scores.max():.4f}")

    valid_count = int((scores[0] >= threshold).sum())
    print(f"   在阈值 {threshold} 下检出实例数: {valid_count}")
    print(f"=======================================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L8 FPC 实例分割模型导出为内嵌预处理的 NHWC ONNX 模型")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"训练生成的权重路径 (默认: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="ONNX 模型导出目标路径 (默认: 与权重同目录下的 model_l8_fpc_nhwc.onnx)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="输入图像高度 (默认 512)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="输入图像宽度 (默认 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="导出的 Batch 大小 (默认 1；配合 --static-batch 固定为该静态值，否则保留动态 Batch 维度)",
    )
    parser.add_argument(
        "--static-batch",
        action="store_true",
        help="若开启则固定为静态 Batch 维度 (Batch 大小由 --batch-size 决定，默认保留动态 Batch 维度)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX Opset 版本 (默认 17)",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="关闭 onnxslim 模型优化",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="关闭 ONNX 结构校验与冒烟测试",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="冒烟测试过滤置信度阈值 (默认 0.5)",
    )
    return parser.parse_args()


def main() -> None:
    # 判断是否直接在 IDE 中运行（未传入命令行参数）
    if len(sys.argv) <= 1:
        print("💡 [运行提示] 当前未检测到命令行传参，已自动采用脚本顶部的【单独直接运行配置区】参数。")
        raw_checkpoint = CHECKPOINT_PATH or DEFAULT_CHECKPOINT
        checkpoint = raw_checkpoint.resolve()
        raw_onnx = ONNX_OUT_PATH or (checkpoint.parent / "model_l8_fpc_nhwc.onnx")
        onnx_out = raw_onnx.resolve()
        height = IMAGE_HEIGHT
        width = IMAGE_WIDTH
        batch_size = BATCH_SIZE
        dynamic_batch = not STATIC_BATCH
        opset = OPSET_VERSION
        simplify = SIMPLIFY_ONNX
        verify = VERIFY_ONNX
        threshold = SMOKE_TEST_THRESHOLD
    else:
        args = parse_args()
        checkpoint = args.checkpoint.resolve()
        onnx_out = (
            args.out.resolve()
            if args.out
            else checkpoint.parent / "model_l8_fpc_nhwc.onnx"
        )
        height = args.height
        width = args.width
        batch_size = args.batch_size
        dynamic_batch = not args.static_batch
        opset = args.opset
        simplify = not args.no_simplify
        verify = not args.no_verify
        threshold = args.threshold

    if not checkpoint.is_file():
        # 如果 exported_best.pt 不存在，尝试回退查找 exported_last.pt
        alt_checkpoint = checkpoint.parent / "exported_last.pt"
        if alt_checkpoint.is_file():
            print(f"⚠️ 指定的 {checkpoint} 不存在，已自动切换使用 {alt_checkpoint}")
            checkpoint = alt_checkpoint
        else:
            raise FileNotFoundError(
                f"❌ 未找到模型权重文件: {checkpoint}\n"
                f"请先运行训练脚本: python tools/train_l8_fpc_inst.py"
            )

    export_l8_fpc_onnx(
        checkpoint=checkpoint,
        onnx_out=onnx_out,
        classes=DEFAULT_CLASSES,
        image_size=(height, width),
        batch_size=batch_size,
        dynamic_batch=dynamic_batch,
        opset_version=opset,
        simplify=simplify,
        verify=verify,
    )

    if verify:
        smoke_test_onnx(
            onnx_path=onnx_out,
            meta_path=onnx_out.parent / META_FILENAME,
            threshold=threshold,
        )


if __name__ == "__main__":
    main()
