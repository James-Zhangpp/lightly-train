"""破损微小缺陷语义分割模型 ONNX 导出脚本（与 tools/train_breakage.py 配套）。

特点与优化：
1. 默认对齐 train_breakage.py 的默认输出目录与 Checkpoint。
2. 兼容矩形高分辨率长宽比原图导出（如 624x1536），完美支持 DINOv3-EUPE 架构。
3. 支持内嵌预处理模式（Embedded Preprocessing）：C++/C# 上位机直接输入 [0,255] RGB 图像矩阵，单通道 Mask 输出。
4. 支持 TensorRT-friendly 模式（固定 Batch=1，Output 为 int32），解决上位机 TensorRT 引擎转换问题。
5. 内置 ONNX 结构与数值精度校验以及 onnxruntime 冒烟测试。
"""

from __future__ import annotations

import argparse
import copy
import inspect
from pathlib import Path

import lightly_train
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. 默认路径（与 tools/train_breakage.py 输出保持一致）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "out" / "Breakage_small_seg_1024x416_full_vitl16_eomt"
DEFAULT_CHECKPOINT = DEFAULT_EXPERIMENT / "exported_models" / "exported_best.pt"

# ImageNet 标准归一化参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ModelWithPreprocessing(nn.Module):
    """embedded 模式包装：float32 NCHW [0,255] -> 自动归一化 -> [自动正方形 Padding] -> 模型推理 -> [自动裁切还原] -> [B,1,H,W]。"""

    def __init__(
        self,
        model: nn.Module,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        *,
        output_dtype: torch.dtype = torch.int64,
    ) -> None:
        super().__init__()
        self.model = model
        self.output_dtype = output_dtype
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 输入 [0, 255] -> 转换为 [0, 1] 后做标准归一化
        x = x / 255.0
        x = (x - self.mean) / self.std

        # 2. 自动获取当前输入的原图高度 H 和宽度 W
        _, _, h, w = x.shape

        # 3. 解决框架内部 `H != W` 抛出 ValueError 的限制：
        #    自动进行零损 Padding 补齐至正方形 (MaxDim x MaxDim)，绕过框架校验
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w

        if pad_h > 0 or pad_w > 0:
            # (left, right, top, bottom) 填充
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
        else:
            x_padded = x

        # 4. 送入基础模型推理（此时输入为绝对正方形 max_dim x max_dim）
        masks_padded, _logits = self.model(x_padded)

        # 5. 自动裁切还原回原始的长方形原图尺寸 [B, h, w]
        if pad_h > 0 or pad_w > 0:
            masks = masks_padded[:, :h, :w]
        else:
            masks = masks_padded

        return masks.unsqueeze(1).to(self.output_dtype)


class CachedRopeEmbedding(nn.Module):
    """导出前固定 DINOv3 RoPE，避免 ONNX If 条件分支在 TensorRT 转换时报错。"""

    def __init__(self, sin: torch.Tensor, cos: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("sin", sin.detach().clone())
        self.register_buffer("cos", cos.detach().clone())

    def forward(self, *, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sin, self.cos


def _cache_rope_embed_for_export(model: nn.Module, height: int, width: int) -> None:
    """针对长宽比原图（如 624x1536）静态缓存 RoPE 位置编码。"""
    backbone = getattr(model, "backbone", None)
    rope_embed = getattr(backbone, "rope_embed", None)
    patch_size = getattr(backbone, "patch_size", None)
    if backbone is None or rope_embed is None or patch_size is None:
        return

    # 正方形 Padding 后的特征网格维度
    max_dim = max(height, width)
    grid_h = max_dim // int(patch_size)
    grid_w = max_dim // int(patch_size)
    with torch.no_grad():
        sin, cos = rope_embed(H=grid_h, W=grid_w)
    backbone.rope_embed = CachedRopeEmbedding(sin, cos)
    print(f"🔒 已为 DINOv3 静态缓存 RoPE 位置编码: grid={grid_h}x{grid_w} (正方形 {max_dim}x{max_dim})")


def _resolve_paths(
    checkpoint: Path | None,
    onnx_out: Path | None,
) -> tuple[Path, Path]:
    """解析 Checkpoint 与 ONNX 导出输出路径。"""
    checkpoint_path = (checkpoint or DEFAULT_CHECKPOINT).resolve()
    if onnx_out is not None:
        onnx_path = onnx_out.resolve()
    else:
        onnx_path = checkpoint_path.parent / "breakage_model.onnx"
    return checkpoint_path, onnx_path


def _get_image_size(model: nn.Module, height: int | None, width: int | None) -> tuple[int, int]:
    """获取目标尺寸（支持长方形原图如 624x1536）。"""
    if height is not None and width is not None:
        return height, width
    image_size = getattr(model, "image_size", (416, 1024))
    h = height if height is not None else int(image_size[0])
    w = width if width is not None else int(image_size[1])
    return h, w


def _check_patch_size(model: nn.Module, height: int, width: int) -> None:
    """校验尺寸是否能被 patch_size（通常 16）整除，支持长方形长宽比。"""
    patch_size = getattr(getattr(model, "backbone", None), "patch_size", 16)
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"❌ 高度 height={height} 和宽度 width={width} 必须是 patch_size={patch_size} 的整数倍！"
        )


def export_official(
    checkpoint: Path,
    onnx_out: Path,
    *,
    height: int | None = None,
    width: int | None = None,
    dynamic_batch: bool = True,
    simplify: bool = True,
    verify: bool = True,
) -> None:
    """官方模式：直接调用 lightly_train 内置 export_onnx。"""
    print(f"📦 正在加载模型权重: {checkpoint}")
    model = lightly_train.load_model(checkpoint)
    model.eval()

    h, w = _get_image_size(model, height, width)
    _check_patch_size(model, h, w)

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"🚀 开始导出 Official ONNX 格式 -> {onnx_out} ({h}x{w}, dynamic_batch={dynamic_batch})")

    export_fn = getattr(model, "export_onnx", None)
    if export_fn is None:
        raise RuntimeError(f"模型类型 {type(model).__name__} 不支持 export_onnx()")

    export_kwargs: dict = {
        "out": onnx_out,
        "height": h,
        "width": w,
        "dynamic_batch_size": dynamic_batch,
        "simplify": simplify,
        "verify": verify,
    }
    export_fn(**export_kwargs)
    print("✅ Official ONNX 导出完成！")


def export_embedded_preprocessing(
    checkpoint: Path,
    onnx_out: Path,
    *,
    height: int | None = None,
    width: int | None = None,
    dynamic_batch: bool = True,
    opset_version: int = 17,
    verify: bool = True,
    output_dtype: torch.dtype = torch.int64,
) -> None:
    """图内预处理模式：适用于 C#/C++ 上位机部署，直接输入 [0,255] RGB float 矩阵。"""
    print(f"📦 正在加载模型权重: {checkpoint}")
    base_model = lightly_train.load_model(checkpoint)
    base_model.eval()

    h, w = _get_image_size(base_model, height, width)
    _check_patch_size(base_model, h, w)
    _cache_rope_embed_for_export(base_model, h, w)

    wrapped = ModelWithPreprocessing(base_model, output_dtype=output_dtype).eval()
    device = next(base_model.parameters()).device
    wrapped = wrapped.to(device)

    batch_size = 2 if dynamic_batch else 1
    dummy = torch.rand(batch_size, 3, h, w, device=device, dtype=torch.float32) * 255.0

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"🚀 开始导出 Embedded-Preprocessing ONNX 模型 -> {onnx_out} ({h}x{w}), "
        f"output_dtype={output_dtype}, dynamic_batch={dynamic_batch}"
    )

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    export_args = (
        wrapped,
        (dummy,),
        str(onnx_out),
    )
    export_kwargs = {
        "input_names": ["input"],
        "output_names": ["output"],
        "opset_version": opset_version,
        "do_constant_folding": True,
        "dynamic_axes": dynamic_axes,
    }

    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(*export_args, **export_kwargs)

    if verify:
        _verify_embedded_onnx(onnx_out, wrapped, dummy)

    _print_onnx_summary(onnx_out)
    print("✅ Embedded-Preprocessing ONNX 导出成功！")


def _verify_embedded_onnx(
    onnx_path: Path,
    wrapped: ModelWithPreprocessing,
    dummy: torch.Tensor,
) -> None:
    """校验导出后的 ONNX 模型与 PyTorch 原模型推理掩码一致性。"""
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(onnx_path), full_check=True)

    ref_model = copy.deepcopy(wrapped).cpu().float().eval()
    with torch.no_grad():
        ref_out = ref_model(dummy.cpu())

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    ort_out = session.run(["output"], {input_name: dummy.cpu().numpy()})[0]

    ref_np = ref_out.cpu().numpy()
    match_ratio = float((ort_out == ref_np).mean())
    print(f"🔬 ONNX vs PyTorch 掩码匹配率: {match_ratio * 100:.2f}%")
    if match_ratio < 0.99:
        raise RuntimeError(
            f"❌ ONNX 数值验证失败：掩码匹配率 {match_ratio:.4f} < 0.99"
        )
    print("✅ ONNX 数值与逻辑验证通过！")


def _print_onnx_summary(onnx_path: Path) -> None:
    try:
        import onnx
    except ImportError:
        print("💡 提示：安装 onnx (pip install onnx) 可打印模型的网络节点结构")
        return

    model = onnx.load(str(onnx_path))
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"📊 ONNX 模型文件信息: {onnx_path} ({size_mb:.2f} MB)")
    print(f"   输入节点:  {[i.name for i in model.graph.input]}")
    print(f"   输出节点:  {[o.name for o in model.graph.output]}")
    print(f"   总节点数:  {len(model.graph.node)}")


def smoke_test_onnx(onnx_path: Path, *, height: int = 416, width: int = 1024) -> None:
    """使用 onnxruntime 执行单次冒烟推理测试。"""
    if not onnx_path.exists():
        print(f"❌ 未找到 ONNX 模型: {onnx_path}")
        return

    try:
        import onnxruntime as ort
    except ImportError:
        print("💡 提示：安装 onnxruntime (pip install onnxruntime) 可自动运行推理测试")
        return

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    inp = session.get_inputs()[0]
    print(f"\n🧪 启动 ONNXRuntime 冒烟测试: {onnx_path}")
    print(f"   输入: {inp.name} | shape={inp.shape} | type={inp.type}")
    for out in session.get_outputs():
        print(f"   输出: {out.name} | shape={out.shape} | type={out.type}")

    outputs = session.get_outputs()
    if len(outputs) > 1:
        x = np.random.randn(1, 3, height, width).astype(np.float32)
    elif "float" in inp.type and len(inp.shape) == 4 and inp.shape[1] in (3, "3"):
        x = (np.random.rand(1, 3, height, width) * 255.0).astype(np.float32)
    else:
        x = (np.random.rand(1, 3, height, width) * 255.0).astype(np.float32)

    outputs = session.run(None, {inp.name: x})
    for i, arr in enumerate(outputs):
        print(f"   输出[{i}] shape={arr.shape} dtype={arr.dtype} range=[{arr.min()}, {arr.max()}]")
    print("🎉 ONNX 推理冒烟测试顺利通过！")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导出破损微小缺陷语义分割 Checkpoint 为 ONNX 模型（支持 TensorRT 与 C++/C# 上位机）",
    )
    parser.add_argument(
        "--mode",
        choices=("official", "embedded"),
        default="embedded",
        help=(
            "默认 embedded模式（内嵌预处理）。"
            "official: 输入为已归一化 float NCHW；"
            "embedded: 输入为原生 [0,255] float NCHW，输出单通道 0/1 Mask，极其适合上位机"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"权重路径，默认：{DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="导出 .onnx 路径；默认：<checkpoint_dir>/breakage_model.onnx",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="输入图像高度 H（默认 None -> 从 checkpoint 自动读取，如 624）",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="输入图像宽度 W（默认 None -> 从 checkpoint 自动读取，如 1536）",
    )
    parser.add_argument(
        "--no-dynamic-batch",
        action="store_true",
        help="关闭动态 batch（默认开移动态 batch 支持）",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过导出后数值与掩码一致性校验",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="跳过 onnxruntime 冒烟推理测试",
    )
    parser.add_argument(
        "--tensorrt-friendly",
        action="store_true",
        help="TensorRT 友好模式：固定 batch=1，输出 dtype 设为 int32",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset 版本，默认 17",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint, onnx_out = _resolve_paths(args.checkpoint, args.out)

    if args.mode == "embedded" and args.out is None:
        onnx_out = onnx_out.with_name("breakage_model_embedded.onnx")

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"❌ 找不到权重文件: {checkpoint}\n"
            "请先运行训练 (python tools/train_breakage.py) 或通过 --checkpoint 指定现有权重。"
        )

    print(f"PyTorch 版本: {torch.__version__}")

    dynamic_batch = not args.no_dynamic_batch
    output_dtype = torch.int64
    if args.tensorrt_friendly:
        dynamic_batch = False
        output_dtype = torch.int32
        print("⚡ 开启 TensorRT-friendly 模式: batch 固定为 1, 输出类型设为 int32[1,1,H,W]")
    verify = not args.no_verify

    if args.mode == "official":
        export_official(
            checkpoint,
            onnx_out,
            height=args.height,
            width=args.width,
            dynamic_batch=dynamic_batch,
            verify=verify,
        )
    else:
        export_embedded_preprocessing(
            checkpoint,
            onnx_out,
            height=args.height,
            width=args.width,
            dynamic_batch=dynamic_batch,
            opset_version=args.opset,
            verify=verify,
            output_dtype=output_dtype,
        )

    if not args.no_test:
        h, w = args.height or 416, args.width or 1024
        smoke_test_onnx(onnx_out, height=h, width=w)


if __name__ == "__main__":
    main()
