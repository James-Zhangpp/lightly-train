"""将语义分割 checkpoint 导出为 ONNX（与 tools/train.py 配套）。

导出模式（--mode）：
  official  调用 lightly_train 内置 export_onnx
            输入 images：float32 NCHW，已做 ImageNet 归一化（约 [0,1] 再减 mean/std）
            输出 masks + logits；输入名 images，动态 batch 默认可变
  embedded  默认。图内 /255 + mean/std，仅输出 mask
            输入 input：float32 NCHW，像素约 [0,255]
            输出 output：int64 [B,1,H,W]（--tensorrt-friendly 时为 int32、batch=1）

空间尺寸须与训练一致（默认 512×512）。DINOv3 要求 H==W 且为 patch_size 倍数。
动态 batch 支持；H/W 在 ONNX 中为静态（TensorRT 亦需固定 H/W）。

TensorRT：若 trtexec 报 RoPE/If 等错误，请使用 --tensorrt-friendly。
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import lightly_train
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 路径（与 tools/train.py OUT_DIR 一致）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "out" / "welding_dinov3_vits16_eomt_v2"
# 默认导出 embedded 时常用 last；也可改为 exported_best.pt
DEFAULT_CHECKPOINT = DEFAULT_EXPERIMENT / "exported_models" / "exported_best.pt"

# 与 tools/train.py transform_args.normalize 一致（框架默认即 ImageNet）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ModelWithPreprocessing(nn.Module):
    """embedded 模式包装：float32 NCHW [0,255] -> 归一化 -> masks -> [B,1,H,W]。"""

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
        x = x / 255.0
        x = (x - self.mean) / self.std
        masks, _logits = self.model(x)
        return masks.unsqueeze(1).to(self.output_dtype)


class CachedRopeEmbedding(nn.Module):
    """导出前固定 DINOv3 RoPE，避免 ONNX If 在 TensorRT 中两分支 shape 不一致。"""

    def __init__(self, sin: torch.Tensor, cos: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("sin", sin.detach().clone())
        self.register_buffer("cos", cos.detach().clone())

    def forward(self, *, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sin, self.cos


def _cache_rope_embed_for_export(model: nn.Module, height: int, width: int) -> None:
    """按静态导出尺寸缓存 RoPE（embedded / TensorRT 路径调用）。"""
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
    print(f"Cached DINOv3 RoPE for TensorRT export: grid={grid_h}x{grid_w}")


def _resolve_paths(
    checkpoint: Path | None,
    onnx_out: Path | None,
) -> tuple[Path, Path]:
    """解析 checkpoint 与输出路径。

    checkpoint 默认：DEFAULT_CHECKPOINT（exported_last.pt）
    onnx_out 默认：checkpoint 同目录下的 model.onnx
    """
    checkpoint_path = (checkpoint or DEFAULT_CHECKPOINT).resolve()
    if onnx_out is not None:
        onnx_path = onnx_out.resolve()
    else:
        onnx_path = checkpoint_path.parent / "model.onnx"
    return checkpoint_path, onnx_path


def _get_image_size(model: nn.Module, height: int | None, width: int | None) -> tuple[int, int]:
    """高宽默认取自 model.image_size，通常为 (512, 512)。"""
    if height is not None and width is not None:
        return height, width
    image_size = getattr(model, "image_size", (512, 512))
    h = height if height is not None else int(image_size[0])
    w = width if width is not None else int(image_size[1])
    return h, w


def _check_patch_size(model: nn.Module, height: int, width: int) -> None:
    """DINOv3：H、W 须为 patch_size（通常 16）倍数，且 H==W。"""
    patch_size = getattr(getattr(model, "backbone", None), "patch_size", 16)
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"height={height} and width={width} must be multiples of patch_size={patch_size}"
        )
    if height != width:
        raise ValueError(
            f"DINOv3 ONNX export requires square input (height == width), got {height}x{width}"
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
    """official：走 lightly_train.export_onnx（与库默认一致）。

    库内默认（DINOv3EoMTSemanticSegmentation.export_onnx）：
      precision="auto", batch_size=1, dynamic_batch_size=True,
      height/width=None -> model.image_size, opset_version=None（PyTorch 默认）,
      simplify=True, verify=True
    图结构：输入 images [N,3,H,W] float 已归一化；输出 masks、logits
    """
    print(f"Loading model: {checkpoint}")
    model = lightly_train.load_model(checkpoint)
    model.eval()

    h, w = _get_image_size(model, height, width)
    _check_patch_size(model, h, w)

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting official ONNX -> {onnx_out} ({h}x{w}, dynamic_batch={dynamic_batch})")

    export_fn = getattr(model, "export_onnx", None)
    if export_fn is None:
        raise RuntimeError(f"Model type {type(model).__name__} does not support export_onnx()")

    export_kwargs: dict = {
        "out": onnx_out,
        "height": h,
        "width": w,
        "dynamic_batch_size": dynamic_batch,
        "simplify": simplify,
        "verify": verify,
    }
    export_fn(**export_kwargs)
    print("Official ONNX export finished.")


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
    """embedded：图内含 /255 与 ImageNet 归一化，单输出 output。

    本函数默认：
      dynamic_batch=True（trace 时 batch=2，与 lightly_train 约定一致）
      opset_version=17
      output_dtype=int64 -> [B,1,H,W]
      verify=True（与 PyTorch 掩码一致率 >= 0.99）
    --tensorrt-friendly 时由 main() 改为 batch=1、int32。
    """
    print(f"Loading model: {checkpoint}")
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
        f"Exporting embedded-preprocess ONNX -> {onnx_out} ({h}x{w}), "
        f"output_dtype={output_dtype}"
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

    import inspect

    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(*export_args, **export_kwargs)

    if verify:
        _verify_embedded_onnx(onnx_out, wrapped, dummy)

    _print_onnx_summary(onnx_out)
    print("Embedded-preprocess ONNX export finished.")


def _verify_embedded_onnx(
    onnx_path: Path,
    wrapped: ModelWithPreprocessing,
    dummy: torch.Tensor,
) -> None:
    """导出后数值校验；匹配率阈值默认 0.99。"""
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
    print(f"ONNX vs PyTorch mask match ratio: {match_ratio:.4f}")
    if match_ratio < 0.99:
        raise RuntimeError(
            f"ONNX verification failed: mask match ratio {match_ratio:.4f} < 0.99"
        )
    print("ONNX numerical verification passed.")


def _print_onnx_summary(onnx_path: Path) -> None:
    try:
        import onnx
    except ImportError:
        print("Tip: install onnx to print graph summary (pip install onnx)")
        return

    model = onnx.load(str(onnx_path))
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"ONNX file: {onnx_path} ({size_mb:.2f} MB)")
    print(f"  inputs:  {[i.name for i in model.graph.input]}")
    print(f"  outputs: {[o.name for o in model.graph.output]}")
    print(f"  nodes:   {len(model.graph.node)}")


def smoke_test_onnx(onnx_path: Path, *, height: int = 512, width: int = 512) -> None:
    """用 onnxruntime 做一次冒烟推理（非 pytest 用例）。

    height/width 默认 512，仅用于构造随机输入；须与导出时尺寸一致。
    """
    if not onnx_path.exists():
        print(f"ONNX not found: {onnx_path}")
        return

    try:
        import onnxruntime as ort
    except ImportError:
        print("Tip: pip install onnxruntime to run inference test")
        return

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    inp = session.get_inputs()[0]
    print(f"\nTest session: {onnx_path}")
    print(f"  input:  {inp.name} shape={inp.shape} type={inp.type}")
    for out in session.get_outputs():
        print(f"  output: {out.name} shape={out.shape} type={out.type}")

    outputs = session.get_outputs()
    if len(outputs) > 1:
        x = np.random.randn(1, 3, height, width).astype(np.float32)
    elif "float" in inp.type and len(inp.shape) == 4 and inp.shape[1] in (3, "3"):
        x = (np.random.rand(1, 3, height, width) * 255.0).astype(np.float32)
    elif len(inp.shape) == 4 and inp.shape[-1] in (3, "3"):
        x = np.random.randint(0, 256, (1, height, width, 3), dtype=np.uint8)
    else:
        x = (np.random.rand(1, 3, height, width) * 255.0).astype(np.float32)

    outputs = session.run(None, {inp.name: x})
    for i, arr in enumerate(outputs):
        print(f"  out[{i}] shape={arr.shape} dtype={arr.dtype} range=[{arr.min()}, {arr.max()}]")
    print("ONNX inference smoke test OK.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export LightlyTrain segmentation checkpoint to ONNX",
    )
    parser.add_argument(
        "--mode",
        choices=("official", "embedded"),
        default="embedded",
        help=(
            "默认 embedded。"
            "official: 输入 images 已归一化 float NCHW，输出 masks+logits；"
            "embedded: 输入 input [0,255] float NCHW，输出 output 单通道 mask"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"权重路径，默认 {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 .onnx 路径；默认 <checkpoint_dir>/model.onnx（embedded 且未指定时改为 model_with_preprocessing.onnx）",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="输入高；默认 None -> 从 checkpoint 的 image_size 读取（通常 512）",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="输入宽；默认 None -> 与 height 相同规则",
    )
    parser.add_argument(
        "--no-dynamic-batch",
        action="store_true",
        help="关闭动态 batch；默认开启动态 batch（official/embedded 均为 True）",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过导出后数值校验；默认 verify=True",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="跳过 onnxruntime 冒烟测试；默认导出后会 smoke_test",
    )
    parser.add_argument(
        "--tensorrt-friendly",
        action="store_true",
        help="TRT 友好：固定 batch=1、output 为 int32；默认 batch 动态、output int64",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="embedded 模式 ONNX opset；official 由库决定（默认 None=PyTorch 默认）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint, onnx_out = _resolve_paths(args.checkpoint, args.out)

    # embedded 且未显式 --out 时，避免与 official 的 model.onnx 混淆
    if args.mode == "embedded" and args.out is None:
        onnx_out = onnx_out.with_name("model_with_preprocessing.onnx")

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Train first (tools/train.py) or pass --checkpoint."
        )

    print(f"PyTorch: {torch.__version__}")

    dynamic_batch = not args.no_dynamic_batch
    output_dtype = torch.int64
    if args.tensorrt_friendly:
        dynamic_batch = False
        output_dtype = torch.int32
        print("TensorRT-friendly: batch fixed to 1, output int32[1,1,H,W]")
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
        h, w = args.height or 512, args.width or 512
        smoke_test_onnx(onnx_out, height=h, width=w)


if __name__ == "__main__":
    main()
