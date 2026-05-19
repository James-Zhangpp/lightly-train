"""Export semantic segmentation checkpoints to ONNX.

Two export modes:
  - official: uses lightly_train's built-in export_onnx (float NCHW, normalized).
  - embedded: wraps preprocessing (uint8 HWC [0,255]) and exports masks only.

Input spatial size must match training (default 512x512). DINOv3 ONNX export
requires height == width; dynamic batch is supported, dynamic H/W is not.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import lightly_train
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "out" / "welding_dinov3_vits16_eomt_lr1e4_q50"
DEFAULT_CHECKPOINT = DEFAULT_EXPERIMENT / "exported_models" / "exported_last.pt"

# Same as tools/train.py transform_args.normalize
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ModelWithPreprocessing(nn.Module):
    """uint8 HWC [0, 255] RGB -> normalize -> segmentation masks [B, H, W]."""

    def __init__(
        self,
        model: nn.Module,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fixed layout at export time: [B, H, W, 3], do not branch on x.shape[1].
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std
        masks, _logits = self.model(x)
        return masks


def _resolve_paths(
    checkpoint: Path | None,
    onnx_out: Path | None,
) -> tuple[Path, Path]:
    checkpoint_path = (checkpoint or DEFAULT_CHECKPOINT).resolve()
    if onnx_out is not None:
        onnx_path = onnx_out.resolve()
    else:
        onnx_path = checkpoint_path.parent / "model.onnx"
    return checkpoint_path, onnx_path


def _get_image_size(model: nn.Module, height: int | None, width: int | None) -> tuple[int, int]:
    if height is not None and width is not None:
        return height, width
    image_size = getattr(model, "image_size", (512, 512))
    h = height if height is not None else int(image_size[0])
    w = width if width is not None else int(image_size[1])
    return h, w


def _check_patch_size(model: nn.Module, height: int, width: int) -> None:
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
    """Export via lightly_train (input: normalized float NCHW, outputs: masks + logits)."""
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
    # torch>=2.4 passes dynamo=False inside export_onnx; older versions ignore it.
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
) -> None:
    """Export uint8 HWC model with preprocessing baked in (output: masks only)."""
    print(f"Loading model: {checkpoint}")
    base_model = lightly_train.load_model(checkpoint)
    base_model.eval()

    h, w = _get_image_size(base_model, height, width)
    _check_patch_size(base_model, h, w)

    wrapped = ModelWithPreprocessing(base_model).eval()
    device = next(base_model.parameters()).device
    wrapped = wrapped.to(device)

    # Use batch_size=2 when tracing dynamic batch (lightly_train convention).
    batch_size = 2 if dynamic_batch else 1
    dummy = torch.randint(
        0, 256, (batch_size, h, w, 3), dtype=torch.uint8, device=device
    )

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting embedded-preprocess ONNX -> {onnx_out} ({h}x{w})")

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input_image": {0: "batch_size"},
            "segmentation_mask": {0: "batch_size"},
        }

    export_args = (
        wrapped,
        (dummy,),
        str(onnx_out),
    )
    export_kwargs = {
        "input_names": ["input_image"],
        "output_names": ["segmentation_mask"],
        "opset_version": opset_version,
        "do_constant_folding": True,
        "dynamic_axes": dynamic_axes,
    }

    # PyTorch 2.4+ uses dynamo exporter by default; DINOv3 needs legacy path.
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
    ort_out = session.run(
        ["segmentation_mask"], {input_name: dummy.cpu().numpy()}
    )[0]

    ort_tensor = torch.from_numpy(ort_out)
    match_ratio = (ort_tensor == ref_out.cpu()).float().mean().item()
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
    """Smoke-test an exported ONNX file with onnxruntime (not a pytest test)."""
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

    if len(inp.shape) == 4 and inp.shape[-1] in (3, "3"):
        # embedded: [N, H, W, 3] uint8
        x = np.random.randint(0, 256, (1, height, width, 3), dtype=np.uint8)
    else:
        # official: [N, 3, H, W] float32, already normalized
        x = np.random.randn(1, 3, height, width).astype(np.float32)

    outputs = session.run(None, {inp.name: x})
    for i, arr in enumerate(outputs):
        print(f"  out[{i}] shape={arr.shape} dtype={arr.dtype} range=[{arr.min()}, {arr.max()}]")
    print("ONNX inference smoke test OK.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LightlyTrain segmentation model to ONNX")
    parser.add_argument(
        "--mode",
        choices=("official", "embedded"),
        default="embedded",
        help=(
            "official: float NCHW normalized input, outputs masks+logits; "
            "embedded: uint8 HWC [0,255] input, output masks only"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to exported_last.pt")
    parser.add_argument("--out", type=Path, default=None, help="Output .onnx path")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--no-dynamic-batch", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--no-test", action="store_true", help="Skip onnxruntime smoke test")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint, onnx_out = _resolve_paths(args.checkpoint, args.out)

    if args.mode == "embedded" and onnx_out.name == "model.onnx":
        onnx_out = onnx_out.with_name("model_with_preprocessing.onnx")

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Train first (tools/train.py) or pass --checkpoint."
        )

    print(f"PyTorch: {torch.__version__}")

    dynamic_batch = not args.no_dynamic_batch
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
        )

    if not args.no_test:
        h, w = args.height or 512, args.width or 512
        smoke_test_onnx(onnx_out, height=h, width=w)


if __name__ == "__main__":
    main()
