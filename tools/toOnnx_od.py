"""将目标检测 checkpoint 导出为 ONNX（与 tools/train_od.py 配套）。

调用 lightly_train DINOv3LTDETRObjectDetection.export_onnx：
  输入 images：float32 NCHW，已 Resize + ImageNet 归一化（与训练一致）
  输出 labels, boxes, scores（boxes 为模型输入分辨率下的 xyxy 像素坐标）

同时写出 export_meta.json（image_size、normalize、classes），供 test_od.py 推理使用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightly_train
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "out" / "welding_dinov3_vits16_ltdetr_od_v1"
DEFAULT_CHECKPOINT = DEFAULT_EXPERIMENT / "exported_models" / "exported_best.pt"
META_FILENAME = "export_meta.json"


def save_export_meta(model: torch.nn.Module, path: Path) -> None:
    classes = getattr(model, "classes", {})
    image_size = getattr(model, "image_size", (512, 512))
    image_normalize = getattr(model, "image_normalize", None)
    meta = {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "image_normalize": image_normalize,
        "classes": {str(k): v for k, v in classes.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved export meta -> {path}")


def export_onnx(
    checkpoint: Path,
    onnx_out: Path,
    *,
    dynamic_batch: bool = True,
    simplify: bool = True,
    verify: bool = True,
) -> None:
    print(f"Loading model: {checkpoint}")
    model = lightly_train.load_model(checkpoint)
    model.eval()

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Exporting ONNX -> {onnx_out} "
        f"(image_size={model.image_size}, dynamic_batch={dynamic_batch})"
    )

    export_fn = getattr(model, "export_onnx", None)
    if export_fn is None:
        raise RuntimeError(f"Model {type(model).__name__} does not support export_onnx()")

    export_fn(
        out=onnx_out,
        dynamic_batch_size=dynamic_batch,
        simplify=simplify,
        verify=verify,
    )

    save_export_meta(model, onnx_out.parent / META_FILENAME)
    print("ONNX export finished.")


def smoke_test_onnx(onnx_path: Path, meta_path: Path, *, threshold: float = 0.5) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("Tip: pip install onnxruntime")
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    h, w = meta["image_size"]
    mean = meta["image_normalize"]["mean"]
    std = meta["image_normalize"]["std"]

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    inp = session.get_inputs()[0]
    print(f"\nSmoke test: {onnx_path}")
    print(f"  input:  {inp.name} {inp.shape} {inp.type}")
    for out in session.get_outputs():
        print(f"  output: {out.name} {out.shape} {out.type}")

    dummy = torch.rand(1, 3, h, w)
    for t in [
        dummy,
        (dummy - torch.tensor(mean).view(3, 1, 1))
        / torch.tensor(std).view(3, 1, 1),
    ]:
        x = t.numpy().astype("float32")
        labels, boxes, scores = session.run(None, {"images": x})
        keep = scores > threshold
        n = int(keep.sum()) if keep.ndim else 0
        print(f"  detections (threshold={threshold}): {n}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export LightlyTrain object-detection checkpoint to ONNX",
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
        help="输出 .onnx；默认 <checkpoint_dir>/model.onnx",
    )
    parser.add_argument(
        "--no-dynamic-batch",
        action="store_true",
        help="固定 batch=1；默认动态 batch",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过导出后数值校验",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="跳过 onnxslim 简化",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="跳过 onnxruntime 冒烟测试",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="冒烟测试置信度阈值，默认 0.5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = (args.checkpoint or DEFAULT_CHECKPOINT).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Train first: python tools/train_od.py"
        )

    if args.out is not None:
        onnx_out = args.out.resolve()
    else:
        onnx_out = checkpoint.parent / "model.onnx"

    print(f"PyTorch: {torch.__version__}")

    export_onnx(
        checkpoint,
        onnx_out,
        dynamic_batch=not args.no_dynamic_batch,
        simplify=not args.no_simplify,
        verify=not args.no_verify,
    )

    if not args.no_test:
        meta_path = onnx_out.parent / META_FILENAME
        smoke_test_onnx(onnx_out, meta_path, threshold=args.threshold)


if __name__ == "__main__":
    main()
