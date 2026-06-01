"""将图像分类 checkpoint 导出为 ONNX（与 tools/train_cls.py 配套）。

ONNX 输入 images：float32 NCHW，ImageNet 归一化后的张量，
空间尺寸为 export_meta.json 中的 image_size（外部需 resize_and_pad，见 test_cls.py）。

输出 labels, scores（multiclass top-1；labels 为 data.classes 中的类别 id）。

同时写出 export_meta.json，供 test_cls.py 使用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightly_train
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "out" / "welding_dinov3_vitt16_cls_v1"
DEFAULT_CHECKPOINT = DEFAULT_EXPERIMENT / "exported_models" / "exported_best.pt"
META_FILENAME = "export_meta.json"


def save_export_meta(model: torch.nn.Module, path: Path) -> None:
    classes = getattr(model, "classes", {})
    image_size = getattr(model, "image_size", (224, 224))
    image_normalize = getattr(model, "image_normalize", None)
    classification_task = getattr(model, "classification_task", "multiclass")
    meta = {
        "task": "image_classification",
        "classification_task": classification_task,
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


def smoke_test_onnx(onnx_path: Path, meta_path: Path) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("Tip: pip install onnxruntime")
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    h, w = meta["image_size"]
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    inp = session.get_inputs()[0]
    print(f"\nSmoke test: {onnx_path}")
    print(f"  input:  {inp.name} {inp.shape} {inp.type}")
    for out in session.get_outputs():
        print(f"  output: {out.name} {out.shape} {out.type}")

    dummy = torch.randn(1, 3, h, w, dtype=torch.float32).numpy()
    labels, scores = session.run(None, {"images": dummy})
    print(f"  labels={labels}, scores={scores}")


def main() -> None:
    parser = argparse.ArgumentParser(description="分类模型导出 ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="exported_best.pt 路径",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="ONNX 输出路径，默认 <experiment>/exported_models/model.onnx",
    )
    parser.add_argument("--no-dynamic-batch", action="store_true")
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    onnx_out = args.out
    if onnx_out is None:
        onnx_out = checkpoint.parent / "model.onnx"
    else:
        onnx_out = args.out.resolve()

    export_onnx(
        checkpoint,
        onnx_out,
        dynamic_batch=not args.no_dynamic_batch,
        simplify=not args.no_simplify,
        verify=not args.no_verify,
    )

    meta_path = onnx_out.parent / META_FILENAME
    if args.smoke_test:
        smoke_test_onnx(onnx_out, meta_path)


if __name__ == "__main__":
    main()
