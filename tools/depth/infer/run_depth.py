"""
Depth Estimation Verification Script
Runs monocular depth estimation on a sample image and saves the visualized result.
"""

import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

import lightly_train


def test_depth_estimation():
    # 1. 确定运行设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Running on device: {device}")
    if device == "cuda":
        print(f"[Info] GPU: {torch.cuda.get_device_name(0)}")

    # 2. 加载最高精度模型 (Depth Anything V3 Large，基于 DINOv2 ViT-L/14，334M 参数)
    # 首次运行会自动从官方托管源下载预训练权重 (~1.3 GB) 并缓存
    model_name = "dinov2/dav3-relative-large"
    print(f"[Info] Loading highest accuracy model '{model_name}'...")
    start_load = time.perf_counter()
    model = lightly_train.load_model(model_name, device=device)
    load_time = (time.perf_counter() - start_load) * 1000
    print(f"[Info] Model loaded in {load_time:.1f} ms")

    # 3. 指定测试图片
    sample_img_path = Path(r"C:\Users\ASUS\Desktop\Line15\新建文件夹\3.bmp")
    if not sample_img_path.exists():
        # 如果指定路径不存在，自动回退到项目内示例图
        fallback_path = Path("docs/source/_static/images/instance_segmentation/cats.jpg")
        print(f"[Warn] Specified image not found: {sample_img_path}, using fallback: {fallback_path}")
        sample_img_path = fallback_path

    raw_img = Image.open(sample_img_path).convert("RGB")
    w, h = raw_img.size
    print(f"[Info] Input image: {sample_img_path.name} ({w}x{h})")

    # 4. 执行深度预测 (使用 lower_bound_resize 保持长宽比不变，短边518，细节最丰富)
    # GPU 预热
    if device == "cuda":
        _ = model.predict(sample_img_path, process_res_method="lower_bound_resize")
        torch.cuda.synchronize()

    num_runs = 3
    latencies = []
    depth_tensor = None
    for i in range(num_runs):
        t0 = time.perf_counter()
        depth_tensor = model.predict(sample_img_path, process_res_method="lower_bound_resize")
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)

    avg_latency = np.mean(latencies)
    print(f"[Info] Inference latency (avg over {num_runs} runs): {avg_latency:.2f} ms ({1000/avg_latency:.1f} FPS)")
    print(f"[Info] Output depth map shape: {depth_tensor.shape}, min: {depth_tensor.min().item():.4f}, max: {depth_tensor.max().item():.4f}")

    # 5. 生成并保存工业级输出结果
    output_dir = Path("depth_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = sample_img_path.stem

    depth_np = depth_tensor.detach().cpu().numpy()

    # (1) 保存原始浮点矩阵 (.npy)，用于精密数值分析、阈值分割或三维重建
    npy_path = output_dir / f"{base_name}_depth_raw.npy"
    np.save(npy_path, depth_np)
    print(f"[Save] Raw depth float32 matrix saved: {npy_path}")

    # (2) 归一化为 16-bit 灰度无损 PNG (0~65535)，便于工控软件/OpenCV直接载入
    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
    depth_uint16 = (depth_norm * 65535.0).astype(np.uint16)
    img_16bit = Image.fromarray(depth_uint16, mode="I;16")
    png_16bit_path = output_dir / f"{base_name}_depth_16bit.png"
    img_16bit.save(png_16bit_path)
    print(f"[Save] 16-bit depth image saved: {png_16bit_path}")

    # (3) 绘制高对比度伪彩热力图对比展示 (PNG)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Input Image ({w}x{h})")
    axes[0].axis("off")

    im_plot = axes[1].imshow(depth_np, cmap="inferno")
    axes[1].set_title(f"DAv3-Large Relative Depth ({avg_latency:.1f} ms)")
    axes[1].axis("off")

    cbar = fig.colorbar(im_plot, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Relative Depth (Dark=Near, Bright=Far)")

    vis_path = output_dir / f"{base_name}_depth_vis.png"
    plt.tight_layout()
    plt.savefig(vis_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Save] Visualization comparison saved: {vis_path}")


if __name__ == "__main__":
    test_depth_estimation()
