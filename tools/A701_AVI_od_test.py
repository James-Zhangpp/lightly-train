"""A701_AVI 目标检测测试脚本。

加载导出的 ONNX 模型并在指定目录下的图片上运行推理，保存可视化结果。
"""

import argparse
import json
import glob
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "out" / "A701_AVI_wpoint_detect_ltdetrv2_l_coco"
DEFAULT_ONNX = DEFAULT_EXPERIMENT / "exported_models" / "model.onnx"
# DEFAULT_TEST_IMAGES = PROJECT_ROOT / "tools" / "DataSet" / "A701_AVI_wpoint_detect" / "images" / "val"
DEFAULT_TEST_IMAGES=r'C:\Users\ASUS\Desktop\A07_01\原图'
def parse_args():
    parser = argparse.ArgumentParser(description="A701_AVI 目标检测模型 ONNX 推理")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="ONNX 模型路径")
    parser.add_argument("--img-dir", type=Path, default=DEFAULT_TEST_IMAGES, help="测试图片目录")
    parser.add_argument("--out-dir", type=Path, default=r'C:\Users\ASUS\Desktop\A07_01\testimg', help="输出可视化目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="置信度阈值")
    return parser.parse_args()

def draw_boxes(image, boxes, labels, scores, classes, threshold):
    img_h, img_w = image.shape[:2]
    out_img = image.copy()
    
    keep = scores > threshold
    if not keep.any():
        return out_img
        
    for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
        x1, y1, x2, y2 = [int(v) for v in box]
        class_name = classes.get(str(label), str(label))
        text = f"{class_name}: {score:.2f}"
        
        # Draw box
        cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        cv2.rectangle(out_img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
        cv2.putText(out_img, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
    return out_img

def main():
    args = parse_args()
    
    onnx_path = args.onnx.resolve()
    meta_path = onnx_path.parent / "export_meta.json"
    
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX 模型未找到: {onnx_path}\n请先运行 tools/toOnnx_A701_AVI_od.py")
    if not meta_path.is_file():
        raise FileNotFoundError(f"元数据未找到: {meta_path}")
        
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    h, w = meta["image_size"]
    mean = np.array(meta["image_normalize"]["mean"]).reshape(3, 1, 1).astype(np.float32)
    std = np.array(meta["image_normalize"]["std"]).reshape(3, 1, 1).astype(np.float32)
    classes = meta["classes"]
    
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    
    out_dir = args.out_dir if args.out_dir else onnx_path.parent / "test_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    img_paths = glob.glob(str(args.img_dir / "*.*"))
    if not img_paths:
        print(f"未在 {args.img_dir} 找到测试图片。")
        return
        
    print(f"找到 {len(img_paths)} 张测试图片，开始推理...")
    for img_path in img_paths:
        img_path = Path(img_path)
        if img_path.suffix.lower() not in [".jpg", ".png", ".bmp", ".jpeg"]:
            continue
            
        # 使用 imdecode 代替 imread，防止 Windows 下路径存在中文字符导致静默读取失败
        img_data = np.fromfile(str(img_path), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"⚠️ [警告] 无法读取图片，可能文件已损坏: {img_path.name}")
            continue
            
        # 预处理
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (w, h))
        img_tensor = img_resized.transpose(2, 0, 1).astype(np.float32)
        img_tensor = img_tensor / 255.0
        img_tensor = (img_tensor - mean) / std
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        # 推理
        t0 = time.time()
        logits_out, boxes_out = session.run(None, {"images": img_tensor})
        t1 = time.time()
        infer_time_ms = (t1 - t0) * 1000
        
        # 去除 batch 维度
        logits = logits_out[0]  # shape: [300, num_classes]
        boxes = boxes_out[0]    # shape: [300, 4] (通常为归一化的 cx, cy, w, h)
        
        # 对 logits 使用 Sigmoid 转换为概率
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        probs = sigmoid(logits)
        
        # 获取得分和类别标签
        scores_out = np.max(probs, axis=-1)
        labels_out = np.argmax(probs, axis=-1)
        
        # 统计超过阈值的有效目标数量并打印日志
        num_targets = np.sum(scores_out > args.threshold)
        print(f"[{img_path.name}] 推理耗时: {infer_time_ms:.2f} ms | 检测到目标数: {num_targets}")
        
        # 解析坐标 (cx, cy, w, h) -> (x1, y1, x2, y2)
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        
        # 恢复坐标到原图尺寸
        orig_h, orig_w = img_bgr.shape[:2]
        
        # DETR 模型的 ONNX 导出结果一定是 0~1 的归一化坐标，
        # 但部分置信度极低的废弃框坐标可能会略微溢出 1.0 (比如 1.05)，
        # 所以不能用 max() <= 1.0 来判断，直接乘上原图宽高即可！
        x1 = x1 * orig_w
        x2 = x2 * orig_w
        y1 = y1 * orig_h
        y2 = y2 * orig_h
        
        boxes_out = np.stack([x1, y1, x2, y2], axis=-1)
        
        # 调试信息：如果检测到了目标，打印第一个目标的原始框和转换后的框
        if num_targets > 0:
            valid_idx = np.where(scores_out > args.threshold)[0][0]
            print(f"  -> [DEBUG] 第一个有效目标的原始坐标 (cx/cy/w/h 或 x1/y1/x2/y2?): {boxes[valid_idx]}")
            print(f"  -> [DEBUG] 转换后的实际绘制坐标 (x1, y1, x2, y2): {boxes_out[valid_idx]}")
            
        # 可视化
        res_img = draw_boxes(img_bgr, boxes_out, labels_out, scores_out, classes, args.threshold)
        
        save_path = out_dir / f"res_{img_path.name}"
        cv2.imwrite(str(save_path), res_img)
        
    print(f"推理完成，可视化结果保存在: {out_dir}")

if __name__ == "__main__":
    main()
