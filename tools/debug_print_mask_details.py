import os
import cv2
import numpy as np
from pathlib import Path

def debug_masks():
    # 兼容 Windows 路径的精准解析
    SCRIPT_DIR = Path(__file__).resolve().parent
    data_dir = SCRIPT_DIR / "Breakage_small_seg" / "ann_dir"
    
    print("=" * 65)
    print("🔬 [Mask 标注图底层像素诊断与调试系统]")
    print(f"扫描目标路径: {data_dir}")
    print("=" * 65)
    
    if not data_dir.exists():
        print(f"❌ 目录不存在: {data_dir}")
        return

    # 递归查找所有图片文件
    ann_paths = [p for p in data_dir.rglob("*.*") if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]]
    
    if not ann_paths:
        print(f"❌ 未找到任何 Mask 图像文件！检查路径: {data_dir}")
        return
        
    for p in ann_paths:
        rel_path = p.relative_to(SCRIPT_DIR)
        mask = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        
        if mask is None:
            print(f"❌ 无法解码读取文件: {rel_path}")
            continue
            
        shape_info = mask.shape
        unique_vals = np.unique(mask)
        non_zero_count = np.count_nonzero(mask)
        total_count = mask.size
        ratio = (non_zero_count / total_count) * 100
        
        print(f"📄 标注文件: {rel_path}")
        print(f"   图像尺寸(H,W): {shape_info}")
        print(f"   唯一像素值列表: {unique_vals}")
        print(f"   非零(缺陷)像素数: {non_zero_count} / {total_count} (占比: {ratio:.5f}%)")
        
        if set(unique_vals).issubset({0, 1}):
            print(f"   STATUS: ✅ 完美合规！像素值为 [0, 1]")
        elif 255 in unique_vals or any(v > 1 for v in unique_vals):
            print(f"   STATUS: ⚠️ 像素值超出范围 (发现 >1 像素值如 {unique_vals[unique_vals>1]})！需修成 1！")
        elif len(unique_vals) == 1 and unique_vals[0] == 0:
            print(f"   STATUS: ℹ️ 纯黑底图 (没有标注任何缺陷)")
        print("-" * 65)

if __name__ == "__main__":
    debug_masks()
