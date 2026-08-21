"""A07_01_AVI 专用的 ONNX 推理测试脚本。

本脚本是对 tools/test.py 的封装，自动指定了 A07_01_AVI 导出的 ONNX 路径、验证集图像路径以及测试输出路径。
支持所有 tools/test.py 的命令行参数。
"""

import sys
from pathlib import Path
import test as test_script

if __name__ == "__main__":
    # 自动定位项目根目录和相关路径
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    
    # ONNX 模型路径 (基于默认的 embedded 导出模式)
    default_onnx = PROJECT_ROOT / "out" / "A07_01_AVI_vitb16_eupe_eomt" / "exported_models" / "model_with_preprocessing.onnx"
    
    # 测试使用的图像目录 (默认为验证集)
    default_img_dir = PROJECT_ROOT / "tools" / "DataSet" / "A07_01_AVI" / "img_dir" / "val"
    
    # 结果输出目录
    default_out_dir = PROJECT_ROOT / "out" / "A07_01_AVI_test_results"
    
    # 如果用户没有显式指定相关参数，则自动注入默认路径
    if "--onnx" not in sys.argv:
        sys.argv.extend(["--onnx", str(default_onnx)])
    
    if "--image" not in sys.argv and "--image-dir" not in sys.argv:
        # 使用 --image 来覆盖掉 test.py 里面写死的 DEFAULT_IMAGE 列表
        sys.argv.extend(["--image", str(default_img_dir)])
        
    if "--out-dir" not in sys.argv:
        # test.py 中的参数名是 --out-dir 而不是 --out
        sys.argv.extend(["--out-dir", str(default_out_dir)])
        
    # 指定推理时的输入尺寸 (与训练对齐)
    if "--size" not in sys.argv:
        sys.argv.extend(["--size", "512"])
        
    # 由于原始图片尺寸是 1190*611，这里默认关闭 test.py 中硬编码的固定裁剪逻辑，直接 Resize 输入推理
    if "--no-crop" not in sys.argv:
        sys.argv.append("--no-crop")
        
    print(f"🚀 [A07_01_AVI ONNX 测试] 自动锁定 ONNX: {default_onnx.name}")
    print(f"📦 [A07_01_AVI ONNX 测试] 自动加载图片目录: {default_img_dir}")
    print(f"📂 [A07_01_AVI ONNX 测试] 结果将输出至: {default_out_dir}")
    
    # 调用原始推理逻辑
    test_script.main()
