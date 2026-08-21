"""A07_01_AVI 专用的 ONNX 导出脚本。

本脚本是对 tools/toOnnx.py 的封装，自动指定了针对 A07_01_AVI 数据集训练所产生的权重路径。
支持所有 tools/toOnnx.py 的命令行参数（例如 --mode official 等）。
"""

import sys
from pathlib import Path
import toOnnx

if __name__ == "__main__":
    # 自动定位项目根目录和 A07_01_AVI 的权重文件
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    best_checkpoint = PROJECT_ROOT / "out" / "A07_01_AVI_vitb16_eupe_eomt" / "exported_models" / "exported_best.pt"
    
    # 如果用户没有通过命令行显式指定 --checkpoint，则自动注入默认路径
    if "--checkpoint" not in sys.argv:
        sys.argv.extend(["--checkpoint", str(best_checkpoint)])
        
    # 如果没有指定 height 和 width，自动指定为 512 (与训练默认保持一致)
    if "--height" not in sys.argv:
        sys.argv.extend(["--height", "512"])
    if "--width" not in sys.argv:
        sys.argv.extend(["--width", "512"])
        
    print(f"🚀 [A07_01_AVI ONNX 导出] 自动锁定权重文件: {best_checkpoint}")
    
    # 调用原始导出逻辑
    toOnnx.main()
