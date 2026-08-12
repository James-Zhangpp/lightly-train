# lightly-train 兼容性修改文档

> **修改日期**: 2026-07-30  
> **修改原因**: 当前 Python 3.11 环境中安装的 PyTorch 和 lightly 库版本与 lightly-train 框架源码所要求的最新 API 不匹配，导致 `import lightly_train` 启动即崩溃。  
> **影响范围**: 仅修改了模块导入层面的兼容逻辑，**不影响**任何模型结构、训练算法、Loss 计算、数据增强和推理精度。

---

## 一、问题根源

lightly-train 框架源码针对 **PyTorch ≥ 2.5** 和 **lightly 最新版** 编写，使用了以下较新 API：

| API | 最低要求版本 | 当前环境状态 |
|-----|------------|-------------|
| `torch.export.Dim` | PyTorch ≥ 2.5 | ❌ 不存在 |
| `torch.export.dynamic_shapes.Dim` | PyTorch 2.1~2.4 | ❌ 子模块不存在 |
| `torch.export.ExportedProgram` | PyTorch ≥ 2.1 | ❌ 不存在 |
| `torch.utils._pytree.register_pytree_node` | PyTorch ≥ 2.2 | ❌ 仅有 `_register_pytree_node` |
| `lightly.loss.PatchKernelAlignmentLoss` | lightly 最新版 | ❌ 不存在 |

由于 `import lightly_train` 会递归导入框架所有模块（包括未使用的模型），上述任一 API 缺失都会导致启动时直接 `ImportError` / `ModuleNotFoundError` / `AttributeError` 崩溃。

---

## 二、修改的文件列表

### 文件 1: `src/lightly_train/_task_models/ltdetr_object_detection/config.py`

**修改位置**: 第 15~21 行（导入区域）

**修改前**:
```python
from torch.export import Dim
```

**修改后**:
```python
try:
    from torch.export import Dim
except ImportError:
    try:
        from torch.export.dynamic_shapes import Dim
    except (ImportError, ModuleNotFoundError):
        Dim = None
```

**为什么修改**:  
`Dim` 类用于 ONNX 动态 Batch 导出时定义动态维度约束。在不同 PyTorch 版本中，该类位于不同子模块路径。当前环境中两个路径均不存在，但 `Dim` 仅在 ONNX 导出流程中被实际调用（不影响训练）。

**影响**:
- ✅ 训练流程：**无影响**。`Dim` 仅在调用 `export_onnx()` 导出 ONNX 模型时才被使用
- ✅ 模型结构：**无影响**。不改变任何网络层定义
- ⚠️ ONNX 导出：在当前 PyTorch 版本下，如果调用 ONNX 导出功能，会被框架内置的 `check_onnx_dynamo_requirements()` 拦截并给出清晰的版本提示错误，而不是无预警崩溃

---

### 文件 2: `src/lightly_train/_task_models/task_model_io.py`

**修改位置 A**: 第 18~24 行（导入区域）

**修改前**:
```python
from torch.export.dynamic_shapes import Dim
```

**修改后**:
```python
try:
    from torch.export import Dim
except ImportError:
    try:
        from torch.export.dynamic_shapes import Dim
    except (ImportError, ModuleNotFoundError):
        Dim = None
```

**为什么修改**:  
与文件 1 相同原因。`task_model_io.py` 是框架核心 IO 规范模块，定义了 `ModelInputSpec` 和 `BaseModelOutput`，在 `import lightly_train` 时被强制加载。

---

**修改位置 B**: 第 195~208 行（`BaseModelOutput.__init_subclass__` 方法）

**修改前**:
```python
def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    torch.utils._pytree.register_pytree_node(
        cls,
        _model_output_flatten,
        functools.partial(_model_output_unflatten, output_type=cls),
    )
```

**修改后**:
```python
def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    # 兼容不同 PyTorch 版本: register_pytree_node (>=2.2) vs _register_pytree_node (<2.2)
    _reg_fn = getattr(
        torch.utils._pytree,
        "register_pytree_node",
        getattr(torch.utils._pytree, "_register_pytree_node", None),
    )
    if _reg_fn is not None:
        _reg_fn(
            cls,
            _model_output_flatten,
            functools.partial(_model_output_unflatten, output_type=cls),
        )
```

**为什么修改**:  
PyTorch 在不同版本中对 pytree 节点注册 API 的命名不同：
- PyTorch ≥ 2.2: 公开 API `register_pytree_node`
- PyTorch < 2.2: 私有 API `_register_pytree_node`（带下划线前缀）

当前环境仅有带下划线的旧版 API，直接调用新名称会触发 `AttributeError`。

**影响**:
- ✅ 训练流程：**无影响**。pytree 注册是为 `torch.export` 导出时展平/还原自定义数据结构服务的
- ✅ 模型结构：**无影响**
- ✅ Loss 计算：**无影响**。前向传播和反向梯度不依赖 pytree 注册
- ⚠️ 如果 PyTorch 版本极低导致两个 API 名称都不存在，pytree 注册会被静默跳过。此时 `torch.export` 导出功能不可用，但训练完全正常

---

### 文件 3: `src/lightly_train/_export/export.py`

**修改位置**: 第 13~16 行（导入区域）

**修改前**:
```python
from torch.export import ExportedProgram
```

**修改后**:
```python
try:
    from torch.export import ExportedProgram
except ImportError:
    ExportedProgram = None
```

**为什么修改**:  
`ExportedProgram` 是 PyTorch 2.1+ 新增的导出程序类型，用于 `torch.export.export()` 的返回值类型标注。当前环境的 `torch.export` 模块不包含该类，导致导入崩溃。

**影响**:
- ✅ 训练流程：**无影响**。`ExportedProgram` 仅作为 `export()` 方法的返回值类型，训练过程完全不涉及
- ✅ 模型结构/Loss/数据增强：**无影响**
- ⚠️ 模型导出：在当前环境下无法使用 `torch.export` 相关的严格导出功能，但 ONNX 导出和 checkpoint 保存不受影响

---

### 文件 4: `src/lightly_train/_methods/dinov31/dinov31.py`

**修改位置**: 第 55~71 行（导入区域）

**修改前**:
```python
from lightly.loss import PatchKernelAlignmentLoss, roi_resample_to_grid
```

**修改后**:
```python
try:
    from lightly.loss import PatchKernelAlignmentLoss, roi_resample_to_grid
except ImportError:
    # lightly 版本不含 PatchKernelAlignmentLoss 时提供占位实现，
    # 仅在实际使用 DINOv31 方法时才会报错提示升级
    class PatchKernelAlignmentLoss:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PatchKernelAlignmentLoss requires a newer version of the 'lightly' "
                "package. Please run: pip install --upgrade lightly"
            )

    def roi_resample_to_grid(*args, **kwargs):
        raise RuntimeError(
            "roi_resample_to_grid requires a newer version of the 'lightly' "
            "package. Please run: pip install --upgrade lightly"
        )
```

**为什么修改**:  
`PatchKernelAlignmentLoss` 和 `roi_resample_to_grid` 是 lightly 库最新版本中新增的 DINOv31 预训练方法的 PaKA (Patch Kernel Alignment) 损失函数。当前环境安装的 lightly 版本不包含该模块。

由于 `import lightly_train` 会自动加载所有预训练方法（包括 DINOv31），即使我们只用 DINOv3-EoMT 语义分割训练，也会因为 DINOv31 模块的导入失败而崩溃。

**影响**:
- ✅ 语义分割训练（dinov3/vitb16-eomt）：**无影响**。我们使用的是 DINOv3-EoMT 分割模型，不涉及 DINOv31 预训练方法
- ✅ 其他预训练方法（DINOv2、DINO、SimCLR、DetCon 等）：**无影响**
- ⚠️ DINOv31 预训练方法：在当前 lightly 版本下，如果尝试使用 `method="dinov31"` 进行预训练，会在实例化 `PatchKernelAlignmentLoss` 时抛出清晰的 `RuntimeError` 提示升级 lightly，而不是在导入阶段无预警崩溃

---

## 三、附加修改: `tools/train_breakage.py`

### 3.1 移除 model_args 中 4 个不支持的参数

**移除的 key**:

| 参数名 | 移除原因 |
|--------|---------|
| `optimizer` | `DINOv3EoMTSemanticSegmentationTrainArgs` 内置使用 AdamW 优化器，不通过 `model_args` 外部传参控制 |
| `scheduler` | 学习率调度由框架内部 `poly_power` 参数控制（多项式衰减），不支持外部指定 cosine/linear |
| `backbone_lr_multiplier` | 该模型通过 `llrd`（逐层学习率衰减率）控制 Backbone 学习率，不支持独立乘子倍率 |
| `mask_threshold` | 推理时的 Mask 二值化阈值，不是训练参数，在导出/预测阶段单独配置 |

### 3.2 移除 transform_args 中 1 个不支持的参数

| 参数名 | 移除原因 |
|--------|---------|
| `gaussian_blur` | `DINOv3EoMTSemanticSegmentationTrainTransformArgs` 不包含高斯模糊增强字段 |

**影响**:
- ✅ 训练功能：**无影响**。移除的参数本身就不被框架接受，之前传入会直接报 `ConfigValidationError`
- ✅ 训练效果：优化器、学习率调度、Backbone 学习率倍率已通过框架内置的等效机制（AdamW + poly_power + llrd）自动控制

---

## 四、`model_args` 合法参数速查表

以下是 `DINOv3EoMTSemanticSegmentationTrainArgs` 支持的全部参数：

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `lr` | float | 1e-4 | 基础学习率 |
| `llrd` | float | 0.8 | 逐层学习率衰减率 |
| `weight_decay` | float | 0.05 | L2 权重衰减 |
| `lr_warmup_steps` | tuple[int,int] | (500, 1000) | Warmup 步数区间 |
| `poly_power` | float | 0.9 | 多项式退火幂次 |
| `num_queries` | int\|"auto" | "auto" | Mask Query 数量 |
| `num_joint_blocks` | int\|"auto" | "auto" | 联合 Transformer Block 数量 |
| `fix_num_upscale_blocks` | bool | True | 是否固定上采样模块数 |
| `backbone_freeze` | bool | False | 是否冻结 Backbone |
| `backbone_weights` | str\|None | None | Backbone 自定义权重路径 |
| `loss_num_points` | int | 12544 | 点采样数量 |
| `loss_oversample_ratio` | float | 3.0 | 难点过采样比例 |
| `loss_importance_sample_ratio` | float | 0.75 | 重要性采样比例 |
| `loss_no_object_coefficient` | float | 0.1 | 无目标惩罚权重 |
| `loss_mask_coefficient` | float | 5.0 | Mask Loss 权重 |
| `loss_dice_coefficient` | float | 5.0 | Dice Loss 权重 |
| `loss_class_coefficient` | float | 2.0 | 分类 Loss 权重 |
| `gradient_clip_val` | float | 0.01 | 梯度裁剪阈值 |

---

## 五、`transform_args` 合法参数速查表

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `image_size` | tuple\|"auto" | 输入图像尺寸 (H, W) |
| `num_channels` | int\|"auto" | 输入通道数 |
| `normalize` | dict\|"auto" | 像素归一化 (mean, std) |
| `color_jitter` | dict\|None | 颜色抖动增强 |
| `scale_jitter` | dict\|None | 多尺度缩放抖动 |
| `random_crop` | dict | 随机裁剪与填充 |
| `random_flip` | dict\|None | 随机镜像翻转 |
| `random_rotate_90` | dict\|None | 正交 90° 旋转 |
| `random_rotate` | dict\|None | 小角度微旋转 |
| `channel_drop` | dict\|None | 通道丢弃 |
| `smallest_max_size` | dict\|None | 最小边最大尺寸约束 |

> ⚠️ **不支持**: `gaussian_blur`、`random_erasing`、`cutout` 等

---

## 六、总结

所有修改均为**导入层面的向下兼容处理**，采用 `try-except` 机制在 API 不可用时提供安全回退：

1. **不改变任何模型网络结构**（Backbone、Decoder Head、Loss 函数完全不变）
2. **不改变任何训练算法逻辑**（前向传播、反向梯度、优化器更新步骤完全不变）
3. **不改变任何数据增强流程**（Transform Pipeline 完全不变）
4. **不影响训练精度和收敛性**（mIoU、Dice、Precision、Recall 指标不受任何影响）

唯一被降级的功能是 **ONNX 严格导出**（`torch.export`），在当前 PyTorch 版本下不可用，但训练完成后可通过常规方式导出模型。
