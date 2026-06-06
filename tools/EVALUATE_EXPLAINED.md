# evaluate.py 验证函数详解

本文档详细解释 `evaluate.py` 中各个验证函数的区别、用途和使用场景。

## 概述

`evaluate.py` 提供了 **4 个主要验证函数**，它们的用途不同：

1. **`validate()`** - 基础验证函数（训练中使用）
2. **`validate_multiscale()`** - 多尺度验证函数
3. **`validate_on_benchmark()`** - 标准 Benchmark 验证函数
4. **`save_validation_images()`** - 保存验证图像函数

## 重要说明

**Q: 这几个验证函数都会执行吗？**
**A: 不会！训练过程中只会执行你在 `train.py` 中调用的函数。**

默认情况下，`train.py` **只调用 `validate()` 函数**。其他函数是工具函数，你可以手动调用或写脚本使用。

---

## 函数 1: `validate()` - 基础验证（训练中使用）⭐

### 功能
在单个验证集上评估模型的 PSNR。

### 何时使用
- **训练过程中**周期性自动调用（在 `train.py` 中）
- 快速评估模型在验证集上的性能

### 参数
```python
def validate(model, val_loader, config, verbose=True):
    """
    Args:
        model: 待评估的模型
        val_loader: 验证数据加载器
        config: 配置字典（包含 data_norm、eval_type 等）
        verbose: 是否显示进度条
    
    Returns:
        平均 PSNR 值（单个浮点数）
    """
```

### 在 train.py 中的使用
```python
# train.py 中的代码（已存在）
if val_loader is not None and (epoch + 1) % val_interval == 0:
    print('==> Validating...')
    val_metric = validate(model, val_loader, config)  # ← 这里调用
    writer.add_scalar('val/psnr', val_metric, epoch)
    log(f'Validation PSNR: {val_metric:.4f} dB')
```

### 特点
- ✅ **自动执行**（在训练循环中）
- ✅ 只在一个数据集上测试（通常是 Set5）
- ✅ 只在一个固定尺度上测试（config 中指定的 scale）
- ✅ 返回单个 PSNR 值

---

## 函数 2: `validate_multiscale()` - 多尺度验证

### 功能
在**多个缩放因子**（x2, x3, x4）上评估模型。

### 何时使用
- 训练完成后，想测试模型在不同尺度的表现
- 需要手动调用（不在 train.py 的训练循环中）

### 参数
```python
def validate_multiscale(model, dataset_configs, config, scales=[2, 3, 4]):
    """
    Args:
        model: 待评估的模型
        dataset_configs: 数据集配置字典
        config: 全局配置
        scales: 要测试的缩放因子列表（默认 [2, 3, 4]）
    
    Returns:
        字典: {'x2': psnr_x2, 'x3': psnr_x3, 'x4': psnr_x4}
    """
```

### 使用示例
```python
# 创建一个独立脚本 eval_multiscale.py
import torch
import models
from evaluate import validate_multiscale

# 加载模型
checkpoint = torch.load('save/train-div2k/checkpoint_best.pth')
model = models.make(checkpoint['model'], load_sd=True).cuda()

# 配置数据集
dataset_config = {
    'dataset': {
        'name': 'image-folder',
        'args': {'root_path': '/path/to/Set5/HR', 'cache': 'in_memory'}
    },
    'wrapper': {
        'name': 'sr-implicit-downsampled',
        'args': {'inp_size': None}
    }
}

config = {'data_norm': {'inp': {'sub': [0], 'div': [1]}, 'gt': {'sub': [0], 'div': [1]}}}

# 在多个尺度上验证
results = validate_multiscale(model, dataset_config, config, scales=[2, 3, 4])

print("Multi-scale Results:")
print(f"x2: {results['x2']:.4f} dB")
print(f"x3: {results['x3']:.4f} dB")
print(f"x4: {results['x4']:.4f} dB")
```

### 特点
- ❌ **不自动执行**（需要手动调用）
- ✅ 在多个尺度上测试
- ✅ 返回每个尺度的 PSNR 字典

---

## 函数 3: `validate_on_benchmark()` - 标准 Benchmark 验证

### 功能
在**多个标准 benchmark 数据集**（Set5, Set14, B100, Urban100）上评估模型。

### 何时使用
- 训练完成后，想在多个标准数据集上测试
- 与其他论文的结果进行对比
- 需要手动调用

### 参数
```python
def validate_on_benchmark(model, benchmark_paths, config, scale=4):
    """
    Args:
        model: 待评估的模型
        benchmark_paths: benchmark 数据集路径字典
                        例如: {'Set5': '/path/to/Set5', 'Set14': '/path/to/Set14'}
        config: 配置字典
        scale: 缩放因子
    
    Returns:
        字典: {'Set5': psnr_set5, 'Set14': psnr_set14, ...}
    """
```

### 使用示例
```python
# 创建一个独立脚本 eval_benchmarks.py
import torch
import models
from evaluate import validate_on_benchmark

# 加载模型
checkpoint = torch.load('save/train-div2k/checkpoint_best.pth')
model = models.make(checkpoint['model'], load_sd=True).cuda()

# 指定多个 benchmark 数据集路径
benchmark_paths = {
    'Set5': '/path/to/Set5/HR',
    'Set14': '/path/to/Set14/HR',
    'B100': '/path/to/B100/HR',
    'Urban100': '/path/to/Urban100/HR'
}

config = {'data_norm': {'inp': {'sub': [0], 'div': [1]}, 'gt': {'sub': [0], 'div': [1]}}}

# 在所有 benchmark 上验证（x4 尺度）
results = validate_on_benchmark(model, benchmark_paths, config, scale=4)

print("\n=== Benchmark Results (x4) ===")
for name, psnr in results.items():
    print(f"{name:12s}: {psnr:.4f} dB")
```

### 特点
- ❌ **不自动执行**（需要手动调用）
- ✅ 在多个标准数据集上测试
- ✅ 适合发表论文时的完整评估
- ✅ 返回每个数据集的 PSNR 字典

---

## 函数 4: `save_validation_images()` - 保存验证图像

### 功能
保存验证图像（LR, SR, HR）用于可视化对比。

### 何时使用
- 想查看模型的实际输出效果
- 制作可视化结果图
- 需要手动调用

### 参数
```python
def save_validation_images(model, val_loader, save_dir, config, num_images=5):
    """
    Args:
        model: 待评估的模型
        val_loader: 验证数据加载器
        save_dir: 保存目录
        config: 配置字典
        num_images: 保存图像的数量（默认 5）
    """
```

### 使用示例
```python
# 在训练完成后保存验证图像
import torch
import models
from torch.utils.data import DataLoader
import datasets
from evaluate import save_validation_images

# 加载模型
checkpoint = torch.load('save/train-div2k/checkpoint_best.pth')
model = models.make(checkpoint['model'], load_sd=True).cuda()

# 构建验证数据集
val_spec = {
    'dataset': {'name': 'image-folder', 'args': {'root_path': '/path/to/Set5/HR'}},
    'wrapper': {'name': 'sr-implicit-downsampled', 'args': {'scale_min': 4.0, 'scale_max': 4.0}}
}
val_dataset = datasets.make(val_spec['dataset'])
val_dataset = datasets.make(val_spec['wrapper'], args={'dataset': val_dataset})
val_loader = DataLoader(val_dataset, batch_size=1)

config = {'data_norm': {'inp': {'sub': [0], 'div': [1]}, 'gt': {'sub': [0], 'div': [1]}}}

# 保存验证图像
save_validation_images(model, val_loader, './validation_images', config, num_images=5)
```

### 特点
- ❌ **不自动执行**（需要手动调用）
- ✅ 保存 LR、SR、HR 三种图像
- ✅ 用于可视化和调试

---

## 训练中实际使用的验证流程

### train.py 中的验证代码（已存在）

```python
# 这是 train.py 中的代码片段
for epoch in range(start_epoch, num_epochs):
    # ... 训练代码 ...
    
    # 验证阶段（每隔 val_interval 个 epoch 执行一次）
    if val_loader is not None and (epoch + 1) % val_interval == 0:
        print('==> Validating...')
        
        # 只调用 validate() 函数
        val_metric = validate(model, val_loader, config)  # ← 只有这个被调用
        
        writer.add_scalar('val/psnr', val_metric, epoch)
        log(f'Validation PSNR: {val_metric:.4f} dB')
        
        # 保存最佳模型
        is_best = val_metric > best_metric
        if is_best:
            best_metric = val_metric
```

**结论：训练过程中只执行 `validate()` 函数，其他函数需要手动调用。**

---

## calc_psnr() 函数中的 eval_type 参数说明

这个参数控制 PSNR 计算时的**边界裁剪和颜色空间转换**：

### 选项 1: `eval_type: benchmark-4`
```yaml
eval_type: benchmark-4
```
- 裁剪边界：4 个像素（scale=4）
- 转换为灰度图（使用标准系数）
- 适用于：Set5, Set14, B100, Urban100

### 选项 2: `eval_type: div2k-4`
```yaml
eval_type: div2k-4
```
- 裁剪边界：10 个像素（scale=4 + 6）
- 保持 RGB 彩色
- 适用于：DIV2K 数据集

### 选项 3: `eval_type: null`（不指定）
```yaml
eval_type: null
```
- 不裁剪边界
- 保持 RGB 彩色
- 计算整张图像的 PSNR

---

## 实际使用建议

### 训练期间（自动）
- ✅ 使用 `validate()` - 在 train.py 中自动调用
- ✅ 在 Set5 上快速验证
- ✅ 使用 `eval_type: benchmark-4`

### 训练完成后（手动）

**1. 快速测试多尺度性能**
```python
python eval_multiscale.py  # 你需要创建这个脚本
```

**2. 完整 Benchmark 评估（用于论文）**
```python
python eval_benchmarks.py  # 你需要创建这个脚本
```

**3. 可视化结果**
```python
python save_images.py  # 你需要创建这个脚本
```

---

## 总结对比表

| 函数 | 自动执行 | 数据集数量 | 尺度数量 | 返回值 | 使用场景 |
|------|---------|----------|---------|--------|---------|
| `validate()` | ✅ 是 | 1 个 | 1 个 | 单个 PSNR | 训练中周期性验证 |
| `validate_multiscale()` | ❌ 否 | 1 个 | 多个 (x2,x3,x4) | PSNR 字典 | 测试多尺度性能 |
| `validate_on_benchmark()` | ❌ 否 | 多个 | 1 个 | PSNR 字典 | 完整 Benchmark 评估 |
| `save_validation_images()` | ❌ 否 | 1 个 | 1 个 | 无（保存图像） | 可视化结果 |

---

## 如何在配置文件中控制验证

在 `configs/train/train-div2k.yaml` 中：

```yaml
# 验证数据集（训练中使用）
val_dataset:
  dataset:
    name: image-folder
    args:
      root_path: /path/to/Set5/HR  # ← 只在这个数据集上验证
  wrapper:
    name: sr-implicit-downsampled
    args:
      scale_min: 4.0  # ← 只在 x4 尺度上验证
      scale_max: 4.0
  batch_size: 1

# 验证间隔
val_interval: 5  # ← 每 5 个 epoch 验证一次

# 评估类型
eval_type: benchmark-4  # ← PSNR 计算方式
```

这个配置只影响 `validate()` 函数，其他函数需要单独调用并传参。
