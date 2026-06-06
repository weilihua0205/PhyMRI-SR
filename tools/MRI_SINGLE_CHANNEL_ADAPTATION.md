# 单通道 MRI 图像训练适配说明

## 📋 修改总结

本文档记录了将 ContinuousSR 框架从 RGB 三通道自然图像适配到单通道 MRI 图像所做的全部修改。

---

## ✅ 已完成的修改

### 1. **数据加载层** ✓

#### `datasets/npy_folder.py`
- ✅ 实现了与 `image_folder.py` 完全一致的接口
- ✅ 支持 `cache` 模式：`'none'`, `'bin'`, `'in_memory'`
- ✅ 支持 `split_file`, `first_k`, `repeat` 参数
- ✅ 自动处理 2D 和 3D npy 数组，归一化为 `[C, H, W]` 格式

#### `datasets/wrappers.py`
- ✅ 修改 `resize_fn` 函数，支持单通道和多通道图像
- ✅ 自动检测并保持输入通道数不变
- ✅ 删除重复的函数定义，保持代码整洁

**关键特性**：
```python
# 单通道 MRI: [1, H, W] -> resize -> [1, H', W']
# 多通道 RGB: [3, H, W] -> resize -> [3, H', W']
```

---

### 2. **评估/验证层** ✓

#### `evaluate.py`
- ✅ `calc_psnr()`: 仅在多通道时转换为灰度，单通道直接计算
- ✅ `calc_ssim()`: 仅在多通道时转换为灰度，单通道直接计算
- ✅ `save_validation_images()`: 自动将单通道转为 3 通道用于可视化保存
- ✅ 更新函数文档说明单通道/多通道处理逻辑

**兼容性**：
- 单通道（C=1）：直接计算指标
- 多通道（C=3）：先转灰度再计算（benchmark 模式）

---

### 3. **模型层** ✓

#### `models/gaussian.py`
**关键修改**：
```python
class ContinuousGaussian(nn.Module):
    def __init__(self, encoder_spec, cnn_spec, fc_spec, output_channels=3, **kwargs):
        super().__init__()
        self.output_channels = output_channels  # 新增参数
        
        # MLP 输出通道从硬编码 3 改为可配置
        mlp_spec = {'name': 'mlp', 'args': {
            'in_dim': 256, 
            'out_dim': self.output_channels,  # 单通道=1，RGB=3
            ...
        }}
        
        # Background 从硬编码 3 改为可配置
        self.background = torch.ones(self.output_channels).cuda()
```

#### `models/swinir.py`
**关键修改**：
```python
@register('swinir')
def make_swinir(no_upsampling=True, in_chans=3, img_range=1.0, ...):
    return SwinIR(in_chans=in_chans, img_range=img_range, ...)
```

---

### 4. **配置文件** ✓

#### 新增配置：`configs/train/train-mri-single-channel.yaml`
**关键配置项**：
```yaml
# 数据集：使用配对 npy 文件
train_dataset:
  dataset:
    name: paired-npy-folders  # 配对 npy 数据集
    args:
      root_path_1: /path/to/LR  # LR 文件夹
      root_path_2: /path/to/HR  # HR 文件夹
  wrapper:
    name: sr-implicit-paired  # 配对模式（不做下采样）

# 模型：单通道配置
model:
  name: continuous-gaussian
  args:
    output_channels: 1  # 【关键】单通道输出
    encoder_spec:
      name: swinir
      args:
        in_chans: 1     # 【关键】单通道输入
        img_range: 1.0

# 损失：不使用 PerceptualLoss（VGG 需要 RGB）
loss:
  name: L1Loss  # 或 MSELoss, CharbonnierLoss
```

---

### 5. **测试工具** ✓

#### 新增测试脚本：`test_mri_single_channel.py`
**测试覆盖**：
- ✅ 数据加载（npy_folder）
- ✅ 数据预处理（resize_fn）
- ✅ 评估指标（PSNR/SSIM）
- ✅ 模型构建和前向传播
- ✅ 损失函数计算

**运行测试**：
```bash
python test_mri_single_channel.py
```

---

## 🚀 快速开始

### 步骤 1: 准备数据
```bash
# 确保你的 MRI 数据组织如下：
# data/
#   ├── train/
#   │   ├── LR/
#   │   │   ├── sample_001.npy  # shape: (1, H, W) 或 (H, W)
#   │   │   ├── sample_002.npy
#   │   │   └── ...
#   │   └── HR/
#   │       ├── sample_001.npy  # 与 LR 同名
#   │       ├── sample_002.npy
#   │       └── ...
#   └── val/
#       ├── LR/
#       └── HR/
```

**数据要求**：
- npy 文件格式，dtype=float32
- 值域归一化到 [0, 1]
- shape: `(H, W)` 或 `(1, H, W)` 或 `(C, H, W)`
- LR 和 HR 文件名必须一一对应

### 步骤 2: 修改配置文件
编辑 `configs/train/train-mri-single-channel.yaml`：
```yaml
train_dataset:
  dataset:
    args:
      root_path_1: /path/to/your/data/train/LR  # 修改为你的路径
      root_path_2: /path/to/your/data/train/HR  # 修改为你的路径

val_dataset:
  dataset:
    args:
      root_path_1: /path/to/your/data/val/LR
      root_path_2: /path/to/your/data/val/HR

# 根据你的 GPU 显存调整
batch_size: 4  # 建议从小值开始（2 或 4）
```

### 步骤 3: 运行测试
```bash
cd /home/ght/MRIxField/ContinuousSR-main_MRI

# 激活环境
conda activate continuous_sr

# 运行单通道测试
python test_mri_single_channel.py

# 如果所有测试通过，继续下一步
```

### 步骤 4: 开始训练
```bash
# 使用单 GPU 训练
python train.py \
  --config configs/train/train-mri-single-channel.yaml \
  --gpu 0 \
  --name mri_experiment_001

# 恢复训练
python train.py \
  --config configs/train/train-mri-single-channel.yaml \
  --resume save/mri_experiment_001/checkpoint_latest.pth \
  --gpu 0
```

### 步骤 5: 监控训练
```bash
# 查看训练日志
tail -f save/mri_experiment_001/log.txt

# 启动 TensorBoard
tensorboard --logdir save/mri_experiment_001 --port 6006

# 在浏览器中打开 http://localhost:6006
```

---

## 📊 预期效果

### 训练输出示例
```
==> Configuration:
model:
  name: continuous-gaussian
  args:
    output_channels: 1      # 单通道
    encoder_spec:
      name: swinir
      args:
        in_chans: 1         # 单通道输入
...

==> Model parameters: 12.34M
==> Loss function: L1Loss
==> Start training...

Epoch 1/500 - Loss: 0.0234, LR: 0.000100
==> Validating...
Validation PSNR: 28.45 dB, SSIM: 0.8532
New best model! PSNR: 28.45 dB, SSIM: 0.8532
```

### 验证指标
- **PSNR**: 通常在 25-35 dB 范围（取决于任务难度）
- **SSIM**: 通常在 0.80-0.95 范围
- 训练曲线应该平滑下降，无明显震荡

---

## ⚠️ 常见问题

### Q1: 训练时报 "通道数不匹配" 错误
**原因**：配置文件中忘记设置 `in_chans=1` 或 `output_channels=1`

**解决**：
```yaml
model:
  args:
    output_channels: 1  # 必须设置
    encoder_spec:
      args:
        in_chans: 1     # 必须设置
```

### Q2: 数据加载时报 shape 错误
**原因**：npy 文件 shape 不符合预期

**解决**：
```python
# 检查你的 npy 文件
import numpy as np
data = np.load('sample.npy')
print(data.shape)  # 应该是 (H, W) 或 (1, H, W)

# 如果是 (H, W, 1)，需要转置
if data.shape[-1] == 1:
    data = data.transpose(2, 0, 1)  # -> (1, H, W)
    np.save('sample.npy', data)
```

### Q3: 显存不足（OOM）
**解决**：
```yaml
# 减小 batch_size
batch_size: 2  # 从 4 降到 2

# 减小 patch 大小
train_dataset:
  wrapper:
    args:
      inp_size: 128  # 从 256 降到 128
```

### Q4: 训练不收敛或 loss 为 NaN
**可能原因**：
1. 数据归一化不正确
2. 学习率过大
3. 数据中有异常值（inf/nan）

**解决**：
```python
# 检查数据范围
import numpy as np
data = np.load('sample.npy')
print(f"Min: {data.min()}, Max: {data.max()}")
print(f"Has NaN: {np.isnan(data).any()}")
print(f"Has Inf: {np.isinf(data).any()}")

# 数据应该在 [0, 1] 范围内
assert data.min() >= 0 and data.max() <= 1
```

---

## 🔧 高级配置

### 自定义损失函数组合
```yaml
loss:
  name: CombinedLoss
  args:
    losses_dict:
      L1Loss: 1.0
      MSELoss: 0.5
      CharbonnierLoss: 0.3
```

### 调整学习率策略
```yaml
optimizer:
  args:
    lr: 5.0e-5  # 降低学习率

lr_scheduler:
  name: CosineAnnealingLR  # 使用余弦退火
  args:
    T_max: 500
    eta_min: 1.0e-6
```

### 数据增强策略
```yaml
train_dataset:
  wrapper:
    args:
      augment: true  # 启用增强
      # 包括：水平翻转、垂直翻转、对角翻转
```

---

## 📈 性能优化建议

### 1. 缓存策略
```yaml
# 小数据集（<100 samples）
dataset:
  args:
    cache: in_memory  # 全部加载到内存

# 中等数据集（100-1000 samples）
dataset:
  args:
    cache: bin  # 预处理后缓存为 .pkl

# 大数据集（>1000 samples）
dataset:
  args:
    cache: none  # 动态加载
```

### 2. 数据加载并行
```yaml
num_workers: 8  # 根据 CPU 核心数调整（建议 4-8）
```

### 3. 混合精度训练（可选）
修改 `train.py` 添加 AMP 支持（需要 PyTorch ≥ 1.6）

---

## 📝 总结

**所有修改点**：
1. ✅ `datasets/npy_folder.py` - 新建单通道 npy 数据集
2. ✅ `datasets/wrappers.py` - 修改 resize_fn 支持单通道
3. ✅ `evaluate.py` - 修改 PSNR/SSIM 计算适配单通道
4. ✅ `models/gaussian.py` - 添加 output_channels 参数
5. ✅ `models/swinir.py` - 添加 in_chans 参数
6. ✅ `configs/train/train-mri-single-channel.yaml` - 新建单通道配置
7. ✅ `test_mri_single_channel.py` - 新建测试脚本

**修改原则**：
- ✅ 最小侵入：仅修改必要的地方
- ✅ 向后兼容：保持对 RGB 三通道的支持
- ✅ 可配置：通过参数控制单通道/多通道
- ✅ 可测试：提供完整的测试工具

**下一步建议**：
1. 运行 `test_mri_single_channel.py` 验证所有修改
2. 用小数据集（10-20 samples）快速测试训练流程
3. 确认训练正常后，使用完整数据集训练
4. 定期监控训练曲线和验证指标
5. 保存最佳模型用于推理

祝训练顺利！🎉
