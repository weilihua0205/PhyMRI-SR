# ContinuousSR 训练指南

本指南帮助你快速开始训练 ContinuousSR 模型。

## 1. 数据准备

### 1.1 下载数据集

**训练数据集：DIV2K**
- 官网：https://data.vision.ee.ethz.ch/cvl/DIV2K/
- 下载 DIV2K_train_HR.zip（800 张 2K 分辨率图像）
- 解压到：`/path/to/DIV2K/train/HR/`

**验证数据集：Set5 / Set14**
- 下载地址：https://github.com/xinntao/BasicSR/blob/master/docs/DatasetPreparation.md
- Set5：5 张测试图像
- Set14：14 张测试图像
- 解压到：`/path/to/Set5/HR/` 和 `/path/to/Set14/HR/`

### 1.2 目录结构

```
/path/to/datasets/
├── DIV2K/
│   └── train/
│       └── HR/
│           ├── 0001.png
│           ├── 0002.png
│           └── ... (800 images)
├── Set5/
│   └── HR/
│       ├── baby.png
│       ├── bird.png
│       ├── butterfly.png
│       ├── head.png
│       └── woman.png
└── Set14/
    └── HR/
        ├── baboon.png
        ├── barbara.png
        └── ... (14 images)
```

## 2. 配置文件修改

打开 `configs/train/train-div2k.yaml`，修改以下路径：

```yaml
train_dataset:
  dataset:
    args:
      root_path: /path/to/DIV2K/train/HR  # 修改为实际路径

val_dataset:
  dataset:
    args:
      root_path: /path/to/Set5/HR  # 修改为实际路径
```

## 3. 开始训练

### 3.1 使用 Python 命令训练（推荐，跨平台）

**基础训练：**
```bash
# 使用默认配置训练
python train.py --config configs/train/train-div2k.yaml --gpu 0

# 指定实验名称
python train.py --config configs/train/train-div2k.yaml --gpu 0 --name my_experiment

# 添加标签
python train.py --config configs/train/train-div2k.yaml --gpu 0 --name div2k --tag v1
```

### 3.2 使用启动脚本（更方便）

**Linux/Mac 系统：**
```bash
# 首次使用需要赋予执行权限
chmod +x train.sh

# 基础训练
./train.sh -c configs/train/train-div2k.yaml -g 0

# 带实验名称
./train.sh -c configs/train/train-div2k.yaml -g 0 -n my_experiment

# 快速训练模式
./train.sh -q

# 环境检查
./train.sh -k

# 查看帮助
./train.sh -h
```

**Windows PowerShell：**
```powershell
# 基础训练
.\train.ps1 -Config configs\train\train-div2k.yaml -GPU 0

# 带实验名称
.\train.ps1 -Config configs\train\train-div2k.yaml -GPU 0 -Name my_experiment

# 快速训练模式
.\train.ps1 -Quick

# 环境检查
.\train.ps1 -Check
```

### 3.3 快速验证训练流程

如果你想快速验证训练流程是否正常（使用小数据集）：

```bash
# Python 命令
python train.py --config configs/train/train-quick.yaml --gpu 0

# 或使用脚本（Linux/Mac）
./train.sh -q

# 或使用脚本（Windows）
.\train.ps1 -Quick
```

### 3.4 从 checkpoint 恢复训练

**Linux/Mac:**
```bash
# 使用 Python 命令
python train.py --config configs/train/train-div2k.yaml --gpu 0 \
    --resume ./save/train-div2k/checkpoint_latest.pth

# 或使用脚本
./train.sh -r save/train-div2k/checkpoint_latest.pth
```

**Windows:**
```powershell
# 使用 Python 命令
python train.py --config configs/train/train-div2k.yaml --gpu 0 `
    --resume .\save\train-div2k\checkpoint_latest.pth

# 或使用脚本
.\train.ps1 -Resume save\train-div2k\checkpoint_latest.pth
```

### 3.5 多 GPU 训练（待实现）

**Linux/Mac:**
```bash
# 使用 GPU 0 和 1
CUDA_VISIBLE_DEVICES=0,1 python train.py --config configs/train/train-div2k.yaml
```

**Windows:**
```powershell
# 设置环境变量
$env:CUDA_VISIBLE_DEVICES="0,1"
python train.py --config configs/train/train-div2k.yaml
```

## 4. 监控训练

### 4.1 查看日志

训练日志保存在：`./save/<experiment_name>/log.txt`

```bash
tail -f ./save/train-div2k/log.txt
```

### 4.2 TensorBoard 可视化

```bash
tensorboard --logdir ./save/train-div2k/tensorboard --port 6006
```

然后在浏览器打开：http://localhost:6006

可以查看：
- 训练损失曲线
- 验证 PSNR 曲线
- 学习率变化

## 5. Checkpoint 说明

训练过程中会自动保存以下 checkpoint：

- `checkpoint_latest.pth`：最新的模型（每个 epoch 更新）
- `checkpoint_best.pth`：验证 PSNR 最高的模型
- `checkpoint_epoch_N.pth`：每隔 N 个 epoch 保存（根据 `save_interval` 设置）

Checkpoint 包含：
- 模型权重（`model['sd']`）
- 优化器状态（`optimizer`）
- 调度器状态（`scheduler`）
- 当前 epoch（`epoch`）
- 最佳验证指标（`best_metric`）

## 6. 配置文件说明

项目提供了 3 个预设配置：

### 6.1 `train-div2k.yaml`（推荐）
- 数据集：DIV2K（800 张图像）
- 训练轮数：1000 epochs
- Batch size：16
- 学习率：1e-4（MultiStepLR 调度）
- 适用于：正式训练高质量模型

### 6.2 `train-quick.yaml`
- 数据集：Set14（14 张图像，重复 20 次）
- 训练轮数：200 epochs
- Batch size：8
- 适用于：快速验证训练流程、调试

### 6.3 `train-advanced.yaml`
- 更大的模型（32 ResBlocks, 256 features）
- 更大的 patch size（64）
- Charbonnier Loss
- CosineAnnealing 学习率
- 适用于：追求更高性能

## 7. 超参数调整建议

### 7.1 根据显存调整 batch_size

| 显存大小 | 推荐 batch_size | inp_size |
|---------|----------------|----------|
| 8GB     | 4-8            | 48       |
| 12GB    | 8-16           | 48       |
| 16GB    | 16-24          | 48-64    |
| 24GB+   | 32+            | 64-96    |

### 7.2 学习率调整

- 如果训练不稳定：降低学习率（1e-5）
- 如果收敛太慢：增加学习率（2e-4）
- 如果使用更大的 batch_size：按比例增加学习率

### 7.3 损失函数选择

- **L1Loss**（推荐）：收敛稳定，结果清晰
- **MSELoss**：可能过平滑
- **CharbonnierLoss**：介于 L1 和 L2 之间，效果平衡

## 8. 训练完成后

### 8.1 测试模型

```bash
python test.py --config configs/test/test-set5-4.yaml \
    --model ./save/train-div2k/checkpoint_best.pth
```

### 8.2 推理单张图像

```bash
python demo.py \
    --input butterflyx4.png \
    --model ./save/train-div2k/checkpoint_best.pth \
    --scale 4,4 \
    --output output.png
```

### 8.3 在多个 benchmark 上评估

创建评估脚本 `eval_benchmarks.py`：

```python
from evaluate import validate_on_benchmark
import torch
import models

# 加载模型
checkpoint = torch.load('./save/train-div2k/checkpoint_best.pth')
model = models.make(checkpoint['model'], load_sd=True).cuda()

# 评估
benchmark_paths = {
    'Set5': '/path/to/Set5/HR',
    'Set14': '/path/to/Set14/HR',
    'B100': '/path/to/B100/HR',
    'Urban100': '/path/to/Urban100/HR'
}

config = {'data_norm': {'inp': {'sub': [0], 'div': [1]}, 'gt': {'sub': [0], 'div': [1]}}}

results = validate_on_benchmark(model, benchmark_paths, config, scale=4)

print('\n=== Final Results ===')
for name, psnr in results.items():
    print(f'{name}: {psnr:.4f} dB')
```

## 9. 常见问题

### 9.1 CUDA Out of Memory
- 减小 `batch_size`
- 减小 `inp_size`
- 减少模型层数（`n_resblocks`）

### 9.2 训练很慢
- 增加 `num_workers`（数据加载并行）
- 使用 `cache: in_memory`（加载数据到内存）
- 减小验证频率（`val_interval`）

### 9.3 验证 PSNR 不提升
- 检查数据路径是否正确
- 检查学习率是否太大/太小
- 增加训练轮数
- 尝试不同的损失函数

### 9.4 模型无法加载
- 检查 checkpoint 格式是否正确
- 确认 model_spec 中的参数与当前配置一致

## 10. 高级功能（可选实现）

- [ ] 混合精度训练（AMP）
- [ ] 分布式训练（DDP）
- [ ] 感知损失（Perceptual Loss）
- [ ] EMA 模型
- [ ] 动态学习率（ReduceLROnPlateau）

## 联系与支持

如有问题，请：
1. 检查日志文件：`./save/<experiment_name>/log.txt`
2. 查看 TensorBoard 的训练曲线
3. 参考项目 README 和论文

祝训练顺利！🚀
