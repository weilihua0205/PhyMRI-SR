# 训练模块文件清单

本文档列出为 ContinuousSR 项目补全的所有训练相关文件。

## 核心训练文件

### 1. `train.py` - 训练主脚本
**功能**：
- 训练循环的核心控制器
- 支持从 checkpoint 恢复训练
- 自动保存最佳模型和定期 checkpoint
- TensorBoard 日志记录
- 验证集评估

**使用方法**：
```bash
python train.py --config configs/train/train-div2k.yaml --gpu 0
```

**命令行参数**：
- `--config`: 训练配置文件（必需）
- `--resume`: 恢复训练的 checkpoint 路径
- `--gpu`: GPU 设备 ID
- `--name`: 实验名称
- `--tag`: 实验标签

---

### 2. `losses.py` - 损失函数模块
**功能**：
- 提供多种损失函数选项
- 注册器模式，易于扩展

**支持的损失函数**：
- `L1Loss` - L1 损失（推荐）
- `MSELoss` - MSE 损失
- `SmoothL1Loss` - Smooth L1 损失
- `CharbonnierLoss` - Charbonnier 损失
- `PSNRLoss` - PSNR 损失
- `CombinedLoss` - 组合损失
- `PerceptualLoss` - 感知损失（需要 VGG）
- `EdgeLoss` - 边缘保持损失

**使用示例**：
```python
from losses import make as make_loss

# 单个损失
criterion = make_loss({'name': 'L1Loss'})

# 组合损失
criterion = make_loss({
    'name': 'CombinedLoss',
    'args': {
        'losses_dict': {'L1Loss': 1.0, 'CharbonnierLoss': 0.5}
    }
})
```

---

### 3. `schedulers.py` - 学习率调度器
**功能**：
- 提供多种学习率调整策略
- 支持 Warmup 机制

**支持的调度器**：
- `StepLR` - 固定步长衰减
- `MultiStepLR` - 多阶段衰减（推荐）
- `ExponentialLR` - 指数衰减
- `CosineAnnealingLR` - 余弦退火
- `CosineAnnealingWarmRestarts` - 带重启的余弦退火
- `ReduceLROnPlateau` - 基于验证指标动态调整
- `Warmup` - Warmup 包装器

**使用示例**：
```python
from schedulers import make_scheduler

scheduler = make_scheduler(optimizer, {
    'name': 'MultiStepLR',
    'args': {
        'milestones': [200, 400, 600],
        'gamma': 0.5
    }
})
```

---

### 4. `evaluate.py` - 验证/评估模块
**功能**：
- 在验证集上评估模型 PSNR
- 支持多尺度验证
- 支持标准 benchmark 评估
- 保存验证图像

**主要函数**：
- `validate()` - 基础验证函数
- `calc_psnr()` - PSNR 计算
- `validate_multiscale()` - 多尺度验证
- `validate_on_benchmark()` - Benchmark 评估
- `save_validation_images()` - 保存验证图像

**使用示例**：
```python
from evaluate import validate

psnr = validate(model, val_loader, config)
print(f'Validation PSNR: {psnr:.4f} dB')
```

---

## 配置文件

### 5. `configs/train/train-div2k.yaml`
**用途**：标准训练配置（推荐）
**数据集**：DIV2K (800 images)
**训练轮数**：1000 epochs
**适用场景**：正式训练高质量模型

### 6. `configs/train/train-quick.yaml`
**用途**：快速验证配置
**数据集**：Set14 (14 images × 20 repeats)
**训练轮数**：200 epochs
**适用场景**：快速验证训练流程、调试

### 7. `configs/train/train-advanced.yaml`
**用途**：高级训练配置
**特点**：
- 更大的模型（32 ResBlocks, 256 features）
- Charbonnier Loss
- CosineAnnealing 学习率
**适用场景**：追求更高性能

---

## 文档和工具

### 8. `TRAINING.md` - 详细训练指南
**内容**：
- 数据准备步骤
- 配置文件说明
- 训练命令示例
- 超参数调整建议
- 常见问题解决方案
- 训练后的评估方法

### 9. `QUICKSTART.md` - 快速启动指南
**内容**：
- 6 步快速开始训练
- 简化的操作流程
- 常见问题速查

### 10. `check_training_env.py` - 环境检查脚本
**功能**：
- 检查依赖库是否安装
- 检查 CUDA 可用性
- 验证配置文件格式
- 检查数据集路径是否存在
- 估算显存需求

**使用方法**：
```bash
python check_training_env.py --config configs/train/train-div2k.yaml
```

### 11. `train.ps1` - PowerShell 启动脚本（Windows）
**功能**：
- 简化 Windows 用户的训练启动
- 交互式确认
- 参数验证
- 颜色输出

**使用方法**：
```powershell
# 基础训练
.\train.ps1 -Config configs\train\train-div2k.yaml -GPU 0

# 快速训练
.\train.ps1 -Quick

# 环境检查
.\train.ps1 -Check

# 恢复训练
.\train.ps1 -Resume save\train-div2k\checkpoint_latest.pth
```

### 12. `train.sh` - Bash 启动脚本（Linux/Mac）⭐ 新增
**功能**：
- 简化 Linux/Mac 用户的训练启动
- 交互式确认
- 参数验证
- 颜色输出
- GPU 检测

**使用方法**：
```bash
# 首次使用需要赋予执行权限
chmod +x train.sh

# 基础训练
./train.sh -c configs/train/train-div2k.yaml -g 0

# 快速训练
./train.sh -q

# 环境检查
./train.sh -k

# 恢复训练
./train.sh -r save/train-div2k/checkpoint_latest.pth

# 查看帮助
./train.sh -h
```

**命令行参数**：
- `-c, --config`: 配置文件路径
- `-g, --gpu`: GPU 设备 ID
- `-n, --name`: 实验名称
- `-r, --resume`: 从 checkpoint 恢复
- `-q, --quick`: 使用快速配置
- `-k, --check`: 仅运行环境检查
- `-h, --help`: 显示帮助信息

---

## 文件结构总览

```
ContinuousSR-main/
├── train.py                    # 训练主脚本 ⭐
├── losses.py                   # 损失函数模块 ⭐
├── schedulers.py               # 学习率调度器 ⭐
├── evaluate.py                 # 验证/评估模块 ⭐
├── check_training_env.py       # 环境检查脚本
├── train.ps1                   # PowerShell 启动脚本（Windows）
├── train.sh                    # Bash 启动脚本（Linux/Mac）⭐ 新增
├── TRAINING.md                 # 详细训练指南（已更新跨平台）
├── QUICKSTART.md               # 快速启动指南（已更新跨平台）
├── configs/
│   └── train/
│       ├── train-div2k.yaml    # 标准配置 ⭐
│       ├── train-quick.yaml    # 快速测试配置
│       └── train-advanced.yaml # 高级配置
└── save/                       # 训练输出目录（自动创建）
    └── <experiment_name>/
        ├── checkpoint_latest.pth
        ├── checkpoint_best.pth
        ├── checkpoint_epoch_*.pth
        ├── log.txt
        ├── config.yaml
        └── tensorboard/
```

---

## 快速开始清单

完成以下步骤即可开始训练：

- [ ] **步骤 1**: 下载 DIV2K 和 Set5 数据集
- [ ] **步骤 2**: 修改 `configs/train/train-div2k.yaml` 中的数据路径
- [ ] **步骤 3**: 运行环境检查
  ```bash
  python check_training_env.py
  ```
- [ ] **步骤 4**: 开始训练
  ```bash
  # 跨平台方式（推荐）
  python train.py --config configs/train/train-div2k.yaml --gpu 0
  
  # Linux/Mac 使用脚本
  chmod +x train.sh  # 首次使用
  ./train.sh -c configs/train/train-div2k.yaml -g 0
  
  # Windows 使用脚本
  .\train.ps1 -Config configs\train\train-div2k.yaml -GPU 0
  ```
- [ ] **步骤 5**: 监控训练（TensorBoard）
  ```bash
  tensorboard --logdir ./save/train-div2k/tensorboard
  ```
- [ ] **步骤 6**: 测试模型
  ```bash
  python test.py --config configs/test/test-set5-4.yaml \
      --model ./save/train-div2k/checkpoint_best.pth
  ```

---

## 关键特性

### ✅ 已实现
- [x] 完整的训练循环（train.py）
- [x] 多种损失函数（losses.py）
- [x] 多种学习率调度器（schedulers.py）
- [x] 验证和评估（evaluate.py）
- [x] Checkpoint 保存/恢复
- [x] TensorBoard 日志
- [x] 梯度裁剪
- [x] 数据增强（在 wrappers.py 中）
- [x] 多 GPU 支持（基础）
- [x] 配置文件系统
- [x] 环境检查工具
- [x] 详细文档

### 🔄 可选扩展（未实现）
- [ ] 混合精度训练（AMP）
- [ ] 分布式训练（DDP）
- [ ] EMA 模型
- [ ] 更多数据增强（颜色抖动等）
- [ ] 自动超参数搜索
- [ ] 训练可视化（实时图像对比）

---

## 技术亮点

1. **模块化设计**：每个组件独立，易于扩展和维护
2. **灵活配置**：YAML 配置文件，无需修改代码
3. **兼容性**：保存的模型与 `demo.py`/`test.py` 完全兼容
4. **鲁棒性**：完善的错误检查和异常处理
5. **易用性**：提供多个配置模板和详细文档
6. **可监控性**：TensorBoard + 日志文件

---

## 贡献者指南

如需添加新功能：

1. **新损失函数**：在 `losses.py` 中添加 `@register('YourLoss')` 类
2. **新调度器**：在 `schedulers.py` 的 `make_scheduler()` 中添加分支
3. **新评估指标**：在 `evaluate.py` 中添加计算函数
4. **新配置**：在 `configs/train/` 中添加 YAML 文件

---

## 支持与反馈

如遇问题：
1. 查看 `TRAINING.md` 的"常见问题"部分
2. 运行 `python check_training_env.py` 检查环境
3. 检查 `./save/<experiment_name>/log.txt` 日志
4. 查看 TensorBoard 的训练曲线

祝训练顺利！🚀
