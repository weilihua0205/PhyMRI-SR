# 快速开始训练 ContinuousSR

## 步骤 1：准备数据集

下载并解压数据集：
- **DIV2K**: https://data.vision.ee.ethz.ch/cvl/DIV2K/
- **Set5/Set14**: https://github.com/xinntao/BasicSR/blob/master/docs/DatasetPreparation.md

目录结构：
```
/your/path/
├── DIV2K/train/HR/
├── Set5/HR/
└── Set14/HR/
```

## 步骤 2：修改配置文件

编辑 `configs/train/train-div2k.yaml`：
```yaml
train_dataset:
  dataset:
    args:
      root_path: /your/path/DIV2K/train/HR  # 修改这里

val_dataset:
  dataset:
    args:
      root_path: /your/path/Set5/HR  # 修改这里
```

## 步骤 3：检查环境

```bash
python check_training_env.py --config configs/train/train-div2k.yaml
```

如果所有检查通过，继续下一步。

## 步骤 4：开始训练

### 方式 1：直接使用 Python 命令（跨平台）

```bash
# 基础训练
python train.py --config configs/train/train-div2k.yaml --gpu 0

# 或快速测试（小数据集）
python train.py --config configs/train/train-quick.yaml --gpu 0
```

### 方式 2：使用启动脚本

**Linux/Mac:**
```bash
# 赋予执行权限（首次使用）
chmod +x train.sh

# 基础训练
./train.sh -c configs/train/train-div2k.yaml -g 0

# 快速训练
./train.sh -q

# 环境检查
./train.sh -k

# 从 checkpoint 恢复
./train.sh -r save/train-div2k/checkpoint_latest.pth
```

**Windows PowerShell:**
```powershell
# 基础训练
.\train.ps1 -Config configs\train\train-div2k.yaml -GPU 0

# 快速训练
.\train.ps1 -Quick

# 环境检查
.\train.ps1 -Check
```

## 步骤 5：监控训练

**查看日志：**
```bash
tail -f ./save/train-div2k/log.txt
```

**TensorBoard：**
```bash
tensorboard --logdir ./save/train-div2k/tensorboard --port 6006
```

## 步骤 6：测试模型

```bash
# 测试最佳模型
python test.py --config configs/test/test-set5-4.yaml \
    --model ./save/train-div2k/checkpoint_best.pth

# 推理单张图像
python demo.py --input butterflyx4.png \
    --model ./save/train-div2k/checkpoint_best.pth \
    --scale 4,4 --output output.png
```

## 常见问题

**Q: CUDA Out of Memory?**
- 减小 batch_size（例如从 16 改为 8 或 4）
- 减小 inp_size（例如从 48 改为 32）

**Q: 训练太慢?**
- 增加 num_workers（数据加载并行度）
- 使用 `cache: in_memory` 将数据加载到内存

**Q: 找不到 gsplat?**
```bash
cd gsplat
pip install -e .
```

详细文档请查看 [TRAINING.md](TRAINING.md)
