# Linux 系统训练 ContinuousSR 快速指南

本指南专门针对 Linux 系统用户，提供最简化的训练流程。

## 系统要求

- **操作系统**: Ubuntu 18.04+ / CentOS 7+ / 其他主流 Linux 发行版
- **Python**: 3.9+
- **CUDA**: 11.0+ (推荐)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (推荐)
- **存储**: 至少 10GB 可用空间

## 一、环境准备

### 1.1 检查 Python 环境

```bash
python3 --version  # 应该是 3.9 或更高版本
```

如果没有安装 Python 3.9+：
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.9 python3.9-pip python3.9-venv

# CentOS/RHEL
sudo yum install python39 python39-pip
```

### 1.2 检查 CUDA 和 GPU

```bash
nvidia-smi  # 查看 GPU 信息和 CUDA 版本
```

如果没有显示 GPU 信息，请先安装 NVIDIA 驱动和 CUDA。

### 1.3 安装依赖

```bash
# 进入项目目录
cd /path/to/ContinuousSR-main

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 安装基础依赖
pip install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt

# 安装 gsplat（关键依赖）
cd gsplat
pip install -e .
cd ..
```

## 二、数据准备

### 2.1 下载数据集

```bash
# 创建数据目录
mkdir -p ~/datasets

# 下载 DIV2K（训练集）
cd ~/datasets
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip
mv DIV2K_train_HR DIV2K
mkdir -p DIV2K/train
mv DIV2K DIV2K/train/HR

# 下载 Set5（验证集）
# 请从 https://github.com/xinntao/BasicSR/blob/master/docs/DatasetPreparation.md
# 下载并解压到 ~/datasets/Set5/HR/
```

### 2.2 验证数据结构

```bash
tree -L 3 ~/datasets
# 应该看到：
# ~/datasets/
# ├── DIV2K/
# │   └── train/
# │       └── HR/
# └── Set5/
#     └── HR/
```

### 2.3 修改配置文件

```bash
cd /path/to/ContinuousSR-main

# 使用你喜欢的编辑器修改配置
nano configs/train/train-div2k.yaml
# 或
vim configs/train/train-div2k.yaml
```

修改以下路径：
```yaml
train_dataset:
  dataset:
    args:
      root_path: /home/your_username/datasets/DIV2K/train/HR  # 修改为你的路径

val_dataset:
  dataset:
    args:
      root_path: /home/your_username/datasets/Set5/HR  # 修改为你的路径
```

## 三、环境检查

```bash
# 确保在项目目录和虚拟环境中
cd /path/to/ContinuousSR-main
source venv/bin/activate  # 如果使用了虚拟环境

# 运行环境检查
python check_training_env.py --config configs/train/train-div2k.yaml
```

如果所有检查通过（显示绿色 ✓），继续下一步。

## 四、开始训练

### 4.1 使用训练脚本（推荐）

```bash
# 赋予执行权限（首次使用）
chmod +x train.sh

# 查看脚本帮助
./train.sh -h

# 基础训练
./train.sh -c configs/train/train-div2k.yaml -g 0

# 或者快速测试（使用小数据集）
./train.sh -q
```

### 4.2 使用 Python 命令

```bash
# 基础训练
python train.py --config configs/train/train-div2k.yaml --gpu 0

# 指定实验名称
python train.py --config configs/train/train-div2k.yaml --gpu 0 --name my_experiment

# 快速测试
python train.py --config configs/train/train-quick.yaml --gpu 0
```

### 4.3 后台运行（长时间训练）

```bash
# 使用 nohup 后台运行
nohup python train.py --config configs/train/train-div2k.yaml --gpu 0 > training.log 2>&1 &

# 查看日志
tail -f training.log

# 或使用 screen/tmux
screen -S training
python train.py --config configs/train/train-div2k.yaml --gpu 0
# 按 Ctrl+A 然后 D 分离会话
# 重新连接: screen -r training
```

### 4.4 多 GPU 训练

```bash
# 指定使用 GPU 0 和 1
CUDA_VISIBLE_DEVICES=0,1 python train.py --config configs/train/train-div2k.yaml
```

## 五、监控训练

### 5.1 查看日志

```bash
# 实时查看日志
tail -f ./save/train-div2k/log.txt

# 查看最后 50 行
tail -n 50 ./save/train-div2k/log.txt

# 搜索特定内容
grep "PSNR" ./save/train-div2k/log.txt
```

### 5.2 TensorBoard 可视化

```bash
# 启动 TensorBoard
tensorboard --logdir ./save/train-div2k/tensorboard --port 6006 --bind_all

# 如果是远程服务器，使用 SSH 端口转发
# 在本地机器运行:
ssh -L 6006:localhost:6006 user@remote-server
# 然后在本地浏览器访问: http://localhost:6006
```

### 5.3 监控 GPU 使用

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 或使用更友好的工具
pip install gpustat
watch -n 1 gpustat -cpu
```

## 六、恢复训练

如果训练中断，可以从 checkpoint 恢复：

```bash
# 使用脚本
./train.sh -r save/train-div2k/checkpoint_latest.pth

# 或使用 Python 命令
python train.py --config configs/train/train-div2k.yaml --gpu 0 \
    --resume ./save/train-div2k/checkpoint_latest.pth
```

## 七、测试模型

```bash
# 在 Set5 上测试
python test.py --config configs/test/test-set5-4.yaml \
    --model ./save/train-div2k/checkpoint_best.pth

# 推理单张图像
python demo.py --input butterflyx4.png \
    --model ./save/train-div2k/checkpoint_best.pth \
    --scale 4,4 --output output.png
```

## 常见问题 (Linux 特定)

### Q1: Permission denied when running train.sh

```bash
chmod +x train.sh
./train.sh
```

### Q2: CUDA out of memory

修改配置文件，降低 batch_size：
```yaml
batch_size: 8  # 原来是 16，改为 8 或 4
```

### Q3: 数据加载太慢

- 使用 SSD 存储数据
- 增加 num_workers (但不要超过 CPU 核心数)
- 使用 `cache: in_memory` 将数据加载到内存

### Q4: ModuleNotFoundError: No module named 'gsplat'

```bash
cd gsplat
pip install -e .
cd ..
```

### Q5: 远程服务器训练，SSH 断开后进程终止

使用 screen 或 tmux：
```bash
# 使用 screen
screen -S training
python train.py --config configs/train/train-div2k.yaml --gpu 0
# Ctrl+A D (分离)
# screen -r training (重新连接)

# 或使用 tmux
tmux new -s training
python train.py --config configs/train/train-div2k.yaml --gpu 0
# Ctrl+B D (分离)
# tmux attach -t training (重新连接)
```

## 性能优化建议（Linux）

### 1. 使用更快的数据加载

```bash
# 安装 pillow-simd (比标准 Pillow 快 4-6 倍)
pip uninstall pillow
pip install pillow-simd
```

### 2. 使用 PyTorch 最新版本

```bash
# 使用最新的稳定版本可能有性能提升
pip install torch torchvision --upgrade
```

### 3. 设置环境变量优化

在训练前设置：
```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0
```

或添加到 `~/.bashrc`：
```bash
echo 'export OMP_NUM_THREADS=8' >> ~/.bashrc
echo 'export MKL_NUM_THREADS=8' >> ~/.bashrc
source ~/.bashrc
```

## 自动化训练脚本示例

创建 `run_training.sh`：
```bash
#!/bin/bash

# 激活虚拟环境
source /path/to/ContinuousSR-main/venv/bin/activate

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# 进入项目目录
cd /path/to/ContinuousSR-main

# 运行训练
python train.py \
    --config configs/train/train-div2k.yaml \
    --gpu 0 \
    --name experiment_$(date +%Y%m%d_%H%M%S)

# 训练完成后发送通知（可选）
# 需要安装: pip install pushbullet.py
# python -c "from pushbullet import Pushbullet; pb = Pushbullet('YOUR_API_KEY'); pb.push_note('Training', 'Training completed!')"
```

使用：
```bash
chmod +x run_training.sh
nohup ./run_training.sh > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## 更多资源

- 详细训练指南: [TRAINING.md](TRAINING.md)
- 快速开始: [QUICKSTART.md](QUICKSTART.md)
- 文件说明: [TRAINING_FILES.md](TRAINING_FILES.md)

祝训练顺利！🚀
