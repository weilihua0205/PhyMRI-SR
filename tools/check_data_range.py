"""
快速验证数据范围问题
检查训练时输入数据的实际范围
"""
import torch
import yaml
from torch.utils.data import DataLoader
import datasets

# 加载配置
config_path = 'save/div2k_sample_swinir_1000epochs_debug6/config.yaml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# 构建数据集
train_spec = config['train_dataset']
train_dataset = datasets.make(train_spec['dataset'])
train_dataset = datasets.make(train_spec['wrapper'], args={'dataset': train_dataset})
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# 获取一个 batch
batch = next(iter(train_loader))

# 检查归一化参数
data_norm = config.get('data_norm', None)
if data_norm:
    inp_sub = torch.FloatTensor(data_norm['inp']['sub']).view(1, -1, 1, 1)
    inp_div = torch.FloatTensor(data_norm['inp']['div']).view(1, -1, 1, 1)
    gt_sub = torch.FloatTensor(data_norm['gt']['sub']).view(1, -1, 1, 1)
    gt_div = torch.FloatTensor(data_norm['gt']['div']).view(1, -1, 1, 1)
    
    print("=" * 60)
    print("数据范围检查")
    print("=" * 60)
    print(f"\n归一化参数:")
    print(f"  inp_sub: {inp_sub.squeeze().tolist()}")
    print(f"  inp_div: {inp_div.squeeze().tolist()}")
    print(f"  gt_sub: {gt_sub.squeeze().tolist()}")
    print(f"  gt_div: {gt_div.squeeze().tolist()}")
    
    print(f"\n原始数据范围（来自 DataLoader）:")
    print(f"  batch['inp']: min={batch['inp'].min():.6f}, max={batch['inp'].max():.6f}, mean={batch['inp'].mean():.6f}")
    print(f"  batch['gt']:  min={batch['gt'].min():.6f}, max={batch['gt'].max():.6f}, mean={batch['gt'].mean():.6f}")
    
    # 应用归一化
    inp = (batch['inp'] - inp_sub) / inp_div
    gt = (batch['gt'] - gt_sub) / gt_div
    
    print(f"\n归一化后的数据范围（训练时实际输入）:")
    print(f"  inp: min={inp.min():.6f}, max={inp.max():.6f}, mean={inp.mean():.6f}")
    print(f"  gt:  min={gt.min():.6f}, max={gt.max():.6f}, mean={gt.mean():.6f}")
    
    print("\n" + "=" * 60)
    print("问题诊断:")
    print("=" * 60)
    if inp.max() < 0.01:
        print("❌ 错误：输入被过度归一化！")
        print("   数据已经在 [0,1] 范围（ToTensor 做的），不应该再除以 255")
        print("   当前输入范围约为 [0, 0.004]，模型无法正常学习")
        print("\n修复方案：")
        print("   在 config.yaml 中移除 data_norm 配置，或将 div 改为 [1]")
    else:
        print("✓ 数据范围正常")
else:
    print("=" * 60)
    print("未使用 data_norm，数据范围:")
    print(f"  batch['inp']: min={batch['inp'].min():.6f}, max={batch['inp'].max():.6f}")
    print(f"  batch['gt']:  min={batch['gt'].min():.6f}, max={batch['gt'].max():.6f}")
    print("=" * 60)
