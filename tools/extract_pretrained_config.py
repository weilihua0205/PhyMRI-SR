"""
深度分析预训练模型的完整配置
提取用于训练的精确参数
"""

import torch
import yaml
import json
from collections import OrderedDict

def extract_training_config(checkpoint_path):
    """从checkpoint中提取完整的训练配置"""
    print("="*80)
    print("提取预训练模型的训练配置")
    print("="*80)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 1. 提取模型配置
    print("\n[1] 模型架构配置")
    print("-"*80)
    model_spec = checkpoint['model']
    
    # 打印完整的模型配置
    config_dict = {
        'model': {
            'name': model_spec['name'],
            'args': model_spec['args']
        }
    }
    
    print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))
    
    # 2. 详细解析encoder配置
    print("\n[2] Encoder详细配置 (HAT)")
    print("-"*80)
    encoder_spec = model_spec['args']['encoder_spec']
    print(f"  名称: {encoder_spec['name']}")
    print(f"  参数:")
    for key, value in encoder_spec['args'].items():
        print(f"    {key}: {value}")
    
    # 3. 解析MLP配置
    print("\n[3] MLP配置")
    print("-"*80)
    
    # Color MLP (fc_spec)
    print("  Color MLP (用于Gaussian颜色):")
    fc_spec = model_spec['args']['fc_spec']
    print(f"    名称: {fc_spec['name']}")
    print(f"    参数: {fc_spec['args']}")
    
    # CNN spec
    print("\n  CNN配置:")
    cnn_spec = model_spec['args']['cnn_spec']
    print(f"    名称: {cnn_spec['name']}")
    print(f"    参数: {cnn_spec['args']}")
    
    # 4. 从state_dict反向推断其他MLP配置
    print("\n[4] 从权重反向推断MLP配置")
    print("-"*80)
    sd = model_spec['sd']
    
    # 分析color MLP结构
    print("  Color MLP (mlp) 层结构:")
    mlp_layers = [name for name in sd.keys() if name.startswith('mlp.layers.') and 'weight' in name]
    mlp_layers.sort(key=lambda x: int(x.split('.')[2]))
    
    mlp_dims = []
    for layer in mlp_layers:
        weight = sd[layer]
        out_dim, in_dim = weight.shape
        mlp_dims.append((in_dim, out_dim))
        print(f"    {layer}: {in_dim} → {out_dim}")
    
    print(f"\n  推断配置: in_dim={mlp_dims[0][0]}, out_dim={mlp_dims[-1][1]}, hidden_list={[d[1] for d in mlp_dims[:-1]]}")
    
    # 分析offset MLP结构
    print("\n  Offset MLP (mlp_offset) 层结构:")
    offset_layers = [name for name in sd.keys() if name.startswith('mlp_offset.layers.') and 'weight' in name]
    offset_layers.sort(key=lambda x: int(x.split('.')[2]))
    
    offset_dims = []
    for layer in offset_layers:
        weight = sd[layer]
        out_dim, in_dim = weight.shape
        offset_dims.append((in_dim, out_dim))
        print(f"    {layer}: {in_dim} → {out_dim}")
    
    print(f"\n  推断配置: in_dim={offset_dims[0][0]}, out_dim={offset_dims[-1][1]}, hidden_list={[d[1] for d in offset_dims[:-1]]}")
    
    # 分析vector MLP结构
    print("\n  Vector MLP (mlp_vector) 层结构:")
    vector_layers = [name for name in sd.keys() if name.startswith('mlp_vector.layers.') and 'weight' in name]
    vector_layers.sort(key=lambda x: int(x.split('.')[2]))
    
    vector_dims = []
    for layer in vector_layers:
        weight = sd[layer]
        out_dim, in_dim = weight.shape
        vector_dims.append((in_dim, out_dim))
        print(f"    {layer}: {in_dim} → {out_dim}")
    
    print(f"\n  推断配置: in_dim={vector_dims[0][0]}, out_dim={vector_dims[-1][1]}, hidden_list={[d[1] for d in vector_dims[:-1]]}")
    
    # 5. 检查激活函数
    print("\n[5] 激活函数分析")
    print("-"*80)
    
    # 检查是否有BatchNorm或其他归一化层
    norm_layers = [name for name in sd.keys() if any(x in name.lower() for x in ['norm', 'bn', 'ln'])]
    if norm_layers:
        print(f"  找到归一化层: {len(norm_layers)}个")
        for name in norm_layers[:5]:
            print(f"    {name}")
    
    # 模拟推断激活函数（通过测试输出范围）
    print("\n  通过forward pass推断激活函数:")
    try:
        import models
        model = models.make(model_spec, load_sd=True)
        model.eval()
        
        with torch.no_grad():
            # 测试不同输入范围
            test_inputs = [
                torch.randn(1, 3, 12, 12) * 0.1,  # 小值
                torch.randn(1, 3, 12, 12),        # 标准正态
                torch.randn(1, 3, 12, 12) * 2,    # 大值
            ]
            
            for i, inp in enumerate(test_inputs):
                out = model(inp, torch.tensor([4.0]))
                print(f"    测试{i+1}: input range=[{inp.min():.3f}, {inp.max():.3f}] → output range=[{out.min():.3f}, {out.max():.3f}]")
            
            # 分析输出特性
            if all(out.min() >= -0.1 and out.max() <= 1.1 for out in [model(inp, torch.tensor([4.0])) for inp in test_inputs]):
                print("  ✓ 输出被约束在[0,1]附近，可能有Sigmoid/Clamp")
            else:
                print("  ✗ 输出未被约束，没有输出激活函数")
                
    except Exception as e:
        print(f"  无法进行测试: {e}")
    
    # 6. 生成完整配置文件
    print("\n[6] 生成训练配置文件")
    print("-"*80)
    
    # 推断的完整配置
    inferred_config = {
        'model': {
            'name': 'continuous-gaussian',
            'args': {
                'encoder_spec': encoder_spec,
                'cnn_spec': cnn_spec,
                'fc_spec': fc_spec,
                # 注意：这些可能需要根据代码推断
                'BLOCK_H': 16,
                'BLOCK_W': 16,
            }
        }
    }
    
    # 保存配置
    with open('config_pretrained_model.yaml', 'w') as f:
        yaml.dump(inferred_config, f, default_flow_style=False, sort_keys=False)
    
    print("  已保存到: config_pretrained_model.yaml")
    
    # 7. 检查优化器配置
    print("\n[7] 优化器配置")
    print("-"*80)
    if 'optimizer' in checkpoint:
        opt_state = checkpoint['optimizer']
        print(f"  State keys: {opt_state.keys()}")
        if 'param_groups' in opt_state:
            for i, pg in enumerate(opt_state['param_groups']):
                print(f"\n  参数组 {i}:")
                for key, value in pg.items():
                    if key != 'params':
                        print(f"    {key}: {value}")
    
    return inferred_config


def compare_with_current_config(pretrained_config, current_config_path):
    """对比预训练配置和当前配置的差异"""
    print("\n" + "="*80)
    print("配置差异对比")
    print("="*80)
    
    with open(current_config_path, 'r') as f:
        current_config = yaml.safe_load(f)
    
    print("\n[Encoder对比]")
    print(f"  预训练: {pretrained_config['model']['args']['encoder_spec']['name']}")
    print(f"  当前:   {current_config['model']['args']['encoder_spec']['name']}")
    
    if pretrained_config['model']['args']['encoder_spec']['name'] != current_config['model']['args']['encoder_spec']['name']:
        print("  ⚠️ Encoder不同！这会导致性能差异")
        print("\n  预训练Encoder参数:")
        for k, v in pretrained_config['model']['args']['encoder_spec']['args'].items():
            print(f"    {k}: {v}")
        print("\n  当前Encoder参数:")
        for k, v in current_config['model']['args']['encoder_spec']['args'].items():
            print(f"    {k}: {v}")


def analyze_gaussian_mechanism(checkpoint_path):
    """分析Gaussian相关机制"""
    print("\n" + "="*80)
    print("Gaussian Splatting机制分析")
    print("="*80)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    sd = checkpoint['model']['sd']
    
    print("\n[1] Adaptive Position Drifting (Offset MLP)")
    print("-"*80)
    print("  功能: 预测每个Gaussian的位置偏移")
    print("  输入: 特征向量 (256维)")
    print("  输出: 2D偏移量 (x, y)")
    
    offset_layers = [name for name in sd.keys() if 'mlp_offset' in name and 'weight' in name]
    print(f"\n  层数: {len(offset_layers)}")
    print("  网络结构:")
    for layer in sorted(offset_layers, key=lambda x: int(x.split('.')[2])):
        weight = sd[layer]
        print(f"    {layer}: shape={tuple(weight.shape)}")
    
    print("\n  输出层分析:")
    last_weight = sd['mlp_offset.layers.10.weight']
    last_bias = sd['mlp_offset.layers.10.bias']
    print(f"    最后一层权重: shape={tuple(last_weight.shape)}, range=[{last_weight.min():.4f}, {last_weight.max():.4f}]")
    print(f"    最后一层偏置: shape={tuple(last_bias.shape)}, value={last_bias.numpy()}")
    print(f"  注意: 输出维度为2，对应(delta_x, delta_y)")
    
    print("\n[2] Color Gaussian Mapping (Color MLP)")
    print("-"*80)
    print("  功能: 为每个Gaussian预测RGB颜色")
    print("  输入: 特征向量 (256维)")
    print("  输出: RGB颜色 (3维)")
    
    color_layers = [name for name in sd.keys() if name.startswith('mlp.layers.') and 'weight' in name]
    print(f"\n  层数: {len(color_layers)}")
    print("  网络结构:")
    for layer in sorted(color_layers, key=lambda x: int(x.split('.')[2])):
        weight = sd[layer]
        print(f"    {layer}: shape={tuple(weight.shape)}")
    
    print("\n  输出层分析:")
    last_weight = sd['mlp.layers.10.weight']
    last_bias = sd['mlp.layers.10.bias']
    print(f"    最后一层权重: shape={tuple(last_weight.shape)}, range=[{last_weight.min():.4f}, {last_weight.max():.4f}]")
    print(f"    最后一层偏置: shape={tuple(last_bias.shape)}, value={last_bias.numpy()}")
    print(f"  注意: 输出维度为3，对应RGB")
    
    # 检查输出范围特性
    print("\n[3] 输出约束机制")
    print("-"*80)
    
    # 分析偏置值
    offset_bias = sd['mlp_offset.layers.10.bias'].numpy()
    color_bias = sd['mlp.layers.10.bias'].numpy()
    
    print(f"  Offset偏置: {offset_bias} (均值={offset_bias.mean():.6f})")
    print(f"  Color偏置:  {color_bias} (均值={color_bias.mean():.6f})")
    
    if abs(color_bias.mean()) < 0.01:
        print("  → Color偏置接近0，可能依赖激活函数约束到[0,1]")
    elif color_bias.mean() > 0.4:
        print("  → Color偏置较大，可能无Sigmoid（直接输出）")
    
    print("\n  推荐配置:")
    print("    - 如果预训练模型工作正常，说明训练时有以下可能:")
    print("      1. 使用了Sigmoid但保存时未包含在state_dict")
    print("      2. 通过损失函数和长时间训练让模型学会输出[0,1]范围")
    print("      3. 在inference时外部添加了Sigmoid/Clamp")


if __name__ == '__main__':
    checkpoint_path = 'ContinuousSR.pth'
    current_config_path = 'configs/train/train-div2k.yaml'
    
    # 1. 提取配置
    pretrained_config = extract_training_config(checkpoint_path)
    
    # 2. 对比配置
    compare_with_current_config(pretrained_config, current_config_path)
    
    # 3. 分析Gaussian机制
    analyze_gaussian_mechanism(checkpoint_path)
    
    print("\n" + "="*80)
    print("分析完成")
    print("="*80)
    print("\n下一步:")
    print("  1. 检查 config_pretrained_model.yaml")
    print("  2. 根据差异修改你的训练配置")
    print("  3. 特别注意: HAT encoder vs EDSR encoder")
    print("  4. 确认MLP配置是否一致")
