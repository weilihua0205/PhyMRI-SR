"""
分析预训练模型 ContinuousSR.pth 的结构和参数
"""

import torch
import numpy as np

def analyze_checkpoint(checkpoint_path):
    """详细分析checkpoint内容"""
    print("="*80)
    print(f"分析模型: {checkpoint_path}")
    print("="*80)
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n[1] Checkpoint 顶层结构")
    print("-"*80)
    for key in checkpoint.keys():
        value = checkpoint[key]
        if isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} keys")
        elif isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor {value.shape}")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")
    
    # 分析模型结构
    if 'model' in checkpoint:
        print("\n[2] Model Spec (模型配置)")
        print("-"*80)
        model_spec = checkpoint['model']
        for key, value in model_spec.items():
            if key != 'sd':
                print(f"  {key}: {value}")
    
    # 分析state_dict
    if 'model' in checkpoint and 'sd' in checkpoint['model']:
        sd = checkpoint['model']['sd']
        print("\n[3] State Dict 参数统计")
        print("-"*80)
        
        total_params = 0
        param_groups = {}
        
        for name, param in sd.items():
            total_params += param.numel()
            
            # 按模块分组
            module_name = name.split('.')[0]
            if module_name not in param_groups:
                param_groups[module_name] = {'count': 0, 'params': 0, 'layers': []}
            param_groups[module_name]['count'] += 1
            param_groups[module_name]['params'] += param.numel()
            param_groups[module_name]['layers'].append(name)
        
        print(f"  总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  总层数: {len(sd)}")
        
        print("\n  各模块参数分布:")
        for module, info in sorted(param_groups.items()):
            print(f"    {module}:")
            print(f"      - 层数: {info['count']}")
            print(f"      - 参数量: {info['params']:,} ({info['params']/1e6:.2f}M, {100*info['params']/total_params:.1f}%)")
        
        # 详细查看关键层
        print("\n[4] 关键层详细信息")
        print("-"*80)
        
        keywords = ['mlp', 'color', 'offset', 'conv', 'encoder']
        for keyword in keywords:
            matching_layers = [name for name in sd.keys() if keyword in name.lower()]
            if matching_layers:
                print(f"\n  包含 '{keyword}' 的层 ({len(matching_layers)}个):")
                for name in matching_layers[:10]:  # 只显示前10个
                    param = sd[name]
                    print(f"    {name}:")
                    print(f"      shape: {tuple(param.shape)}")
                    if param.dtype in [torch.float32, torch.float64, torch.float16]:
                        print(f"      range: [{param.min().item():.6f}, {param.max().item():.6f}]")
                        print(f"      mean: {param.mean().item():.6f}, std: {param.std().item():.6f}")
                    else:
                        print(f"      dtype: {param.dtype} (non-float, skipping stats)")
                if len(matching_layers) > 10:
                    print(f"    ... 还有 {len(matching_layers)-10} 层")
        
        # 分析MLP输出层
        print("\n[5] MLP输出层分析（关键！）")
        print("-"*80)
        mlp_output_layers = [name for name in sd.keys() if 'mlp' in name and ('weight' in name or 'bias' in name)]
        mlp_output_layers = [name for name in mlp_output_layers if 'layers' in name]
        
        # 找到最后一层
        color_mlp_last = None
        for name in mlp_output_layers:
            if 'mlp.layers' in name:
                param = sd[name]
                print(f"  {name}:")
                print(f"    shape: {tuple(param.shape)}")
                print(f"    range: [{param.min().item():.6f}, {param.max().item():.6f}]")
                print(f"    mean: {param.mean().item():.6f}, std: {param.std().item():.6f}")
                
                # 检查是否是输出层（输出维度为3）
                if 'weight' in name and param.shape[0] == 3:
                    color_mlp_last = name
                    print(f"    ★ 这是color MLP的输出层（输出3通道）")
        
        # 检查是否有激活函数的参数
        print("\n[6] 激活函数检查")
        print("-"*80)
        activation_layers = [name for name in sd.keys() if any(act in name.lower() for act in ['sigmoid', 'tanh', 'relu', 'activation'])]
        if activation_layers:
            print(f"  找到激活函数相关层: {len(activation_layers)}个")
            for name in activation_layers:
                print(f"    {name}")
        else:
            print("  ⚠️ 未找到显式的激活函数参数")
            print("  注意: 如果color MLP输出层没有Sigmoid激活，输出可能不在[0,1]范围")
        
        # 分析Gaussian相关参数
        print("\n[7] Gaussian相关参数")
        print("-"*80)
        gaussian_keys = [name for name in sd.keys() if any(k in name.lower() for k in ['gaussian', 'cov', 'offset', 'opacity'])]
        if gaussian_keys:
            print(f"  找到Gaussian相关层: {len(gaussian_keys)}个")
            for name in gaussian_keys[:20]:
                param = sd[name]
                print(f"    {name}: {tuple(param.shape)}, range=[{param.min().item():.4f}, {param.max().item():.4f}]")
        
        # 测试模型输出范围
        print("\n[8] 模拟前向传播测试")
        print("-"*80)
        print("  使用随机输入测试模型输出范围...")
        
        try:
            import models
            model_spec = checkpoint['model']
            model = models.make(model_spec, load_sd=True)
            model.eval()
            
            # 创建随机输入
            dummy_input = torch.randn(1, 3, 12, 12)
            dummy_scale = torch.tensor([4.0])
            
            with torch.no_grad():
                output = model(dummy_input, dummy_scale)
            
            print(f"  输入: shape={tuple(dummy_input.shape)}, range=[{dummy_input.min():.4f}, {dummy_input.max():.4f}]")
            print(f"  输出: shape={tuple(output.shape)}, range=[{output.min():.4f}, {output.max():.4f}]")
            print(f"  输出统计: mean={output.mean():.6f}, std={output.std():.6f}")
            
            if output.max() < 0.1:
                print("  ⚠️ 警告: 输出值非常小（<0.1），可能缺少输出激活函数")
            elif output.min() < 0 or output.max() > 1:
                print("  ⚠️ 警告: 输出超出[0,1]范围，可能缺少Sigmoid/Tanh激活")
            else:
                print("  ✓ 输出范围正常")
                
        except Exception as e:
            print(f"  ✗ 无法加载模型进行测试: {e}")
    
    # 分析训练状态
    if 'epoch' in checkpoint:
        print("\n[9] 训练状态")
        print("-"*80)
        print(f"  训练轮数: {checkpoint['epoch']}")
        if 'best_metric' in checkpoint:
            print(f"  最佳指标: {checkpoint['best_metric']}")
    
    print("\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == '__main__':
    checkpoint_path = 'ContinuousSR.pth'
    analyze_checkpoint(checkpoint_path)
