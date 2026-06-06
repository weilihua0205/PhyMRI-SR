"""
诊断训练流程 - 检查数据、模型、可视化各环节
"""

import os
import torch
import yaml
import argparse
from torch.utils.data import DataLoader

import datasets
import models
import utils


def diagnose_data(config):
    """诊断数据加载"""
    print("\n" + "="*60)
    print("1. 数据加载诊断")
    print("="*60)
    
    # 加载训练数据集
    train_spec = config['train_dataset']
    train_dataset = datasets.make(train_spec['dataset'])
    train_dataset = datasets.make(train_spec['wrapper'], args={'dataset': train_dataset})
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.get('batch_size', 1),
        shuffle=False,
        num_workers=0,  # 单线程便于调试
        pin_memory=True
    )
    
    # 获取一个batch
    batch = next(iter(train_loader))
    
    print(f"\n[训练数据样本]")
    print(f"  Batch keys: {batch.keys()}")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}:")
            print(f"    shape: {val.shape}")
            print(f"    dtype: {val.dtype}")
            print(f"    range: [{val.min().item():.4f}, {val.max().item():.4f}]")
            print(f"    mean: {val.mean().item():.4f}, std: {val.std().item():.4f}")
        else:
            print(f"  {key}: {val}")
    
    # 检查验证数据
    val_spec = config['val_dataset']
    val_dataset = datasets.make(val_spec['dataset'])
    val_dataset = datasets.make(val_spec['wrapper'], args={'dataset': val_dataset})
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_spec.get('batch_size', 1),
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    batch = next(iter(val_loader))
    print(f"\n[验证数据样本]")
    print(f"  Batch keys: {batch.keys()}")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}:")
            print(f"    shape: {val.shape}")
            print(f"    range: [{val.min().item():.4f}, {val.max().item():.4f}]")
            print(f"    mean: {val.mean().item():.4f}")
    
    return train_loader, val_loader


def diagnose_model(config, train_loader, checkpoint_path=None):
    """诊断模型前向传播"""
    print("\n" + "="*60)
    print("2. 模型前向传播诊断")
    print("="*60)
    
    # 构建模型
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"==> Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_spec = checkpoint['model']
        model = models.make(model_spec, load_sd=True).cuda()
        print(f"==> Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print(f"==> Building model from scratch (untrained)")
        model = models.make(config['model']).cuda()
    
    model.eval()
    
    # 获取一个batch
    batch = next(iter(train_loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()
    
    # 数据归一化
    data_norm = config.get('data_norm', None)
    if data_norm:
        print(f"\n[数据归一化]")
        print(f"  inp: sub={data_norm['inp']['sub']}, div={data_norm['inp']['div']}")
        print(f"  gt: sub={data_norm['gt']['sub']}, div={data_norm['gt']['div']}")
        
        inp_sub = torch.FloatTensor(data_norm['inp']['sub']).view(1, -1, 1, 1).cuda()
        inp_div = torch.FloatTensor(data_norm['inp']['div']).view(1, -1, 1, 1).cuda()
        inp = (batch['inp'] - inp_sub) / inp_div
    else:
        print(f"\n[数据归一化] 未启用")
        inp = batch['inp']
    
    print(f"\n[模型输入]")
    print(f"  inp shape: {inp.shape}")
    print(f"  inp range: [{inp.min().item():.4f}, {inp.max().item():.4f}]")
    print(f"  inp mean: {inp.mean().item():.4f}, std: {inp.std().item():.4f}")
    print(f"  scale: {batch['scale']}")
    
    # 前向传播
    with torch.no_grad():
        pred = model(inp, batch['scale'])
    
    print(f"\n[模型输出（原始）]")
    print(f"  pred shape: {pred.shape}")
    print(f"  pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
    print(f"  pred mean: {pred.mean().item():.4f}, std: {pred.std().item():.4f}")
    
    # 检查是否有异常值
    if torch.isnan(pred).any():
        print(f"  ⚠️ WARNING: NaN detected in model output!")
    if torch.isinf(pred).any():
        print(f"  ⚠️ WARNING: Inf detected in model output!")
    
    # 反归一化
    if data_norm:
        gt_sub = torch.FloatTensor(data_norm['gt']['sub']).view(1, -1, 1, 1).cuda()
        gt_div = torch.FloatTensor(data_norm['gt']['div']).view(1, -1, 1, 1).cuda()
        pred = pred * gt_div + gt_sub
        
        print(f"\n[模型输出（反归一化后）]")
        print(f"  pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
        print(f"  pred mean: {pred.mean().item():.4f}")
    
    # Clamp到[0,1]
    pred_clamped = pred.clamp(0, 1)
    print(f"\n[模型输出（clamp后）]")
    print(f"  pred range: [{pred_clamped.min().item():.4f}, {pred_clamped.max().item():.4f}]")
    print(f"  pred mean: {pred_clamped.mean().item():.4f}")
    
    # 检查是否全为0或接近0
    if pred_clamped.max().item() < 0.01:
        print(f"  ⚠️ WARNING: Model output is nearly all zeros!")
    
    # 对比GT
    gt = batch['gt']
    print(f"\n[Ground Truth]")
    print(f"  gt shape: {gt.shape}")
    print(f"  gt range: [{gt.min().item():.4f}, {gt.max().item():.4f}]")
    print(f"  gt mean: {gt.mean().item():.4f}")
    
    return model, batch, pred_clamped


def diagnose_visualization(config, model, val_loader, checkpoint_name=""):
    """诊断可视化流程"""
    print("\n" + "="*60)
    print(f"3. 可视化流程诊断 {checkpoint_name}")
    print("="*60)
    
    model.eval()
    
    # 获取一个batch
    batch = next(iter(val_loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()
    
    # 数据归一化
    data_norm = config.get('data_norm', None)
    if data_norm:
        inp_sub = torch.FloatTensor(data_norm['inp']['sub']).view(1, -1, 1, 1).cuda()
        inp_div = torch.FloatTensor(data_norm['inp']['div']).view(1, -1, 1, 1).cuda()
        inp = (batch['inp'] - inp_sub) / inp_div
    else:
        inp = batch['inp']
    
    # 前向传播
    with torch.no_grad():
        pred = model(inp, batch['scale'])
    
    # 反归一化
    if data_norm:
        gt_sub = torch.FloatTensor(data_norm['gt']['sub']).view(1, -1, 1, 1).cuda()
        gt_div = torch.FloatTensor(data_norm['gt']['div']).view(1, -1, 1, 1).cuda()
        pred = pred * gt_div + gt_sub
    
    pred = pred.clamp(0, 1)
    
    print(f"\n[可视化数据准备]")
    print(f"  Input (batch['inp']): {batch['inp'].shape}, range=[{batch['inp'].min().item():.4f}, {batch['inp'].max().item():.4f}]")
    print(f"  GT: {batch['gt'].shape}, range=[{batch['gt'].min().item():.4f}, {batch['gt'].max().item():.4f}]")
    print(f"  Pred: {pred.shape}, range=[{pred.min().item():.4f}, {pred.max().item():.4f}]")
    
    # 上采样输入
    inp_upsampled = torch.nn.functional.interpolate(
        batch['inp'], 
        size=batch['gt'].shape[-2:], 
        mode='bicubic', 
        align_corners=False
    ).clamp(0, 1)
    
    print(f"  Input (upsampled): {inp_upsampled.shape}, range=[{inp_upsampled.min().item():.4f}, {inp_upsampled.max().item():.4f}]")
    
    # 检查通道数
    print(f"\n[通道处理]")
    print(f"  Input channels: {inp_upsampled.shape[1]}")
    print(f"  Pred channels: {pred.shape[1]}")
    print(f"  GT channels: {batch['gt'].shape[1]}")
    
    # 处理单通道
    if inp_upsampled.shape[1] == 1:
        inp_upsampled = inp_upsampled.repeat(1, 3, 1, 1)
        print(f"  Input converted to 3 channels: {inp_upsampled.shape}")
    
    if pred.shape[1] == 1:
        pred_vis = pred.repeat(1, 3, 1, 1)
        print(f"  Pred converted to 3 channels: {pred_vis.shape}")
    else:
        pred_vis = pred
    
    gt = batch['gt']
    if gt.shape[1] == 1:
        gt = gt.repeat(1, 3, 1, 1)
        print(f"  GT converted to 3 channels: {gt.shape}")
    
    # 拼接
    combined = torch.cat([inp_upsampled, pred_vis, gt], dim=3)
    print(f"\n[拼接结果]")
    print(f"  Combined shape: {combined.shape}")
    print(f"  Combined range: [{combined.min().item():.4f}, {combined.max().item():.4f}]")
    print(f"  Combined mean: {combined.mean().item():.4f}")
    
    # 保存测试图像
    from torchvision.utils import save_image
    import os
    save_path = f'diagnose_output{checkpoint_name}.png'
    save_image(combined, save_path, nrow=1, padding=2, normalize=False)
    print(f"\n==> Saved diagnostic image to {save_path}")
    print(f"    Please check if this image shows black squares!")
    
    # 返回统计信息
    return {
        'pred_mean': pred.mean().item(),
        'pred_max': pred.max().item(),
        'gt_mean': gt.mean().item()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train/train-div2k.yaml')
    parser.add_argument('--checkpoint', default='save/train-div2k/checkpoint_latest.pth',
                       help='Path to checkpoint file (optional)')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("ContinuousSR 训练流程诊断")
    print("="*60)
    print(f"配置文件: {args.config}")
    print(f"Checkpoint: {args.checkpoint if os.path.exists(args.checkpoint) else 'None (untrained)'}")
    
    # 1. 数据诊断
    train_loader, val_loader = diagnose_data(config)
    
    # 2. 模型诊断 - 未训练
    print("\n\n" + "#"*60)
    print("# 对比：未训练模型 vs 训练后模型")
    print("#"*60)
    
    model_untrained, batch, pred_untrained = diagnose_model(config, train_loader, checkpoint_path=None)
    stats_untrained = diagnose_visualization(config, model_untrained, val_loader, checkpoint_name="_untrained")
    
    # 3. 模型诊断 - 训练后
    if os.path.exists(args.checkpoint):
        model_trained, batch, pred_trained = diagnose_model(config, train_loader, checkpoint_path=args.checkpoint)
        stats_trained = diagnose_visualization(config, model_trained, val_loader, checkpoint_name="_trained")
    
    print("\n" + "="*60)
    print("诊断完成！")
    print("="*60)
    
    print("\n对比结果：")
    print(f"  未训练模型：pred_mean={stats_untrained['pred_mean']:.4f}, pred_max={stats_untrained['pred_max']:.4f}")
    if os.path.exists(args.checkpoint):
        print(f"  训练后模型：pred_mean={stats_trained['pred_mean']:.4f}, pred_max={stats_trained['pred_max']:.4f}")
        print(f"  Ground Truth：gt_mean={stats_trained['gt_mean']:.4f}")
    else:
        print(f"  Ground Truth：gt_mean={stats_untrained['gt_mean']:.4f}")
    
    print("\n检查要点：")
    print("  1. 数据范围是否在 [0, 1]")
    print("  2. 训练后模型输出应该明显大于未训练模型")
    print("  3. 如果训练后模型输出仍接近0，可能是：")
    print("     - 学习率过小")
    print("     - 训练轮数不足")
    print("     - 损失函数不合适")
    print("     - 模型结构有问题")
    print("  4. 检查生成的图像：")
    print("     - diagnose_output_untrained.png （应该很暗/黑色）")
    print("     - diagnose_output_trained.png （应该有可见内容）")


if __name__ == '__main__':
    main()
