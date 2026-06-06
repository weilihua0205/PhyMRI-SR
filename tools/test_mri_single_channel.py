#!/usr/bin/env python3
"""
单通道 MRI 训练快速测试脚本
用于验证单通道适配是否正确
"""

import torch
import numpy as np
import os

def test_single_channel_pipeline():
    """测试单通道 MRI 训练流水线"""
    
    print("=" * 60)
    print("单通道 MRI 训练流水线测试")
    print("=" * 60)
    
    # 测试 1: 数据加载
    print("\n[测试 1] 测试 npy_folder 数据集...")
    try:
        from datasets.npy_folder import NpyFolder
        
        # 创建测试数据
        test_dir = r"/home/ght/MRIxField/ContinuousSR-main_MRI/test_pipeline"
        os.makedirs(test_dir, exist_ok=True)
        
        # 保存单通道测试 npy
        test_img = np.random.rand(1, 256, 256).astype(np.float32)
        np.save(os.path.join(test_dir, "test_001.npy"), test_img)
        
        dataset = NpyFolder(root_path=test_dir, cache='in_memory')
        sample = dataset[0]
        
        assert sample.shape[0] == 1, f"期望通道数=1，实际={sample.shape[0]}"
        print(f"✅ 数据加载成功: shape={sample.shape}")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False
    
    # # 测试 2: resize_fn 单通道兼容性
    # print("\n[测试 2] 测试 resize_fn 单通道处理...")
    # try:
    #     from datasets.wrappers import resize_fn
        
    #     # 单通道输入
    #     single_ch = torch.randn(1, 256, 256)
    #     resized = resize_fn(single_ch, (128, 128))
        
    #     assert resized.shape[0] == 1, f"期望通道数=1，实际={resized.shape[0]}"
    #     assert resized.shape[1:] == (128, 128), f"期望尺寸=(128,128)，实际={resized.shape[1:]}"
    #     print(f"✅ resize_fn 单通道测试通过: {single_ch.shape} -> {resized.shape}")
        
    #     # 多通道输入（向后兼容测试）
    #     multi_ch = torch.randn(3, 256, 256)
    #     resized_rgb = resize_fn(multi_ch, (128, 128))
    #     assert resized_rgb.shape[0] == 3, "多通道应保持为 3"
    #     print(f"✅ resize_fn 多通道测试通过（向后兼容）")
        
    # except Exception as e:
    #     print(f"❌ resize_fn 测试失败: {e}")
    #     return False
    
    # 测试 3: 评估函数单通道兼容性
    print("\n[测试 3] 测试评估指标（PSNR/SSIM）...")
    try:
        from evaluate import calc_psnr, calc_ssim
        
        # 单通道数据
        sr_mri = torch.rand(1, 1, 256, 256).cuda()
        hr_mri = torch.rand(1, 1, 256, 256).cuda()
        
        psnr = calc_psnr(sr_mri, hr_mri, dataset='div2k')
        ssim = calc_ssim(sr_mri, hr_mri, dataset='div2k')
        
        print(f"✅ PSNR 计算成功: {psnr:.4f} dB")
        print(f"✅ SSIM 计算成功: {ssim:.4f}")
        
    except Exception as e:
        print(f"❌ 评估指标测试失败: {e}")
        return False
    
    # 测试 4: 模型构建
    print("\n[测试 4] 测试单通道模型构建...")
    try:
        import models
        
        model_spec = {
            'name': 'continuous-gaussian',
            'args': {
                'output_channels': 1,  # 单通道输出
                'encoder_spec': {
                    'name': 'swinir',
                    'args': {
                        'no_upsampling': True
                    }
                },
                'cnn_spec': {},
                'fc_spec': {},
                'BLOCK_H': 16,
                'BLOCK_W': 16
            }
        }
        
        model = models.make(model_spec).cuda()
        
        # 测试前向传播
        test_input = torch.randn(1, 1, 64, 64).cuda()  # [B, C=1, H, W]
        test_scale = torch.tensor([[1.0, 1.0]]).cuda()
        
        with torch.no_grad():
            output = model(test_input, test_scale)
        
        assert output.shape[1] == 1, f"期望输出通道数=1，实际={output.shape[1]}"
        print(f"✅ 模型构建成功")
        print(f"   输入: {test_input.shape}")
        print(f"   输出: {output.shape}")
        print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
    except Exception as e:
        print(f"❌ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试 5: 损失函数
    print("\n[测试 5] 测试损失函数...")
    try:
        from losses import make as make_loss
        
        criterion = make_loss({'name': 'L1Loss'})
        
        pred = torch.randn(2, 1, 128, 128).cuda()
        gt = torch.randn(2, 1, 128, 128).cuda()
        
        loss = criterion(pred, gt)
        print(f"✅ 损失计算成功: {loss.item():.6f}")
        
    except Exception as e:
        print(f"❌ 损失函数测试失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！单通道 MRI 训练流水线已就绪")
    print("=" * 60)
    return True


if __name__ == '__main__':
    import sys
    
    success = test_single_channel_pipeline()
    sys.exit(0 if success else 1)


