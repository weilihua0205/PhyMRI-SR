"""
损失函数模块
支持多种损失函数选项用于超分辨率训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 损失函数注册器
losses = {}


def register(name):
    """装饰器：注册损失函数"""
    def decorator(cls):
        losses[name] = cls
        return cls
    return decorator


def make(loss_spec):
    """工厂函数：根据配置创建损失函数"""
    if isinstance(loss_spec, str):
        loss_spec = {'name': loss_spec}
    
    loss_name = loss_spec['name']
    loss_args = loss_spec.get('args', {})
    
    if loss_name not in losses:
        raise ValueError(f'Loss function {loss_name} not found. Available: {list(losses.keys())}')
    
    return losses[loss_name](**loss_args)


@register('L1Loss')
class L1Loss(nn.Module):
    """L1 损失（Mean Absolute Error）"""
    
    def __init__(self):
        super(L1Loss, self).__init__()
    
    def forward(self, pred, gt):
        """
        Args:
            pred: 预测图像 [B, C, H, W] 或像素值 [B, N, C]
            gt: 真实图像 [B, C, H, W] 或像素值 [B, N, C]
        """
        return F.l1_loss(pred, gt)


@register('MSELoss')
class MSELoss(nn.Module):
    """MSE 损失（Mean Squared Error）"""
    
    def __init__(self):
        super(MSELoss, self).__init__()
    
    def forward(self, pred, gt):
        """
        Args:
            pred: 预测图像 [B, C, H, W] 或像素值 [B, N, C]
            gt: 真实图像 [B, C, H, W] 或像素值 [B, N, C]
        """
        return F.mse_loss(pred, gt)


@register('SmoothL1Loss')
class SmoothL1Loss(nn.Module):
    """Smooth L1 损失（Huber Loss）"""
    
    def __init__(self, beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
    
    def forward(self, pred, gt):
        """
        Args:
            pred: 预测图像 [B, C, H, W]
            gt: 真实图像 [B, C, H, W]
        """
        return F.smooth_l1_loss(pred, gt, beta=self.beta)


@register('CharbonnierLoss')
class CharbonnierLoss(nn.Module):
    """Charbonnier 损失（常用于图像恢复任务）"""
    
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, gt):
        """
        Args:
            pred: 预测图像 [B, C, H, W]
            gt: 真实图像 [B, C, H, W]
        """
        diff = pred - gt
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return torch.mean(loss)


@register('PSNRLoss')
class PSNRLoss(nn.Module):
    """PSNR 损失（负 PSNR，用于最大化 PSNR）"""
    
    def __init__(self, max_val=1.0):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val
    
    def forward(self, pred, gt):
        """
        Args:
            pred: 预测图像 [B, C, H, W]
            gt: 真实图像 [B, C, H, W]
        """
        mse = F.mse_loss(pred, gt)
        psnr = 10 * torch.log10(self.max_val ** 2 / mse)
        return -psnr  # 返回负值，使其成为最小化目标



@register('CombinedLoss')
class CombinedLoss(nn.Module):
    """组合损失：多个损失的加权和"""
    
    def __init__(self, losses_dict):
        """
        Args:
            losses_dict: 字典，格式 {'loss_name': weight}
                        例如: {'L1Loss': 1.0, 'CharbonnierLoss': 0.5}
        """
        super(CombinedLoss, self).__init__()
        self.losses = {}
        self.weights = {}
        
        for loss_name, weight in losses_dict.items():
            self.losses[loss_name] = make({'name': loss_name})
            self.weights[loss_name] = weight
    
    def forward(self, pred, gt):
        """
        Args:
            pred: 预测图像 [B, C, H, W]
            gt: 真实图像 [B, C, H, W]
        """
        total_loss = 0
        for loss_name, loss_fn in self.losses.items():
            weight = self.weights[loss_name]
            total_loss += weight * loss_fn(pred, gt)
        
        return total_loss


@register('PerceptualLoss')
class PerceptualLoss(nn.Module):
    """感知损失（基于 VGG 特征）"""
    
    def __init__(self, layers=[3, 8, 15, 22], weights=None):
        """
        Args:
            layers: VGG 层索引列表
            weights: 每层的权重
        """
        super(PerceptualLoss, self).__init__()
        
        # 加载预训练的 VGG16
        try:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features
            
            # 冻结 VGG 参数
            for param in vgg.parameters():
                param.requires_grad = False
            
            self.vgg = vgg.cuda().eval()
            self.layers = layers
            
            if weights is None:
                self.weights = [1.0] * len(layers)
            else:
                self.weights = weights
            
            # 归一化参数（ImageNet 标准）
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
            
        except ImportError:
            raise ImportError('torchvision is required for PerceptualLoss')
    
    def normalize(self, x):
        """ImageNet 归一化"""
        return (x - self.mean) / self.std
    
    def extract_features(self, x):
        """提取 VGG 特征"""
        x = self.normalize(x)
        features = []
        
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        
        return features
    
    def forward(self, pred, gt):
        """
        Args:
            pred: 预测图像 [B, C, H, W]
            gt: 真实图像 [B, C, H, W]
        """
        pred_features = self.extract_features(pred)
        gt_features = self.extract_features(gt)
        
        loss = 0
        for i, (pred_feat, gt_feat) in enumerate(zip(pred_features, gt_features)):
            loss += self.weights[i] * F.mse_loss(pred_feat, gt_feat)
        
        return loss


# 可选：边缘保持损失
@register('EdgeLoss')
class EdgeLoss(nn.Module):
    """边缘保持损失"""
    
    def __init__(self):
        super(EdgeLoss, self).__init__()
        
        # Sobel 算子
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
    
    def compute_edges(self, x):
        """计算图像边缘"""
        # 转换为灰度
        if x.size(1) == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
        
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        
        return edge
    
    def forward(self, pred, gt):
        """
        Args:
            pred: 预测图像 [B, C, H, W]
            gt: 真实图像 [B, C, H, W]
        """
        pred_edges = self.compute_edges(pred)
        gt_edges = self.compute_edges(gt)
        
        return F.l1_loss(pred_edges, gt_edges)


@register('FrequencyLoss')
class FrequencyLoss(nn.Module):
    """频域损失（基于快速傅里叶变换）"""
    
    def __init__(self, loss_type='l1', use_magnitude=True, use_phase=False):
        """
        Args:
            loss_type: 损失类型 ('l1' 或 'l2')
            use_magnitude: 是否使用幅度谱
            use_phase: 是否使用相位谱
        """
        super(FrequencyLoss, self).__init__()
        self.loss_type = loss_type.lower()
        self.use_magnitude = use_magnitude
        self.use_phase = use_phase
        
        if self.loss_type not in ['l1', 'l2']:
            raise ValueError(f'Unsupported loss_type: {loss_type}. Use "l1" or "l2".')
    
    def forward(self, pred, gt):
        """
        Args:
            pred: 预测图像 [B, C, H, W]
            gt: 真实图像 [B, C, H, W]
        """
        # 对每个通道分别计算FFT
        loss = 0
        for c in range(pred.size(1)):
            pred_ch = pred[:, c:c+1]  # [B, 1, H, W]
            gt_ch = gt[:, c:c+1]
            
            # 2D FFT
            pred_fft = torch.fft.fft2(pred_ch, dim=(-2, -1))
            gt_fft = torch.fft.fft2(gt_ch, dim=(-2, -1))
            
            # 计算幅度谱
            if self.use_magnitude:
                pred_mag = torch.abs(pred_fft)
                gt_mag = torch.abs(gt_fft)
                
                if self.loss_type == 'l1':
                    loss += F.l1_loss(pred_mag, gt_mag)
                else:  # l2
                    loss += F.mse_loss(pred_mag, gt_mag)
            
            # 计算相位谱
            if self.use_phase:
                pred_phase = torch.angle(pred_fft)
                gt_phase = torch.angle(gt_fft)
                
                if self.loss_type == 'l1':
                    loss += F.l1_loss(pred_phase, gt_phase)
                else:  # l2
                    loss += F.mse_loss(pred_phase, gt_phase)
        
        return loss


@register('SSIMLoss')
class SSIMLoss(nn.Module):
    """SSIM 损失（Structural Similarity Index Measure）"""
    
    def __init__(self, window_size=11, sigma=1.5, data_range=1.0, k1=0.01, k2=0.03):
        """
        Args:
            window_size: 高斯窗口大小
            sigma: 高斯标准差
            data_range: 数据范围（通常为1.0）
            k1, k2: SSIM 公式中的常数
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        
        # 创建高斯窗口
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        g_2d = g[:, None] * g[None, :]
        self.register_buffer('window', g_2d.view(1, 1, window_size, window_size))
        
        # SSIM 常数
        self.C1 = (k1 * data_range) ** 2
        self.C2 = (k2 * data_range) ** 2
    
    # def gaussian_filter(self, x):
    #     """应用高斯滤波"""
    #     return F.conv2d(x, self.window, padding=self.window_size // 2, groups=x.size(1))
    def gaussian_filter(self, x):
        # x: [B, C, H, W]
        # 确保 kernel 在同一设备和类型
        window = self.window.to(x.device).type_as(x)  # [1,1,k,k]
        c = x.size(1)
        if window.size(0) != c:
            # 将 kernel 重复为每个通道一个（depthwise 卷积需要 [C,1,k,k]）
            window = window.repeat(c, 1, 1, 1)
        padding = self.window_size // 2
        return F.conv2d(x, window, padding=padding, groups=c)
    
    def forward(self, pred, gt):
        """
        Args:
            pred: 预测图像 [B, C, H, W]
            gt: 真实图像 [B, C, H, W]
        """
        if pred.dim() != 4 or gt.dim() != 4:
            raise ValueError('Input must be 4D tensors [B, C, H, W]')
        
        # 计算局部均值
        mu_pred = self.gaussian_filter(pred)
        mu_gt = self.gaussian_filter(gt)
        
        mu_pred_sq = mu_pred ** 2
        mu_gt_sq = mu_gt ** 2
        mu_pred_mu_gt = mu_pred * mu_gt
        
        # 计算局部方差和协方差
        sigma_pred_sq = self.gaussian_filter(pred ** 2) - mu_pred_sq
        sigma_gt_sq = self.gaussian_filter(gt ** 2) - mu_gt_sq
        sigma_pred_gt = self.gaussian_filter(pred * gt) - mu_pred_mu_gt
        
        # 计算 SSIM
        numerator = (2 * mu_pred_mu_gt + self.C1) * (2 * sigma_pred_gt + self.C2)
        denominator = (mu_pred_sq + mu_gt_sq + self.C1) * (sigma_pred_sq + sigma_gt_sq + self.C2)
        
        ssim_map = numerator / denominator
        ssim = ssim_map.mean()
        
        # 返回损失：1 - SSIM
        return 1 - ssim


@register('LaplacianLoss')
class LaplacianLoss(nn.Module):
    """拉普拉斯损失（高频细节增强）- 鼓励网络恢复高频分量"""
    
    def __init__(self, loss_type='l1'):
        """
        Args:
            loss_type: 损失类型 ('l1' 或 'l2')
        使用原理：
            拉普拉斯算子 ∇² 提取二阶导数（高频成分）
            通过最小化拉普拉斯差异，鼓励网络保持边界和纹理
        """
        super(LaplacianLoss, self).__init__()
        self.loss_type = loss_type.lower()
        
        # 拉普拉斯算子（检测边界和高频）
        laplacian_kernel = torch.tensor(
            [[0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0]], 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        
        self.register_buffer('laplacian_kernel', laplacian_kernel)
    
    def compute_laplacian(self, x):
        """计算拉普拉斯算子输出（高频分量）"""
        # x: [B, C, H, W]
        # 将 kernel 复制为每个通道一个（depthwise 卷积）
        c = x.size(1)
        kernel = self.laplacian_kernel.repeat(c, 1, 1, 1)
        kernel = kernel.to(x.device).type_as(x)
        
        # 应用拉普拉斯算子
        laplacian = F.conv2d(x, kernel, padding=1, groups=c)
        
        return laplacian
    
    def forward(self, pred, gt):
        """
        Args:
            pred: 预测图像 [B, C, H, W]
            gt: 真实图像 [B, C, H, W]
        
        返回：
            预测和真实图像的拉普拉斯差异
        """
        pred_lap = self.compute_laplacian(pred)
        gt_lap = self.compute_laplacian(gt)
        
        if self.loss_type == 'l1':
            return F.l1_loss(pred_lap, gt_lap)
        else:  # l2
            return F.mse_loss(pred_lap, gt_lap)


@register('GradientLoss')
class GradientLoss(nn.Module):
    """梯度损失（边界增强）- 鼓励梯度匹配"""
    
    def __init__(self, loss_type='l1'):
        """
        Args:
            loss_type: 损失类型 ('l1' 或 'l2')
        使用原理：
            通过匹配一阶梯度，鼓励网络保持边界清晰
            梯度大 = 边界处，梯度小 = 平滑区域
        """
        super(GradientLoss, self).__init__()
        self.loss_type = loss_type.lower()
    
    def compute_gradients(self, x):
        """计算 x 和 y 方向的梯度"""
        # Sobel 算子
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]], 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        
        c = x.size(1)
        kernel_x = sobel_x.repeat(c, 1, 1, 1).to(x.device).type_as(x)
        kernel_y = sobel_y.repeat(c, 1, 1, 1).to(x.device).type_as(x)
        
        grad_x = F.conv2d(x, kernel_x, padding=1, groups=c)
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=c)
        
        return grad_x, grad_y
    
    def forward(self, pred, gt):
        """
        Args:
            pred: 预测图像 [B, C, H, W]
            gt: 真实图像 [B, C, H, W]
        
        返回：
            预测和真实图像的梯度差异
        """
        pred_gx, pred_gy = self.compute_gradients(pred)
        gt_gx, gt_gy = self.compute_gradients(gt)
        
        if self.loss_type == 'l1':
            loss_x = F.l1_loss(pred_gx, gt_gx)
            loss_y = F.l1_loss(pred_gy, gt_gy)
        else:  # l2
            loss_x = F.mse_loss(pred_gx, gt_gx)
            loss_y = F.mse_loss(pred_gy, gt_gy)
        
        return (loss_x + loss_y) / 2


@register('SharpnessLoss')
class SharpnessLoss(nn.Module):
    """锐度损失（对比度增强）- 鼓励更尖锐的边界"""
    
    def __init__(self, lambda_param=0.5):
        """
        Args:
            lambda_param: 锐化强度参数
        使用原理：
            通过负反馈，鼓励网络增加对比度和锐度
            S = I - λ * L(I)，其中 L 是拉普拉斯算子
        """
        super(SharpnessLoss, self).__init__()
        self.lambda_param = lambda_param
        
        # 拉普拉斯算子
        laplacian_kernel = torch.tensor(
            [[0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0]], 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        
        self.register_buffer('laplacian_kernel', laplacian_kernel)
    
    def compute_laplacian(self, x):
        """计算拉普拉斯算子输出"""
        c = x.size(1)
        kernel = self.laplacian_kernel.repeat(c, 1, 1, 1)
        kernel = kernel.to(x.device).type_as(x)
        
        laplacian = F.conv2d(x, kernel, padding=1, groups=c)
        return laplacian
    
    def forward(self, pred, gt):
        """
        Args:
            pred: 预测图像 [B, C, H, W]
            gt: 真实图像 [B, C, H, W]
        
        返回：
            锐化后的预测与 GT 的 L1 差异
        """
        # 对 pred 进行锐化：I_sharp = I - λ * L(I)
        pred_lap = self.compute_laplacian(pred)
        pred_sharpened = pred - self.lambda_param * pred_lap
        
        # 对 gt 也进行相同锐化
        gt_lap = self.compute_laplacian(gt)
        gt_sharpened = gt - self.lambda_param * gt_lap
        
        # 计算锐化图像之间的差异
        return F.l1_loss(pred_sharpened, gt_sharpened)


@register('ScaleRegLoss')
class ScaleRegLoss(nn.Module):
    """高斯核 Scale 软约束损失
    
    鼓励网络选择更小的高斯核（cho1/cho3），从而恢复更多高频细节。
    损失计算方式：对每个高斯核的 cho1+cho3 取均值，作为最小化目标。
    
    优点：
    - 不限制字典范围，梯度信号完整（避免直接压缩字典导致的崩溃）
    - 网络仍可在需要时选大核（边缘过渡区），但有一个"倾向小核"的软约束
    - 与重建损失共同优化，通过权重 scale_reg_weight 调节强度
    
    注意：
    - para 来自 model.last_para，需在 train_epoch 中显式传入
    - 权重建议从 0.01 开始，观察 cho1/cho3 均值的变化后再调整
    """
    
    def __init__(self, max_scale: float = 0.5):
        """
        Args:
            max_scale: cho1 和 cho3 的"期望上限"，超过此值才产生惩罚。
                      设为 0.5 意味着鼓励网络使用 scale ≤ 0.5 的高斯核。
                      设为 0.0 则对所有高斯核的 scale 都产生惩罚（最强约束）。
        """
        super(ScaleRegLoss, self).__init__()
        self.max_scale = max_scale
    
    def forward(self, para: torch.Tensor) -> torch.Tensor:
        """
        Args:
            para: [bs, N, 3]，来自 model.last_para
                  para[..., 0] = cho1（x 方向 scale）
                  para[..., 2] = cho3（y 方向 scale）
        
        Returns:
            scalar loss：超过 max_scale 的 cho1/cho3 均值之和
        """
        cho1 = para[..., 0]  # [bs, N]
        cho3 = para[..., 2]  # [bs, N]
        
        # relu(x - max_scale)：仅对超过阈值的部分产生惩罚（Hinge 形式）
        # 若 max_scale=0.0，则等价于直接最小化均值
        penalty_cho1 = torch.relu(cho1 - self.max_scale).mean()
        penalty_cho3 = torch.relu(cho3 - self.max_scale).mean()
        
        return penalty_cho1 + penalty_cho3
    
    
@register('PhysicsRatioLoss')
class PhysicsRatioLoss(nn.Module):
    """物理先验占比正则损失
    
    直接惩罚 phy_ratio（物理项占比）低于目标阈值的情况，
    防止网络退化为"物理项≈0，全靠δ重建"的局部最优解。
    
    计算方式：
        phy_ratio = |signal_physics| / (|signal_physics| + |δ| + ε)
        loss = mean(relu(target_ratio - phy_ratio))
    
    即只对低于 target_ratio 的 token 产生惩罚（Hinge 形式），
    不限制 phy_ratio 上限，允许物理项完全主导。
    
    建议权重：physics_ratio_weight = 0.1，target_ratio = 0.3
    """

    def __init__(self, target_ratio: float = 0.3):
        """
        Args:
            target_ratio: 期望的物理项最低占比（0~1）。
                          设为 0.3 表示惩罚 phy_ratio < 0.3 的情况，
                          不强制要求 phy_ratio ≥ 0.3，而是提供软约束方向。
        """
        super(PhysicsRatioLoss, self).__init__()
        self.target_ratio = target_ratio

    def forward(self, signal_physics: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal_physics: [bs*N, 1] 或 [bs, N, 1]，物理项输出
            delta:          [bs*N, 1] 或 [bs, N, 1]，修正项输出

        Returns:
            scalar loss
        """
        phy_ratio = signal_physics.abs() / (signal_physics.abs() + delta.abs() + 1e-8)
        # Hinge：只惩罚低于目标占比的部分
        penalty = torch.relu(self.target_ratio - phy_ratio).mean()
        return penalty

