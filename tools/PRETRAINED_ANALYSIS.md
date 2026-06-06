# 预训练模型分析报告

## 模型基本信息

### 训练状态
- **训练轮数**: 62 epochs
- **总参数量**: 45.08M (45,081,524 参数)
- **模型架构**: ContinuousGaussian with HAT encoder

### 模块参数分布
```
encoder (HAT):     41.06M (91.1%)  - 主干编码器
conv1:              1.18M (2.6%)   - 额外卷积层
mlp (color):        0.96M (2.1%)   - 颜色预测MLP  ⭐
mlp_offset:         0.96M (2.1%)   - 偏移预测MLP
mlp_vector:         0.92M (2.0%)   - 向量生成MLP
```

---

## 关键发现

### 1. 🔴 **Color MLP 缺少输出激活函数**

**预训练模型配置** (`ContinuousSR.pth`):
```python
# models/mlp.py 第40行
def forward(self, x):
    shape = x.shape[:-1]
    x = self.layers(x.contiguous().view(-1, x.shape[-1]))
    # x = torch.sigmoid(x)  ⬅️ 被注释掉了！
    return x.view(*shape, -1)
```

**Color MLP输出层参数统计**:
```
mlp.layers.10.weight: shape=(3, 64)
  range: [-0.295261, 0.293253]
  mean: 0.007714, std: 0.166305
  
mlp.layers.10.bias: shape=(3,)
  range: [0.005345, 0.006191]
  mean: 0.005730, std: 0.000428
```

**问题分析**:
- Weight范围: [-0.30, +0.29]，无约束
- 如果没有Sigmoid激活，MLP输出可以是 **任意实数值**
- 但图像像素值必须在 [0, 1] 范围内
- **当前代码在`forward`中没有Sigmoid，导致输出可能为负值或超出[0,1]**

---

### 2. 🟡 你的训练模型对比

#### 未训练模型 (随机初始化)
```
pred range: [-0.1881, 0.0443]
pred mean: -0.0058
pred (clamped): [0.0000, 0.0443], mean=0.0004
```

#### 训练5个epoch后 (checkpoint_latest.pth)
```
pred range: [-0.0167, 0.0419]  
pred mean: 0.0013
pred (clamped): [0.0000, 0.0419], mean=0.0014
```

#### Ground Truth
```
gt range: [0.0000, 0.8353]
gt mean: 0.2649
```

**结论**:
- 训练后输出仍然非常小 (max=0.04 vs gt_mean=0.26)
- 即使有63个epochs的预训练模型，**如果没有输出激活函数，模型也无法将输出约束到正确范围**
- 当前在`visualize_training.py`中使用`.clamp(0,1)`作为补救，但这会导致：
  - 负值被截断为0（丢失信息）
  - 无法学习到高亮度区域（max只到0.04，远低于1.0）

---

## 与预训练模型的差异

### 编码器差异
**预训练模型 (HAT)**:
```python
encoder_spec: {
  'name': 'hat',  # Hybrid Attention Transformer
  'embed_dim': 180,
  'depths': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],  # 12层，每层6个block
  'num_heads': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
  'window_size': 16,
  'img_size': 64
}
参数量: 41.06M
```

**你的模型 (EDSR)**:
```python
encoder_spec: {
  'name': 'edsr-baseline',
  'n_resblocks': 16,
  'n_feats': 64,
  'scale': 1
}
参数量: ~1-2M (估计)
```

**差异**:
- HAT是Transformer架构 (41M参数)，EDSR是CNN架构 (~2M参数)
- HAT有更强的表达能力，但也需要更多数据和训练时间
- 预训练模型训练了62个epochs，你只训练了5个epochs

### Color MLP差异
**预训练模型**:
```python
fc_spec: {
  'name': 'mlp',
  'out_dim': 3,
  'hidden_list': [256, 256, 256, 256]  # 4层hidden
}
```

**你的模型**:
```python
mlp_spec: {
  'name': 'mlp',
  'in_dim': 256,
  'out_dim': 3,
  'hidden_list': [512, 1024, 256, 128, 64]  # 5层hidden
}
```

---

## 根本原因总结

### 为什么你的模型输出接近0？

1. **✅ 数据正常**: LR和GT都在[0,1]范围，无归一化问题

2. **❌ 模型输出无约束**: 
   - Color MLP的`forward`中 `torch.sigmoid(x)` 被注释
   - 预训练模型也是这样训练的，但可能通过以下方式补偿：
     - 在训练时使用了不同的损失函数
     - 训练时间更长，让模型自己学会输出小值
     - 可能在其他地方有额外的处理

3. **⚠️ 训练不足**: 
   - 5 epochs vs 63 epochs (预训练模型)
   - PSNR: 2-10 dB vs 应该>25 dB

4. **⚠️ 模型容量不匹配**:
   - EDSR (2M参数) vs HAT (41M参数)
   - 更小的模型可能需要更长时间收敛

---

## 修复建议

### 方案1: 取消注释Sigmoid（推荐）✅
```python
# models/mlp.py 第40行
def forward(self, x):
    shape = x.shape[:-1]
    x = self.layers(x.contiguous().view(-1, x.shape[-1]))
    x = torch.sigmoid(x)  # ⬅️ 取消注释
    return x.view(*shape, -1)
```

**优点**:
- 强制输出在[0,1]，符合图像像素值
- 避免梯度爆炸/消失
- 训练更稳定

**缺点**:
- 需要重新训练（之前的checkpoint不兼容）

### 方案2: 在模型forward中添加Sigmoid
```python
# models/gaussian.py 的forward方法中
def forward(self, inp, scale):
    ...
    # 预测颜色
    colors = self.mlp(feat_flatten)  # (bs*hf*wf, 4, 3)
    colors = torch.sigmoid(colors)   # ⬅️ 添加这行
    ...
```

**优点**:
- 不修改MLP基类，只在ContinuousGaussian中处理
- 可以使用现有的预训练MLP权重

**缺点**:
- 需要小心处理，确保不重复应用Sigmoid

### 方案3: 调整训练配置
```yaml
# 暂时禁用PerceptualLoss
loss:
  name: L1Loss

# 增大学习率
optimizer:
  args:
    lr: 5.0e-4  # 从1e-4增大

# 延长训练时间
num_epochs: 100  # 从30增加到100
```

---

## 下一步行动

### 立即执行:
1. **检查models/mlp.py第40行**，确认Sigmoid是否被注释
2. **决定修复方案**: 方案1（修改MLP）或方案2（修改Gaussian）
3. **重新训练**: 从头开始训练，或继续现有checkpoint

### 验证步骤:
1. 修改后运行 `diagnose_training.py`
2. 检查模型输出是否在[0,1]范围
3. 训练5个epochs，观察PSNR是否提升到>20 dB
4. 检查可视化是否不再是黑色方块

---

## 附录：为什么预训练模型可以工作？

**猜测1**: 预训练时没有注释Sigmoid
- 训练完后为了某种原因注释掉了
- 或者代码版本不同

**猜测2**: 训练时使用了范围更宽的损失
- 可能GT也做了变换（如log space）
- 或者使用了HDR图像训练

**猜测3**: 模型学会了自我约束
- 63个epochs足够长，MSE loss迫使输出接近GT范围
- 但这对于小模型(EDSR)和短训练(5epochs)不适用

**需要验证**: 用预训练模型做inference，看输出范围
