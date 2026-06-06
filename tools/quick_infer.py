# import torch, os, numpy as np
# from torchvision import transforms
# from PIL import Image
# import models

# # 修改为你的checkpoint和图片路径
# ckpt = r'/home/ght/MRIxField/ContinuousSR-main/save/div2k_sample_swinir_1000epochs_fix2/checkpoint_best.pth'
# img_path = r'/home/ght/MRIxField/natural_images/DIV2K_train_HR_sample/0001.png'
# out_img = r'/home/ght/MRIxField/ContinuousSR-main/save/div2k_sample_swinir_1000epochs_fix2/quick_check/debug_pred.png'
# out_npy = r'/home/ght/MRIxField/ContinuousSR-main/save/div2k_sample_swinir_1000epochs_fix2/quick_check/debug_pred.npy'
# gpu = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu

# img_pil = Image.open(img_path).convert('RGB')
# img = transforms.ToTensor()(img_pil)
# # Ensure intensity is normalized to [0,1] (handle cases where ToTensor didn't)
# if img.max() > 1.1:
#     img = img / 255.0
# img = img.cuda()
# img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(256,256), mode='bicubic', align_corners=False).squeeze(0)
# spec = torch.load(ckpt)['model']
# model = models.make(spec, load_sd=True).cuda().eval()

# # scale 与训练使用相同格式，比如 '2,2' -> tensor([[2,2]])
# scale = torch.tensor([[4,4]]).cuda()

# with torch.no_grad():
#     pred = model(img.unsqueeze(0), scale).squeeze(0)  # C,H,W
#     print('PRED shape', pred.shape, 'min/max/mean', pred.min().item(), pred.max().item(), pred.mean().item())
#     # 保存 raw numpy（未 clamp）
#     np.save(out_npy, pred.cpu().numpy())
#     # 保存可视化PNG
#     from torchvision.transforms import ToPILImage
#     p = pred.clamp(0,1).cpu()
#     ToPILImage()(p).save(out_img)
#     print('Saved', out_img, out_npy)

import numpy as np, matplotlib.pyplot as plt
arr = np.load('debug_demo_pred.npy')  # shape e.g. (3,H,W)
print(arr.shape, arr.min(), arr.max())
plt.imshow(arr.transpose(1,2,0).clip(0,1))  # 若 shape (3,H,W) 转为 H,W,3 显示
plt.show()