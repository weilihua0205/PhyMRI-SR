import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import register

@register('npy-folder')
class NpyFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache
        self.root_path = root_path

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]
        self.filenames = [fn for fn in filenames if fn.endswith('.npy')]

        self.files = []
        for filename in filenames:
            if filename.endswith('.npy'):
                file = os.path.join(root_path, filename)

                if cache == 'none':
                    self.files.append(file)

                elif cache == 'bin':
                    bin_root = os.path.join(os.path.dirname(root_path),
                        '_bin_' + os.path.basename(root_path))
                    if not os.path.exists(bin_root):
                        os.mkdir(bin_root)
                        print('mkdir', bin_root)
                    bin_file = os.path.join(
                        bin_root, filename.split('.')[0] + '.pkl')
                    if not os.path.exists(bin_file):
                        with open(bin_file, 'wb') as f:
                            pickle.dump(np.load(file), f)
                        print('dump', bin_file)
                    self.files.append(bin_file)

                elif cache == 'in_memory':
                    self.files.append(torch.from_numpy(np.load(file).astype('float32')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            arr = np.load(x)
            t = torch.from_numpy(arr.astype('float32')).clone()

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                arr = pickle.load(f)
            t = torch.from_numpy(arr.astype('float32')).clone()

        elif self.cache == 'in_memory':
            t = x.clone()

        # normalize shape to [C,H,W]
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3:
            # assume already (C,H,W)
            pass
        else:
            raise ValueError(f'Unsupported npy shape {t.shape} for {x}')
        return t.contiguous()


@register('paired-npy-folders')
class PairedNpyFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = NpyFolder(root_path_1, **kwargs)
        self.dataset_2 = NpyFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]


@register('paired-npy-folders-with-mask')
class PairedNpyFoldersWithMask(Dataset):
    """
    同时加载 LR、HR 图像和对应的分割 mask。
    mask 与 HR 图像同分辨率，文件名必须与 HR 图像同名（不同目录）。

    Args:
        root_path_lr:   LR 图像目录
        root_path_hr:   HR 图像目录
        root_path_mask: 分割 mask 目录（.npy，标签值 0/1/2/3 等整数）
        kwargs:         传递给内部 NpyFolder 的参数（cache、repeat 等）
    """

    def __init__(self, root_path_lr, root_path_hr, root_path_mask, **kwargs):
        self.lr_dataset   = NpyFolder(root_path_lr,   **kwargs)
        self.hr_dataset   = NpyFolder(root_path_hr,   **kwargs)
        self.mask_dataset = NpyFolder(root_path_mask, **kwargs)
        assert len(self.lr_dataset) == len(self.hr_dataset) == len(self.mask_dataset), (
            f"LR/HR/Mask 数据集长度不一致: "
            f"LR={len(self.lr_dataset)}, HR={len(self.hr_dataset)}, Mask={len(self.mask_dataset)}"
        )
        if self.lr_dataset.filenames != self.hr_dataset.filenames or self.lr_dataset.filenames != self.mask_dataset.filenames:
            raise ValueError(
                'LR/HR/mask filenames are not aligned. '
                'Please ensure the three folders contain the same sorted .npy filenames.'
            )

    def __len__(self):
        return len(self.lr_dataset)

    def __getitem__(self, idx):
        lr   = self.lr_dataset[idx]    # [1, H_lr, W_lr]
        hr   = self.hr_dataset[idx]    # [1, H_hr, W_hr]
        mask = self.mask_dataset[idx]  # [1, H_hr, W_hr] 或 [H_hr, W_hr]
        # 统一 mask 为 [1, H, W]，保持整数标签（float32 存储）
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return lr, hr, mask


@register('npy-folder-with-mask')
class NpyFolderWithMask(Dataset):
    """Load HR images and aligned masks for dynamic multi-scale fine-tuning."""

    def __init__(self, root_path_hr, root_path_mask, **kwargs):
        self.hr_dataset = NpyFolder(root_path_hr, **kwargs)
        self.mask_dataset = NpyFolder(root_path_mask, **kwargs)
        assert len(self.hr_dataset) == len(self.mask_dataset), (
            f"HR/mask length mismatch: HR={len(self.hr_dataset)}, Mask={len(self.mask_dataset)}"
        )
        if self.hr_dataset.filenames != self.mask_dataset.filenames:
            raise ValueError(
                'HR/mask filenames are not aligned. '
                'Please ensure the two folders contain the same sorted .npy filenames.'
            )

    def __len__(self):
        return len(self.hr_dataset)

    def __getitem__(self, idx):
        hr = self.hr_dataset[idx]
        mask = self.mask_dataset[idx]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return hr, mask
