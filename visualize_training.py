import json
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image


class TrainingVisualizer:
    """Utility for saving training curves, summaries, and validation image grids."""

    def __init__(self, save_dir, config):
        self.save_dir = save_dir
        self.config = config
        self.vis_dir = os.path.join(save_dir, 'visualizations')
        self.curve_dir = os.path.join(self.vis_dir, 'curves')
        self.image_dir = os.path.join(self.vis_dir, 'images')
        self.fid_interval = int(config.get('fid', {}).get('interval', 10))

        os.makedirs(self.curve_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

        # Keep a persistent metric history so resumed training can continue the same plots.
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'val_dists': [],
            'val_fid': [],
            'val_fid_epoch': [],
            'val_kid': [],
            'val_kid_epoch': [],
            'val_hfen': [],
            'val_hfen_epoch': [],
            'lr': [],
        }
        self.history_file = os.path.join(self.vis_dir, 'training_history.json')
        self.load_history()

    def load_history(self):
        if not os.path.exists(self.history_file):
            return
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
            # Add missing keys when loading history files produced by older runs.
            self.history.setdefault('val_dists', [])
            self.history.setdefault('val_fid', [])
            self.history.setdefault('val_fid_epoch', [])
            self.history.setdefault('val_kid', [])
            self.history.setdefault('val_kid_epoch', [])
            self.history.setdefault('val_hfen', [])
            self.history.setdefault('val_hfen_epoch', [])
            print(f'==> Loaded training history from {self.history_file}')
        except Exception:
            print('==> Failed to load history, starting fresh')

    def save_history(self):
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)

    def update_metrics(self, epoch, train_loss, val_psnr=None, val_ssim=None, val_dists=None, lr=None,
                       val_fid=None, val_kid=None, val_hfen=None):
        self.history['epoch'].append(int(epoch))
        self.history['train_loss'].append(float(train_loss))

        # Store optional validation metrics only when they are available and finite.
        if val_psnr is not None:
            self.history['val_psnr'].append(float(val_psnr))
        if val_ssim is not None:
            self.history['val_ssim'].append(float(val_ssim))
        if val_dists is not None and np.isfinite(val_dists):
            self.history['val_dists'].append(float(val_dists))
        if val_fid is not None and np.isfinite(val_fid):
            self.history['val_fid'].append(float(val_fid))
            self.history['val_fid_epoch'].append(int(epoch))
        if val_kid is not None and np.isfinite(val_kid):
            self.history['val_kid'].append(float(val_kid))
            self.history['val_kid_epoch'].append(int(epoch))
        if val_hfen is not None and np.isfinite(val_hfen):
            self.history['val_hfen'].append(float(val_hfen))
            self.history['val_hfen_epoch'].append(int(epoch))
        if lr is not None:
            self.history['lr'].append(float(lr))

        self.save_history()

    def _plot_metric(self, ax, x, y, color, title, ylabel, lower_is_better=False,
                     linestyle='-', marker=None):
        if not y:
            ax.axis('off')
            return

        ax.plot(x, y, color=color, linestyle=linestyle, marker=marker, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Mark the best observed value directly on each curve for quick inspection.
        best_value = min(y) if lower_is_better else max(y)
        best_idx = y.index(best_value)
        best_epoch = x[best_idx]
        ax.axhline(y=best_value, color='r', linestyle='--', alpha=0.5)
        ax.text(
            0.02, 0.98, f'Best: {best_value:.4f} @ Epoch {best_epoch}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )

    def plot_curves(self, epoch, save_numbered=True):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Training Progress - Epoch {epoch}', fontsize=16, fontweight='bold')

        epochs = self.history['epoch']

        if self.history['train_loss']:
            axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Epoch', fontsize=12)
            axes[0, 0].set_ylabel('Loss', fontsize=12)
            axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].axis('off')

        self._plot_metric(
            axes[0, 1],
            epochs[:len(self.history['val_psnr'])],
            self.history['val_psnr'],
            'g',
            'Validation PSNR',
            'PSNR (dB)',
        )
        self._plot_metric(
            axes[1, 0],
            epochs[:len(self.history['val_ssim'])],
            self.history['val_ssim'],
            'orange',
            'Validation SSIM',
            'SSIM',
        )

        if self.history['val_dists']:
            # Prefer DISTS in the fourth panel when available because it is computed every validation.
            self._plot_metric(
                axes[1, 1],
                epochs[:len(self.history['val_dists'])],
                self.history['val_dists'],
                'm',
                'Validation DISTS',
                'DISTS',
                lower_is_better=True,
            )
        elif self.history['val_fid']:
            fid_epochs = self.history.get('val_fid_epoch', [])
            if len(fid_epochs) != len(self.history['val_fid']):
                # Older histories may not record exact FID epochs, so reconstruct them from the interval.
                fid_epochs = [self.fid_interval * (i + 1) for i in range(len(self.history['val_fid']))]
            self._plot_metric(
                axes[1, 1],
                fid_epochs,
                self.history['val_fid'],
                'm',
                f'Validation FID (Every {self.fid_interval} Epochs)',
                'FID',
                lower_is_better=True,
                marker='o',
            )
        elif self.history['lr']:
            lr_epochs = epochs[:len(self.history['lr'])]
            axes[1, 1].plot(lr_epochs, self.history['lr'], 'r-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
            axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3, which='both')
        else:
            axes[1, 1].axis('off')

        plt.tight_layout()

        if save_numbered:
            save_path = os.path.join(self.curve_dir, f'training_curves_epoch{epoch:04d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # Always update a stable "latest" filename for dashboards or quick checks.
        latest_path = os.path.join(self.curve_dir, 'training_curves_latest.png')
        plt.savefig(latest_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f'==> Saved training curves to {latest_path}')

    def generate_brain_mask(self, img, threshold=0.05):
        batch_size = img.shape[0]
        mask = torch.zeros(batch_size, 1, img.shape[-2], img.shape[-1], dtype=torch.float32, device=img.device)

        for b in range(batch_size):
            img_b = img[b]
            # Use a relative threshold so the mask adapts to each slice intensity range.
            max_vals = torch.amax(torch.abs(img_b.view(img_b.shape[0], -1)), dim=1)
            max_val = torch.max(max_vals)
            if max_val > 0:
                threshold_val = max_val * threshold
                mask[b] = torch.any(torch.abs(img_b) > threshold_val, dim=0, keepdim=True).float()

        return mask

    def enhance_contrast(self, img, mask, enhancement_factor=1.5):
        # Gamma correction improves visual contrast inside the brain mask without changing saved tensors.
        gamma = 1.0 / enhancement_factor
        return ((img.clamp(1e-6, 1.0) ** gamma) * mask.expand_as(img)).clamp(0, 1)

    def apply_mask_to_image(self, img, mask):
        return img * mask.expand_as(img)

    def visualize_results(self, model, val_loader, epoch, num_samples=4, save_numbered=True):
        """Save clean HR | SR | LR comparisons without text overlays."""
        model.eval()

        data_norm = self.config.get('data_norm')
        if data_norm:
            # Match training normalization before inference and invert it for visualization.
            inp_sub = torch.FloatTensor(data_norm['inp']['sub']).view(1, -1, 1, 1).cuda()
            inp_div = torch.FloatTensor(data_norm['inp']['div']).view(1, -1, 1, 1).cuda()
            gt_sub = torch.FloatTensor(data_norm['gt']['sub']).view(1, -1, 1, 1).cuda()
            gt_div = torch.FloatTensor(data_norm['gt']['div']).view(1, -1, 1, 1).cuda()

        images_to_save = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_samples:
                    break

                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()

                inp = (batch['inp'] - inp_sub) / inp_div if data_norm else batch['inp']
                scale = batch['scale']
                gt = batch['gt']
                pred = model(inp, scale)

                if data_norm:
                    pred = pred * gt_div + gt_sub

                pred = pred.clamp(0, 1)
                brain_mask = self.generate_brain_mask(gt, threshold=0.05)

                inp_upsampled = torch.nn.functional.interpolate(
                    batch['inp'],
                    size=gt.shape[-2:],
                    mode='bicubic',
                    align_corners=False,
                ).clamp(0, 1)

                # Restrict visualization to foreground anatomy so background does not dominate contrast.
                inp_upsampled = self.apply_mask_to_image(inp_upsampled, brain_mask)
                pred = self.apply_mask_to_image(pred, brain_mask)
                gt = self.apply_mask_to_image(gt, brain_mask)

                inp_upsampled = self.enhance_contrast(inp_upsampled, brain_mask, enhancement_factor=1.2)
                pred = self.enhance_contrast(pred, brain_mask, enhancement_factor=1.2)
                gt = self.enhance_contrast(gt, brain_mask, enhancement_factor=1.2)

                if inp_upsampled.shape[1] == 1:
                    inp_upsampled = inp_upsampled.repeat(1, 3, 1, 1)
                if pred.shape[1] == 1:
                    pred = pred.repeat(1, 3, 1, 1)
                if gt.shape[1] == 1:
                    gt = gt.repeat(1, 3, 1, 1)

                # Each row is HR | SR | bicubic LR, concatenated horizontally.
                images_to_save.append(torch.cat([gt, pred, inp_upsampled], dim=3))

        if images_to_save:
            # Stack sample rows vertically into one image grid.
            grid = torch.cat(images_to_save, dim=2)

            if save_numbered:
                save_path = os.path.join(self.image_dir, f'results_epoch{epoch:04d}.png')
                save_image(grid, save_path, nrow=1, padding=2, normalize=False)
                print(f'==> Saved numbered visualization to {save_path}')

            # Keep a stable latest image for monitoring the current training state.
            latest_path = os.path.join(self.image_dir, 'results_latest.png')
            save_image(grid, latest_path, nrow=1, padding=2, normalize=False)
            print(f'==> Saved visualization to {latest_path}')

        model.train()

    def create_summary_report(self, epoch):
        report_path = os.path.join(self.vis_dir, f'summary_epoch{epoch:04d}.txt')

        # Write a text summary that can be read without opening plots or JSON history.
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('=' * 60 + '\n')
            f.write(f'Training Summary - Epoch {epoch}\n')
            f.write(f'Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write('=' * 60 + '\n\n')

            if self.history['train_loss']:
                f.write(f'Latest Training Loss: {self.history["train_loss"][-1]:.6f}\n')
                f.write(f'Best Training Loss: {min(self.history["train_loss"]):.6f}\n\n')

            if self.history['val_psnr']:
                best_psnr = max(self.history['val_psnr'])
                best_psnr_epoch = self.history['epoch'][self.history['val_psnr'].index(best_psnr)]
                f.write(f'Latest Validation PSNR: {self.history["val_psnr"][-1]:.4f} dB\n')
                f.write(f'Best Validation PSNR: {best_psnr:.4f} dB\n')
                f.write(f'Best PSNR at Epoch: {best_psnr_epoch}\n\n')

            if self.history['val_ssim']:
                best_ssim = max(self.history['val_ssim'])
                best_ssim_epoch = self.history['epoch'][self.history['val_ssim'].index(best_ssim)]
                f.write(f'Latest Validation SSIM: {self.history["val_ssim"][-1]:.4f}\n')
                f.write(f'Best Validation SSIM: {best_ssim:.4f}\n')
                f.write(f'Best SSIM at Epoch: {best_ssim_epoch}\n\n')

            if self.history['val_fid']:
                fid_epochs = self.history.get('val_fid_epoch', [])
                if len(fid_epochs) != len(self.history['val_fid']):
                    fid_epochs = [self.fid_interval * (i + 1) for i in range(len(self.history['val_fid']))]
                best_fid = min(self.history['val_fid'])
                best_fid_epoch = fid_epochs[self.history['val_fid'].index(best_fid)]
                f.write(f'Latest Validation FID: {self.history["val_fid"][-1]:.4f} (Epoch {fid_epochs[-1]})\n')
                f.write(f'Best Validation FID: {best_fid:.4f}\n')
                f.write(f'Best FID at Epoch: {best_fid_epoch}\n\n')

            if self.history['val_kid']:
                kid_epochs = self.history.get('val_kid_epoch', [])
                if len(kid_epochs) != len(self.history['val_kid']):
                    kid_epochs = [self.fid_interval * (i + 1) for i in range(len(self.history['val_kid']))]
                best_kid = min(self.history['val_kid'])
                best_kid_epoch = kid_epochs[self.history['val_kid'].index(best_kid)]
                f.write(f'Latest Validation KID: {self.history["val_kid"][-1]:.6f} (Epoch {kid_epochs[-1]})\n')
                f.write(f'Best Validation KID: {best_kid:.6f}\n')
                f.write(f'Best KID at Epoch: {best_kid_epoch}\n\n')

            if self.history['val_hfen']:
                hfen_epochs = self.history.get('val_hfen_epoch', [])
                if len(hfen_epochs) != len(self.history['val_hfen']):
                    hfen_epochs = [self.fid_interval * (i + 1) for i in range(len(self.history['val_hfen']))]
                best_hfen = min(self.history['val_hfen'])
                best_hfen_epoch = hfen_epochs[self.history['val_hfen'].index(best_hfen)]
                f.write(f'Latest Validation HFEN: {self.history["val_hfen"][-1]:.6f} (Epoch {hfen_epochs[-1]})\n')
                f.write(f'Best Validation HFEN: {best_hfen:.6f}\n')
                f.write(f'Best HFEN at Epoch: {best_hfen_epoch}\n\n')

            if self.history['lr']:
                f.write(f'Current Learning Rate: {self.history["lr"][-1]:.2e}\n')

            f.write('\n' + '=' * 60 + '\n')
            f.write('Training Configuration:\n')
            f.write('=' * 60 + '\n')
            f.write(f'Model: {self.config.get("model", {}).get("name", "N/A")}\n')
            f.write(f'Batch Size: {self.config.get("batch_size", "N/A")}\n')
            f.write(f'Num Epochs: {self.config.get("num_epochs", "N/A")}\n')
            f.write(f'Learning Rate: {self.config.get("optimizer", {}).get("args", {}).get("lr", "N/A")}\n')

        latest_report = os.path.join(self.vis_dir, 'summary_latest.txt')
        # Mirror the epoch report to a stable latest filename for quick access.
        with open(report_path, 'r', encoding='utf-8') as src, open(latest_report, 'w', encoding='utf-8') as dst:
            dst.write(src.read())

        print(f'==> Saved summary report to {report_path}')
