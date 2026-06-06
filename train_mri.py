"""
Train a mask-guided ContinuousSR model with simulated multi-resolution 64mT-3T data and fastMRI data.

This is the main MRI super-resolution training entry point. The dataset, model,
losses, optional mask guidance, and validation behavior are controlled by the YAML
configuration passed through --config.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

import datasets
from evaluate import validate
from losses import make as make_loss
from losses import PhysicsRatioLoss, ScaleRegLoss
import models
from schedulers import make_scheduler
import utils
from visualize_training import TrainingVisualizer


def setup_seed(seed):
    """Set all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Set a deterministic seed for each DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_selection_score(val_psnr, val_ssim=None, val_dists=None, config=None):
    """Return a scalar score for best-checkpoint selection. Higher is better."""
    config = config or {}
    mode = str(config.get('selection_metric', 'psnr')).lower()

    if mode == 'psnr':
        return float(val_psnr), f'PSNR={val_psnr:.4f}'
    if mode == 'ssim':
        ssim = 0.0 if val_ssim is None else float(val_ssim)
        return ssim, f'SSIM={ssim:.4f}'
    if mode == 'dists':
        dists = 1e9 if val_dists is None else float(val_dists)
        return -dists, f'DISTS={dists:.4f}'
    if mode == 'composite':
        weights = config.get('selection_weights', {})
        w_psnr = float(weights.get('psnr', 0.0))
        w_ssim = float(weights.get('ssim', 1.0))
        w_dists = float(weights.get('dists', 1.0))
        psnr = float(val_psnr)
        ssim = 0.0 if val_ssim is None else float(val_ssim)
        dists = 0.0 if val_dists is None else float(val_dists)
        score = w_psnr * psnr + w_ssim * ssim - w_dists * dists
        desc = (
            f'Composite={score:.4f} '
            f'(w_psnr={w_psnr}, w_ssim={w_ssim}, w_dists={w_dists}; '
            f'PSNR={psnr:.4f}, SSIM={ssim:.4f}, DISTS={dists:.4f})'
        )
        return score, desc

    raise ValueError(f'Unsupported selection_metric: {mode}')


def prepare_training(config, resume_path=None):
    """Build or resume the model, optimizer, and learning-rate scheduler."""

    # Build the model from a checkpoint when resuming, otherwise from the config.
    if resume_path and os.path.exists(resume_path):
        print(f'==> Resuming from checkpoint: {resume_path}')
        # Some checkpoints contain numpy scalar metadata saved by older training code.
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
        model_spec = checkpoint['model']
        model = models.make(model_spec, load_sd=True)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_metric = checkpoint.get('best_metric', 0.0)
        print(f'Resumed from epoch {start_epoch - 1}, best metric: {best_metric:.4f}')
    else:
        print('==> Building model from scratch')
        checkpoint = None
        model_spec = config['model']
        model = models.make(model_spec)
        start_epoch = 0
        best_metric = 0.0

    # Move the model to GPU for training.
    model = model.cuda()

    # Build optimizer from the YAML optimizer spec.
    optimizer_spec = config['optimizer']
    optimizer = utils.make_optimizer(model.parameters(), optimizer_spec)

    # Restore optimizer state when resuming from an interrupted run.
    if resume_path and os.path.exists(resume_path):
        if checkpoint is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('==> Optimizer state loaded')

    # Build and optionally restore the learning-rate scheduler.
    if 'lr_scheduler' in config:
        scheduler = make_scheduler(optimizer, config['lr_scheduler'])
        if resume_path and os.path.exists(resume_path) and checkpoint is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print('==> Scheduler state loaded')
    else:
        scheduler = None

    return model, optimizer, scheduler, start_epoch, best_metric


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, save_path, config, is_best=False, is_final=False):
    """Save latest, best, and/or final checkpoints."""
    model_spec = {
        'name': config['model']['name'],
        'args': config['model']['args'],
        'sd': model.state_dict(),
    }

    checkpoint = {
        'model': model_spec,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_metric': best_metric,
        'config': config,
    }

    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()

    # Always update the latest checkpoint for easy resume.
    torch.save(checkpoint, os.path.join(save_path, 'checkpoint_latest.pth'))

    # Save separate snapshots for the best validation score and the final epoch.
    if is_best:
        torch.save(checkpoint, os.path.join(save_path, 'checkpoint_best.pth'))
        print(f'==> Best model saved with metric: {best_metric:.4f}')

    if is_final:
        torch.save(checkpoint, os.path.join(save_path, 'checkpoint_final.pth'))
        print(f'==> Final model saved at epoch {epoch + 1}')


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    epoch,
    config,
    writer,
    global_step,
    scale_reg_loss=None,
    physics_ratio_loss=None,
):
    """Train the model for one epoch."""
    model.train()
    loss_avg = utils.Averager()
    reg_loss_avg = utils.Averager()       # Mean scale regularization loss.
    phy_reg_loss_avg = utils.Averager()   # Mean physics-ratio regularization loss.
    cho1_avg = utils.Averager()           # Mean x-direction Gaussian scale.
    cho3_avg = utils.Averager()           # Mean y-direction Gaussian scale.

    # T2 physics monitoring.
    rho_avg = utils.Averager()            # Mean proton-density term.
    r2_avg = utils.Averager()             # Mean T2/TE ratio term.
    delta_avg = utils.Averager()          # Mean residual magnitude.
    phy_ratio_avg = utils.Averager()      # Physics signal / total signal ratio.

    # Data normalization parameters from the YAML config.
    data_norm = config.get('data_norm', None)
    if data_norm:
        inp_sub = torch.FloatTensor(data_norm['inp']['sub']).view(1, -1, 1, 1).cuda()
        inp_div = torch.FloatTensor(data_norm['inp']['div']).view(1, -1, 1, 1).cuda()
        gt_sub = torch.FloatTensor(data_norm['gt']['sub']).view(1, -1, 1, 1).cuda()
        gt_div = torch.FloatTensor(data_norm['gt']['div']).view(1, -1, 1, 1).cuda()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["num_epochs"]}', ncols=100)

    for batch_idx, batch in enumerate(pbar):
        # Move tensor fields to GPU.
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()

        # Normalize input and target before loss computation.
        if data_norm:
            inp = (batch['inp'] - inp_sub) / inp_div
            gt = (batch['gt'] - gt_sub) / gt_div
        else:
            inp = batch['inp']
            gt = batch['gt']

        scale = batch['scale']

        # Mask is optional and is only present for mask-guided paired datasets.
        mask = batch.get('mask', None)  # [bs, 1, H_hr, W_hr] or None

        # Forward pass in full-image/patch training mode.
        pred = model(inp, scale, mask=mask)

        # Main reconstruction loss.
        loss = criterion(pred, gt)

        # T2 physics monitoring: record rho, R2, delta, and physics-ratio statistics.
        with torch.no_grad():
            if hasattr(model, 'last_rho') and model.last_rho is not None:
                rho_avg.add(model.last_rho.mean().item())
                r2_avg.add(model.last_r2.mean().item())
                delta_avg.add(model.last_delta.abs().mean().item())
                # physics_ratio estimates how much the physics prior contributes to the total signal.
                if hasattr(model, 'last_signal_physics'):
                    sp = model.last_signal_physics.abs().mean()
                    dt = model.last_delta.abs().mean()
                    ratio = (sp / (sp + dt + 1e-8)).item()
                    phy_ratio_avg.add(ratio)

        # Scale regularization encourages smaller Gaussian kernels for sharper details.
        scale_reg_weight = config.get('scale_reg_weight', 0.0)
        if scale_reg_loss is not None and scale_reg_weight > 0.0:
            if hasattr(model, 'last_para') and model.last_para is not None:
                reg_loss = scale_reg_loss(model.last_para)
                loss = loss + scale_reg_weight * reg_loss
                reg_loss_avg.add(reg_loss.item())
                # Track cho1/cho3 means to monitor whether learned scales decrease.
                with torch.no_grad():
                    cho1_avg.add(model.last_para[..., 0].mean().item())
                    cho3_avg.add(model.last_para[..., 2].mean().item())

        # Physics-ratio regularization prevents the physics contribution from collapsing.
        physics_ratio_weight = config.get('physics_ratio_weight', 0.0)
        if physics_ratio_loss is not None and physics_ratio_weight > 0.0:
            if hasattr(model, 'last_signal_physics') and model.last_signal_physics is not None:
                phy_reg = physics_ratio_loss(model.last_signal_physics, model.last_delta)
                loss = loss + physics_ratio_weight * phy_reg
                phy_reg_loss_avg.add(phy_reg.item())

        # Backpropagation.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients and warn when the norm suggests unstable optimization.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if grad_norm > 10.0:
            print(f'  [GradWarn] batch {batch_idx}: grad_norm={grad_norm:.2f} (>10, potentially unstable)')

        # Optional config-driven gradient clipping, kept here for quick experiments.
        # if config.get('grad_clip', None):
        #     nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        optimizer.step()

        # Update running statistics.
        loss_avg.add(loss.item())
        global_step[0] += 1

        # Update the progress bar with loss, regularization, and physics diagnostics.
        postfix = {'loss': f'{loss_avg.item():.4f}'}
        if reg_loss_avg.n > 0:
            postfix['reg'] = f'{reg_loss_avg.item():.4f}'
            postfix['cho1'] = f'{cho1_avg.item():.3f}'
            postfix['cho3'] = f'{cho3_avg.item():.3f}'
        if phy_reg_loss_avg.n > 0:
            postfix['phy_reg'] = f'{phy_reg_loss_avg.item():.4f}'
        if rho_avg.n > 0:
            postfix['rho'] = f'{rho_avg.item():.3f}'
            postfix['R2'] = f'{r2_avg.item():.3f}'
            postfix['|delta|'] = f'{delta_avg.item():.4f}'
            if phy_ratio_avg.n > 0:
                postfix['phy%'] = f'{phy_ratio_avg.item() * 100:.1f}'
        pbar.set_postfix(postfix)

        # Log scalar metrics to TensorBoard.
        if batch_idx % config.get('log_interval', 100) == 0:
            writer.add_scalar('train/loss', loss.item(), global_step[0])
            writer.add_scalar('train/loss_avg', loss_avg.item(), global_step[0])
            writer.add_scalar('train/grad_norm', grad_norm.item(), global_step[0])
            if reg_loss_avg.n > 0:
                writer.add_scalar('train/scale_reg_loss', reg_loss_avg.item(), global_step[0])
                writer.add_scalar('train/cho1_mean', cho1_avg.item(), global_step[0])
                writer.add_scalar('train/cho3_mean', cho3_avg.item(), global_step[0])
            if phy_reg_loss_avg.n > 0:
                writer.add_scalar('train/physics_ratio_reg_loss', phy_reg_loss_avg.item(), global_step[0])
            if rho_avg.n > 0:
                writer.add_scalar('train/rho_mean', rho_avg.item(), global_step[0])
                writer.add_scalar('train/r2_mean', r2_avg.item(), global_step[0])
                writer.add_scalar('train/delta_abs_mean', delta_avg.item(), global_step[0])
                if phy_ratio_avg.n > 0:
                    writer.add_scalar('train/physics_ratio', phy_ratio_avg.item(), global_step[0])

    # Print epoch-level scale statistics.
    if reg_loss_avg.n > 0:
        print(
            f'  [ScaleReg] reg_loss={reg_loss_avg.item():.4f} | '
            f'cho1_mean={cho1_avg.item():.4f} | cho3_mean={cho3_avg.item():.4f}'
        )

    # Print epoch-level T2 physics statistics.
    if rho_avg.n > 0:
        phy_str = f'  [T2Physics] rho={rho_avg.item():.4f} | R2={r2_avg.item():.4f} | |delta|={delta_avg.item():.4f}'
        if phy_ratio_avg.n > 0:
            phy_str += f' | phy_ratio={phy_ratio_avg.item() * 100:.1f}%'
            # Warn when the physics branch contributes too little to the prediction.
            if phy_ratio_avg.item() < 0.2 and phy_reg_loss_avg.n > 0:
                phy_str += f'  [WARNING: phy_ratio is too low! phy_reg={phy_reg_loss_avg.item():.4f}]'
            elif phy_ratio_avg.item() < 0.2:
                phy_str += '  [WARNING: phy_ratio is too low! Consider enabling physics_ratio_weight]'
        print(phy_str)

    return {
        'loss': loss_avg.item(),
        'reg_loss': reg_loss_avg.item() if reg_loss_avg.n > 0 else 0.0,
        'cho1_mean': cho1_avg.item() if cho1_avg.n > 0 else None,
        'cho3_mean': cho3_avg.item() if cho3_avg.n > 0 else None,
        'rho_mean': rho_avg.item() if rho_avg.n > 0 else None,
        'r2_mean': r2_avg.item() if r2_avg.n > 0 else None,
        'delta_abs_mean': delta_avg.item() if delta_avg.n > 0 else None,
        'physics_ratio': phy_ratio_avg.item() if phy_ratio_avg.n > 0 else None,
        'phy_reg_loss': phy_reg_loss_avg.item() if phy_reg_loss_avg.n > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description='ContinuousSR Training')
    parser.add_argument('--config', required=True, help='Path to the training YAML config.')
    parser.add_argument('--resume', default=None, help='Checkpoint path for resuming training.')
    parser.add_argument('--gpu', default='0', help='GPU device ID.')
    parser.add_argument('--name', default=None, help='Experiment name used for the save directory.')
    parser.add_argument('--tag', default=None, help='Optional experiment tag appended to the save name.')
    args = parser.parse_args()

    # Select the visible GPU before CUDA tensors are created.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load the YAML training config.
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('==> Configuration:')
    print(yaml.dump(config, default_flow_style=False))

    # Set random seeds for reproducibility.
    setup_seed(config.get('seed', 42))

    # Clear stale CUDA cache from previous runs in the same process.
    torch.cuda.empty_cache()

    # Create the experiment save directory.
    if args.name is None:
        save_name = os.path.basename(args.config).replace('.yaml', '')
    else:
        save_name = args.name

    if args.tag:
        save_name += f'_{args.tag}'

    save_path = os.path.join('./save', save_name)
    log, writer = utils.set_save_path(save_path, remove=False)

    # Initialize curve and validation-image visualization outputs.
    visualizer = TrainingVisualizer(save_path, config)

    # Save the exact config used for this run.
    with open(os.path.join(save_path, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

    print(f'==> Save path: {save_path}')

    # Build the training dataset and wrapper specified by the config.
    print('==> Building training dataset...')
    train_spec = config['train_dataset']
    train_dataset = datasets.make(train_spec['dataset'])
    train_dataset = datasets.make(train_spec['wrapper'], args={'dataset': train_dataset})
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 8),
        pin_memory=True,
    )
    print(f'Training dataset size: {len(train_dataset)}')

    # Build the optional validation dataset.
    if 'val_dataset' in config:
        print('==> Building validation dataset...')
        val_spec = config['val_dataset']
        val_dataset = datasets.make(val_spec['dataset'])
        val_dataset = datasets.make(val_spec['wrapper'], args={'dataset': val_dataset})
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_spec.get('batch_size', 1),
            shuffle=False,
            num_workers=config.get('num_workers', 8),
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
        print(f'Validation dataset size: {len(val_dataset)}')
    else:
        val_loader = None

    # Prepare model, optimizer, scheduler, and resume state.
    model, optimizer, scheduler, start_epoch, best_metric = prepare_training(config, args.resume)

    # Report model size.
    num_params = utils.compute_num_params(model, text=True)
    print(f'==> Model parameters: {num_params}')

    # Build the main reconstruction loss.
    criterion = make_loss(config['loss'])
    print(f'==> Loss function: {config["loss"]["name"]}')

    # Optional scale regularization for Gaussian kernel parameters.
    scale_reg_weight = config.get('scale_reg_weight', 0.0)
    if scale_reg_weight > 0.0:
        scale_reg_cfg = config.get('scale_reg_loss', {})
        max_scale = scale_reg_cfg.get('max_scale', 0.5)
        scale_reg_loss_fn = ScaleRegLoss(max_scale=max_scale)
        print(f'==> ScaleRegLoss enabled: weight={scale_reg_weight}, max_scale={max_scale}')
    else:
        scale_reg_loss_fn = None
        print('==> ScaleRegLoss disabled (scale_reg_weight=0.0)')

    # Optional physics-ratio regularization to keep the physics branch active.
    # Suggested starting point: physics_ratio_weight=0.1, target_ratio=0.3.
    physics_ratio_weight = config.get('physics_ratio_weight', 0.0)
    if physics_ratio_weight > 0.0:
        phy_ratio_cfg = config.get('physics_ratio_loss', {})
        target_ratio = phy_ratio_cfg.get('target_ratio', 0.3)
        physics_ratio_loss_fn = PhysicsRatioLoss(target_ratio=target_ratio)
        print(f'==> PhysicsRatioLoss enabled: weight={physics_ratio_weight}, target_ratio={target_ratio}')
    else:
        physics_ratio_loss_fn = None
        print('==> PhysicsRatioLoss disabled (physics_ratio_weight=0.0)')

    # Training loop.
    global_step = [0]  # Mutable counter shared with train_epoch.
    num_epochs = config['num_epochs']
    val_interval = config.get('val_interval', 1)
    best_ssim = 0.0      # Best observed SSIM.
    best_dists = 1.0     # Best observed DISTS; lower is better.
    best_snapshot_psnr = 0.0
    selection_metric = str(config.get('selection_metric', 'psnr')).lower()
    print(f'==> Best checkpoint selection metric: {selection_metric}')

    print('==> Start training...')
    for epoch in range(start_epoch, num_epochs):
        # Train one epoch.
        train_stats = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            epoch,
            config,
            writer,
            global_step,
            scale_reg_loss=scale_reg_loss_fn,
            physics_ratio_loss=physics_ratio_loss_fn,
        )
        train_loss = train_stats['loss']

        # Log learning rate.
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/lr', current_lr, epoch)

        # Build a compact log line with regularization and physics diagnostics.
        log_msg = f'Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss:.4f}, LR: {current_lr:.6f}'
        if train_stats['cho1_mean'] is not None:
            log_msg += (
                f' | reg_loss: {train_stats["reg_loss"]:.4f}'
                f' | cho1: {train_stats["cho1_mean"]:.4f}'
                f' | cho3: {train_stats["cho3_mean"]:.4f}'
            )
        if train_stats.get('rho_mean') is not None:
            log_msg += (
                f' | rho: {train_stats["rho_mean"]:.4f}'
                f' | R2: {train_stats["r2_mean"]:.4f}'
                f' | |delta|: {train_stats["delta_abs_mean"]:.4f}'
            )
            if train_stats.get('physics_ratio') is not None:
                log_msg += f' | phy%: {train_stats["physics_ratio"] * 100:.1f}'
            if train_stats.get('phy_reg_loss') is not None:
                log_msg += f' | phy_reg: {train_stats["phy_reg_loss"]:.4f}'
        log(log_msg)

        # Step the learning-rate scheduler after each epoch.
        if scheduler is not None:
            scheduler.step()

        # Validate and save checkpoints at the configured interval.
        if val_loader is not None and (epoch + 1) % val_interval == 0:
            print('==> Validating...')
            val_psnr, val_ssim, val_dists = validate(model, val_loader, config, compute_ssim=True, compute_dists=True)
            writer.add_scalar('val/psnr', val_psnr, epoch)
            writer.add_scalar('val/ssim', val_ssim, epoch)
            writer.add_scalar('val/dists', val_dists, epoch)
            log(f'Validation PSNR: {val_psnr:.4f} dB, SSIM: {val_ssim:.4f}, DISTS: {val_dists:.4f}')

            # Update visualization history.
            visualizer.update_metrics(epoch + 1, train_loss, val_psnr, val_ssim, val_dists, current_lr)

            # Select the best checkpoint with the configured validation metric.
            current_score, score_desc = get_selection_score(val_psnr, val_ssim, val_dists, config)
            is_best = current_score > best_metric
            if is_best:
                best_metric = current_score
                best_snapshot_psnr = val_psnr
                best_ssim = val_ssim
                best_dists = val_dists
                log(f'New best model! {score_desc}')
                log(
                    f'Best snapshot metrics -> PSNR: {best_snapshot_psnr:.4f} dB, '
                    f'SSIM: {best_ssim:.4f}, DISTS: {best_dists:.4f}'
                )

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                save_path,
                config,
                is_best,
                is_final=(epoch + 1 == num_epochs),
            )

            # Periodically generate curves, a text summary, and validation image grids.
            if (epoch + 1) % 50 == 0:
                visualizer.plot_curves(epoch + 1)
                visualizer.create_summary_report(epoch + 1)
                visualizer.visualize_results(model, val_loader, epoch + 1, num_samples=4)
        else:
            # Without validation, still record training loss and learning rate.
            visualizer.update_metrics(epoch + 1, train_loss, lr=current_lr)

            # Periodically generate training curves.
            if (epoch + 1) % 50 == 0:
                visualizer.plot_curves(epoch + 1)

            # Save only the final checkpoint when no validation set is available.
            if epoch + 1 == num_epochs:
                save_checkpoint(model, optimizer, scheduler, epoch, best_metric, save_path, config, is_best=False, is_final=True)
                print(f'==> Final model saved at epoch {epoch + 1}')

    print('==> Training completed!')
    print(f'Best selection score: {best_metric:.4f} ({selection_metric})')
    print(f'Best snapshot metrics: PSNR={best_snapshot_psnr:.4f} dB, SSIM={best_ssim:.4f}, DISTS={best_dists:.4f}')


if __name__ == '__main__':
    main()
