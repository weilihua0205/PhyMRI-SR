"""
This script evaluates our MRI Super-Resolution model trained with meta-learning.

Meta-learning is used to mitigate the domain gap between simulated datasets
(e.g., 3T→64mT) and real paired datasets (64mT–3T), improving the model's
generalization to real low-field MRI scans.
"""


import argparse
import os
import random
import re
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import datasets
import models
import utils
from evaluate import validate
from losses import PhysicsRatioLoss, ScaleRegLoss
from losses import make as make_loss
from schedulers import make_scheduler
from visualize_training import TrainingVisualizer

try:
    from torch.func import functional_call as torch_functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call as torch_functional_call


def _functional_forward(module, params, buffers, *args, **kwargs):
    merged_state = OrderedDict()
    merged_state.update(buffers)
    merged_state.update(params)
    return torch_functional_call(module, merged_state, args, kwargs)


class FunctionalLearner:
    def __init__(self, module, params, buffers, lr, first_order=False, allow_unused=True, allow_nograd=True):
        self.module = module
        self.params = OrderedDict(params)
        self.buffers = OrderedDict(buffers)
        self.lr = lr
        self.first_order = first_order
        self.allow_unused = allow_unused
        self.allow_nograd = allow_nograd

    def __call__(self, *args, **kwargs):
        return _functional_forward(self.module, self.params, self.buffers, *args, **kwargs)

    def adapt(self, loss):
        param_items = list(self.params.items())
        diff_params, diff_names = [], []
        for name, param in param_items:
            if param.requires_grad:
                diff_names.append(name)
                diff_params.append(param)
            elif not self.allow_nograd:
                raise RuntimeError(f'Parameter "{name}" does not require grad.')

        grads = torch.autograd.grad(
            loss, diff_params, retain_graph=not self.first_order,
            create_graph=not self.first_order, allow_unused=self.allow_unused,
        )
        grad_map = {name: grad for name, grad in zip(diff_names, grads)}
        updated_params = OrderedDict()
        for name, param in param_items:
            grad = grad_map.get(name)
            if grad is None:
                updated_params[name] = param
            else:
                updated_params[name] = param - self.lr * (grad.detach() if self.first_order else grad)
        self.params = updated_params


class SimpleMAML(nn.Module):
    def __init__(self, model, lr, first_order=False, allow_unused=True, allow_nograd=True):
        super().__init__()
        self.model = model
        self.lr = lr
        self.first_order = first_order
        self.allow_unused = allow_unused
        self.allow_nograd = allow_nograd

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def clone(self):
        return FunctionalLearner(
            self.model,
            OrderedDict(self.model.named_parameters()),
            OrderedDict(self.model.named_buffers()),
            lr=self.lr,
            first_order=self.first_order,
            allow_unused=self.allow_unused,
            allow_nograd=self.allow_nograd,
        )


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_selection_score(val_psnr, val_ssim=None, val_dists=None, val_hfen=None, config=None):
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
    if mode == 'hfen':
        hfen = 1e9 if val_hfen is None else float(val_hfen)
        return -hfen, f'HFEN={hfen:.4f}'
    if mode == 'composite':
        weights = config.get('selection_weights', {})
        w_psnr = float(weights.get('psnr', 0.0))
        w_ssim = float(weights.get('ssim', 1.0))
        w_dists = float(weights.get('dists', 0.0))
        w_hfen = float(weights.get('hfen', 0.0))
        psnr = float(val_psnr)
        ssim = 0.0 if val_ssim is None else float(val_ssim)
        dists = 0.0 if val_dists is None else float(val_dists)
        hfen = 0.0 if val_hfen is None else float(val_hfen)
        score = w_psnr * psnr + w_ssim * ssim - w_dists * dists - w_hfen * hfen
        desc = (f'Composite={score:.4f} '
                f'(w_psnr={w_psnr}, w_ssim={w_ssim}, w_dists={w_dists}, w_hfen={w_hfen}; '
                f'PSNR={psnr:.4f}, SSIM={ssim:.4f}, DISTS={dists:.4f}, HFEN={hfen:.4f})')
        return score, desc

    raise ValueError(f'Unsupported selection_metric: {mode}')


def prepare_meta_training(config, resume_path=None, pretrained_path=None):
    checkpoint = None
    if resume_path and os.path.exists(resume_path):
        print(f'==> Resuming from checkpoint: {resume_path}')
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
        model = models.make(checkpoint['model'], load_sd=True)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_metric = checkpoint.get('best_metric', 0.0)
    elif pretrained_path and os.path.exists(pretrained_path):
        print(f'==> Loading pretrained weights: {pretrained_path}')
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        model = models.make(checkpoint['model'], load_sd=True)
        start_epoch = 0
        best_metric = 0.0
    else:
        print('==> Building model from scratch')
        model = models.make(config['model'])
        start_epoch = 0
        best_metric = 0.0

    model = model.cuda()
    meta_model = SimpleMAML(
        model,
        lr=config.get('inner_lr', 1e-2),
        first_order=config.get('first_order', False),
        allow_unused=True,
        allow_nograd=True,
    )
    optimizer = utils.make_optimizer(meta_model.parameters(), config.get('meta_optimizer', config['optimizer']))
    if resume_path and checkpoint is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = make_scheduler(optimizer, config['lr_scheduler']) if 'lr_scheduler' in config else None
    if resume_path and checkpoint is not None and scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return model, meta_model, optimizer, scheduler, start_epoch, best_metric


def save_meta_checkpoint(model, optimizer, scheduler, epoch, best_metric, save_path, config,
                         is_best=False, is_final=False):
    model_spec = {'name': config['model']['name'], 'args': config['model']['args'], 'sd': model.state_dict()}
    checkpoint = {
        'model': model_spec,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_metric': best_metric,
        'config': config
    }
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    torch.save(checkpoint, os.path.join(save_path, 'checkpoint_latest.pth'))
    if is_best:
        torch.save(checkpoint, os.path.join(save_path, 'checkpoint_best.pth'))
    if is_final:
        torch.save(checkpoint, os.path.join(save_path, 'checkpoint_final.pth'))


def extract_patient_id(filename, patient_id_regex, patient_id_group=1):
    match = re.search(patient_id_regex, filename)
    if match is None:
        raise ValueError(f'Cannot parse patient id from filename "{filename}" using regex "{patient_id_regex}"')
    return match.group(patient_id_group)


def get_dataset_filename(dataset, idx):
    current = dataset
    while hasattr(current, 'dataset'):
        current = current.dataset
    if hasattr(current, 'lr_dataset') and hasattr(current.lr_dataset, 'filenames'):
        return current.lr_dataset.filenames[idx % len(current.lr_dataset.filenames)]
    if hasattr(current, 'lr_dataset') and hasattr(current.lr_dataset, 'files'):
        files = current.lr_dataset.files
        item = files[idx % len(files)]
        if isinstance(item, str):
            return os.path.basename(item)
    if hasattr(current, 'dataset_1') and hasattr(current.dataset_1, 'filenames'):
        return current.dataset_1.filenames[idx % len(current.dataset_1.filenames)]
    if hasattr(current, 'dataset_1') and hasattr(current.dataset_1, 'files'):
        files = current.dataset_1.files
        item = files[idx % len(files)]
        if isinstance(item, str):
            return os.path.basename(item)
    if hasattr(current, 'filenames'):
        return current.filenames[idx % len(current.filenames)]
    if hasattr(current, 'files'):
        item = current.files[idx % len(current.files)]
        if isinstance(item, str):
            return os.path.basename(item)
    raise ValueError('Unable to recover filename from dataset for patient indexing.')


def sample_indices(indices, num_samples):
    return random.sample(indices, num_samples) if len(indices) >= num_samples else random.choices(indices, k=num_samples)


def move_batch_to_cuda(batch):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
    return moved


def create_data_norm_tensors(config):
    data_norm = config.get('data_norm')
    if not data_norm:
        return None
    return {
        'inp_sub': torch.FloatTensor(data_norm['inp']['sub']).view(1, -1, 1, 1).cuda(),
        'inp_div': torch.FloatTensor(data_norm['inp']['div']).view(1, -1, 1, 1).cuda(),
        'gt_sub': torch.FloatTensor(data_norm['gt']['sub']).view(1, -1, 1, 1).cuda(),
        'gt_div': torch.FloatTensor(data_norm['gt']['div']).view(1, -1, 1, 1).cuda(),
    }


def normalize_batch(batch, data_norm):
    if not data_norm:
        return batch
    normalized = dict(batch)
    normalized['inp'] = (batch['inp'] - data_norm['inp_sub']) / data_norm['inp_div']
    normalized['gt'] = (batch['gt'] - data_norm['gt_sub']) / data_norm['gt_div']
    return normalized


def collate_samples(samples):
    collated = {}
    for key in ['inp', 'gt', 'scale', 'mask']:
        if key not in samples[0]:
            continue
        values = [sample[key] for sample in samples]
        collated[key] = torch.as_tensor(values) if key == 'scale' else torch.stack(values, dim=0)
    if 'filename' in samples[0]:
        collated['filename'] = [sample['filename'] for sample in samples]
    collated['patient_id'] = [sample['patient_id'] for sample in samples]
    return collated


def build_task_patient_index(task_dataset, patient_id_regex, patient_id_group):
    patient_to_indices = defaultdict(list)
    index_to_patient = {}
    for idx in range(len(task_dataset)):
        filename = get_dataset_filename(task_dataset, idx)
        patient_id = extract_patient_id(filename, patient_id_regex, patient_id_group)
        patient_to_indices[patient_id].append(idx)
        index_to_patient[idx] = patient_id
    if len(patient_to_indices) < 2:
        raise ValueError('Each meta-train task needs at least two distinct patients.')
    return dict(patient_to_indices), index_to_patient


def build_meta_train_tasks(config):
    task_specs = config.get('meta_train_tasks')
    if not task_specs:
        raise ValueError('config["meta_train_tasks"] is required.')
    default_regex = config.get('patient_id_regex', r'(\d+)')
    default_group = config.get('patient_id_group', 1)
    tasks = []
    for task_spec in task_specs:
        task_name = task_spec['name']
        dataset = datasets.make(task_spec['dataset'])
        dataset = datasets.make(task_spec['wrapper'], args={'dataset': dataset})
        patient_to_indices, index_to_patient = build_task_patient_index(
            dataset,
            task_spec.get('patient_id_regex', default_regex),
            task_spec.get('patient_id_group', default_group),
        )
        tasks.append({
            'name': task_name,
            'dataset': dataset,
            'patient_to_indices': patient_to_indices,
            'index_to_patient': index_to_patient,
        })
        print(f'==> Meta-task {task_name}: {len(dataset)} samples, {len(patient_to_indices)} patients')
    return tasks


def sample_episode(task_state, support_size, query_size, support_patient_num=1, query_patient_num=1):
    patient_ids = list(task_state['patient_to_indices'].keys())
    if len(patient_ids) < support_patient_num + query_patient_num:
        raise ValueError(
            f'Task {task_state["name"]} has only {len(patient_ids)} patients, '
            f'but support/query require {support_patient_num + query_patient_num}.'
        )
    support_patients = random.sample(patient_ids, support_patient_num)
    remaining_patients = [pid for pid in patient_ids if pid not in support_patients]
    query_patients = random.sample(remaining_patients, query_patient_num)

    support_pool, query_pool = [], []
    for patient_id in support_patients:
        support_pool.extend(task_state['patient_to_indices'][patient_id])
    for patient_id in query_patients:
        query_pool.extend(task_state['patient_to_indices'][patient_id])

    support_samples, query_samples = [], []
    for idx in sample_indices(support_pool, support_size):
        sample = dict(task_state['dataset'][idx])
        sample['patient_id'] = task_state['index_to_patient'][idx]
        support_samples.append(sample)
    for idx in sample_indices(query_pool, query_size):
        sample = dict(task_state['dataset'][idx])
        sample['patient_id'] = task_state['index_to_patient'][idx]
        query_samples.append(sample)

    return {
        'task_name': task_state['name'],
        'support': collate_samples(support_samples),
        'query': collate_samples(query_samples),
    }


def build_regularizers(config):
    scale_reg_loss_fn = None
    physics_ratio_loss_fn = None
    if config.get('scale_reg_weight', 0.0) > 0.0:
        scale_reg_cfg = config.get('scale_reg_loss', {})
        scale_reg_loss_fn = ScaleRegLoss(max_scale=scale_reg_cfg.get('max_scale', 0.5))
    if config.get('physics_ratio_weight', 0.0) > 0.0:
        phy_ratio_cfg = config.get('physics_ratio_loss', {})
        physics_ratio_loss_fn = PhysicsRatioLoss(target_ratio=phy_ratio_cfg.get('target_ratio', 0.3))
    return scale_reg_loss_fn, physics_ratio_loss_fn


def compute_total_loss(model_obj, pred, gt, criterion, config, scale_reg_loss=None, physics_ratio_loss=None):
    loss = criterion(pred, gt)
    aux = {'reg_loss': None, 'phy_reg_loss': None}
    if scale_reg_loss is not None and config.get('scale_reg_weight', 0.0) > 0.0:
        if hasattr(model_obj, 'last_para') and model_obj.last_para is not None:
            aux['reg_loss'] = scale_reg_loss(model_obj.last_para)
            loss = loss + config['scale_reg_weight'] * aux['reg_loss']
    if physics_ratio_loss is not None and config.get('physics_ratio_weight', 0.0) > 0.0:
        if hasattr(model_obj, 'last_signal_physics') and model_obj.last_signal_physics is not None:
            aux['phy_reg_loss'] = physics_ratio_loss(model_obj.last_signal_physics, model_obj.last_delta)
            loss = loss + config['physics_ratio_weight'] * aux['phy_reg_loss']
    return loss, aux


def train_meta_epoch(meta_model, base_model, meta_tasks, criterion, optimizer, epoch, config,
                     writer, global_step, scale_reg_loss=None, physics_ratio_loss=None):
    meta_model.train()
    base_model.train()
    loss_avg, support_loss_avg, query_loss_avg = utils.Averager(), utils.Averager(), utils.Averager()
    reg_loss_avg, phy_reg_loss_avg = utils.Averager(), utils.Averager()
    meta_batch_tasks = config.get('meta_batch_tasks', config.get('task_num', 4))
    support_size = config.get('support_size', 4)
    query_size = config.get('query_size', 4)
    num_inner_steps = config.get('num_inner_steps', 1)
    episodes_per_epoch = config.get('episodes_per_epoch', 100)
    support_patient_num = config.get('support_patient_num', 1)
    query_patient_num = config.get('query_patient_num', 1)
    data_norm = create_data_norm_tensors(config)

    pbar = tqdm(range(episodes_per_epoch), desc=f'Meta-Epoch {epoch + 1}/{config["num_epochs"]}', ncols=120)
    for episode_idx in pbar:
        sampled_tasks = random.sample(meta_tasks, meta_batch_tasks)
        optimizer.zero_grad()
        meta_loss_value = 0.0

        for task_state in sampled_tasks:
            episode = sample_episode(task_state, support_size, query_size, support_patient_num, query_patient_num)
            support_batch = normalize_batch(move_batch_to_cuda(episode['support']), data_norm)
            query_batch = normalize_batch(move_batch_to_cuda(episode['query']), data_norm)
            learner = meta_model.clone()

            for _ in range(num_inner_steps):
                support_pred = learner(support_batch['inp'], support_batch['scale'], mask=support_batch.get('mask', None))
                support_loss, support_aux = compute_total_loss(
                    learner.module, support_pred, support_batch['gt'], criterion, config, scale_reg_loss, physics_ratio_loss
                )
                learner.adapt(support_loss)
                support_loss_avg.add(support_loss.item())
                if support_aux['reg_loss'] is not None:
                    reg_loss_avg.add(support_aux['reg_loss'].item())
                if support_aux['phy_reg_loss'] is not None:
                    phy_reg_loss_avg.add(support_aux['phy_reg_loss'].item())

            query_pred = learner(query_batch['inp'], query_batch['scale'], mask=query_batch.get('mask', None))
            query_loss, query_aux = compute_total_loss(
                learner.module, query_pred, query_batch['gt'], criterion, config, scale_reg_loss, physics_ratio_loss
            )
            (query_loss / meta_batch_tasks).backward()
            meta_loss_value += query_loss.item()
            query_loss_avg.add(query_loss.item())
            if query_aux['reg_loss'] is not None:
                reg_loss_avg.add(query_aux['reg_loss'].item())
            if query_aux['phy_reg_loss'] is not None:
                phy_reg_loss_avg.add(query_aux['phy_reg_loss'].item())

        meta_loss_value = meta_loss_value / meta_batch_tasks
        loss_avg.add(meta_loss_value)
        grad_norm = None
        if config.get('grad_clip') is not None:
            grad_norm = nn.utils.clip_grad_norm_(meta_model.parameters(), config['grad_clip'])
        optimizer.step()

        postfix = {'meta_loss': f'{loss_avg.item():.4f}', 'support_loss': f'{support_loss_avg.item():.4f}', 'query_loss': f'{query_loss_avg.item():.4f}'}
        if reg_loss_avg.n > 0:
            postfix['reg'] = f'{reg_loss_avg.item():.4f}'
        if phy_reg_loss_avg.n > 0:
            postfix['phy_reg'] = f'{phy_reg_loss_avg.item():.4f}'
        pbar.set_postfix(postfix)

        if (episode_idx + 1) % config.get('log_interval', 10) == 0:
            writer.add_scalar('train/meta_loss', loss_avg.item(), global_step[0])
            writer.add_scalar('train/support_loss', support_loss_avg.item(), global_step[0])
            writer.add_scalar('train/query_loss', query_loss_avg.item(), global_step[0])
            if reg_loss_avg.n > 0:
                writer.add_scalar('train/scale_reg_loss', reg_loss_avg.item(), global_step[0])
            if phy_reg_loss_avg.n > 0:
                writer.add_scalar('train/physics_ratio_reg_loss', phy_reg_loss_avg.item(), global_step[0])
            if grad_norm is not None:
                writer.add_scalar('train/grad_norm', float(grad_norm), global_step[0])
            global_step[0] += 1

    return {
        'meta_loss': loss_avg.item(),
        'support_loss': support_loss_avg.item(),
        'query_loss': query_loss_avg.item(),
    }


def main():
    parser = argparse.ArgumentParser(description='ContinuousSR Meta-Learning Training')
    parser.add_argument('--config', required=True, help='training config path')
    parser.add_argument('--resume', default=None, help='checkpoint path')
    parser.add_argument('--pretrained', default=None, help='pretrained checkpoint path for initialization only')
    parser.add_argument('--gpu', default='0', help='GPU id')
    parser.add_argument('--name', default=None, help='experiment name')
    parser.add_argument('--tag', default=None, help='experiment tag')
    args = parser.parse_args()

    if args.resume and args.pretrained:
        raise ValueError('Use either --resume or --pretrained, not both.')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config.setdefault('meta_lr', 1e-4)
    config.setdefault('inner_lr', 1e-2)
    config.setdefault('meta_batch_tasks', config.get('task_num', 4))
    config.setdefault('support_size', 4)
    config.setdefault('query_size', 4)
    config.setdefault('support_patient_num', 1)
    config.setdefault('query_patient_num', 1)
    config.setdefault('patient_id_regex', r'(\d+)')
    config.setdefault('patient_id_group', 1)
    config.setdefault('episodes_per_epoch', 100)
    config.setdefault('num_inner_steps', 1)
    config.setdefault('first_order', False)
    config.setdefault('val_compute_dists', False)
    config.setdefault('val_compute_hfen', False)
    if 'meta_optimizer' not in config:
        config['meta_optimizer'] = {
            'name': config['optimizer']['name'],
            'args': dict(config['optimizer']['args'])
        }
        config['meta_optimizer']['args']['lr'] = config['meta_lr']

    print('==> Configuration:')
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))

    setup_seed(config.get('seed', 42))
    torch.cuda.empty_cache()

    save_name = os.path.basename(args.config).replace('.yaml', '') + '_meta' if args.name is None else args.name
    if args.tag:
        save_name += f'_{args.tag}'
    save_path = os.path.join('./save', 'meta_learning', save_name)
    log, writer = utils.set_save_path(save_path, remove=False)
    visualizer = TrainingVisualizer(save_path, config)
    with open(os.path.join(save_path, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, sort_keys=False)

    print(f'==> Save path: {save_path}')
    print('==> Building meta-train task datasets...')
    meta_tasks = build_meta_train_tasks(config)

    if 'val_dataset' in config:
        print('==> Building validation dataset...')
        val_spec = config['val_dataset']
        val_dataset = datasets.make(val_spec['dataset'])
        val_dataset = datasets.make(val_spec['wrapper'], args={'dataset': val_dataset})
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_spec.get('batch_size', 1),
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
        vis_val_loader = DataLoader(Subset(val_dataset, list(range(min(4, len(val_dataset))))), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    else:
        val_loader = None
        vis_val_loader = None

    base_model, meta_model, optimizer, scheduler, start_epoch, best_metric = prepare_meta_training(
        config, args.resume, args.pretrained
    )
    print(f'==> Model parameters: {utils.compute_num_params(base_model, text=True)}')
    criterion = make_loss(config['loss'])
    scale_reg_loss_fn, physics_ratio_loss_fn = build_regularizers(config)

    global_step = [0]
    num_epochs = config['num_epochs']
    val_interval = config.get('val_interval', 1)
    vis_interval = config.get('vis_interval', 50)
    best_ssim = 0.0
    best_dists = 1.0
    best_hfen = 1.0
    best_snapshot_psnr = 0.0
    selection_metric = str(config.get('selection_metric', 'psnr')).lower()

    print('==> Start meta-learning training...')
    for epoch in range(start_epoch, num_epochs):
        train_stats = train_meta_epoch(
            meta_model, base_model, meta_tasks, criterion, optimizer, epoch, config, writer, global_step,
            scale_reg_loss=scale_reg_loss_fn, physics_ratio_loss=physics_ratio_loss_fn,
        )
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/lr', current_lr, epoch)
        log(
            f'Meta-Epoch {epoch + 1}/{num_epochs} - Meta-Loss: {train_stats["meta_loss"]:.4f}, '
            f'Support: {train_stats["support_loss"]:.4f}, Query: {train_stats["query_loss"]:.4f}, LR: {current_lr:.6f}'
        )

        if scheduler is not None:
            scheduler.step()

        if val_loader is not None and (epoch + 1) % val_interval == 0:
            print('==> Validating...')
            compute_val_dists = config.get('val_compute_dists', False)
            compute_val_hfen = config.get('val_compute_hfen', False)
            val_metrics = validate(
                base_model,
                val_loader,
                config,
                compute_ssim=True,
                compute_dists=compute_val_dists,
                compute_hfen=compute_val_hfen,
            )
            if compute_val_dists and compute_val_hfen:
                val_psnr, val_ssim, val_dists, val_hfen = val_metrics
            elif compute_val_dists:
                val_psnr, val_ssim, val_dists = val_metrics
                val_hfen = None
            elif compute_val_hfen:
                val_psnr, val_ssim, val_hfen = val_metrics
                val_dists = None
            else:
                val_psnr, val_ssim = val_metrics
                val_dists = None
                val_hfen = None
            writer.add_scalar('val/psnr', val_psnr, epoch)
            writer.add_scalar('val/ssim', val_ssim, epoch)
            if compute_val_dists:
                writer.add_scalar('val/dists', val_dists, epoch)
            if compute_val_hfen:
                writer.add_scalar('val/hfen', val_hfen, epoch)

            metric_parts = [f'Validation PSNR: {val_psnr:.4f} dB', f'SSIM: {val_ssim:.4f}']
            if val_dists is not None:
                metric_parts.append(f'DISTS: {val_dists:.4f}')
            if val_hfen is not None:
                metric_parts.append(f'HFEN: {val_hfen:.4f}')
            log(', '.join(metric_parts))

            visualizer.update_metrics(
                epoch + 1, train_stats['meta_loss'], val_psnr, val_ssim, val_dists, current_lr, val_hfen=val_hfen
            )
            current_score, score_desc = get_selection_score(
                val_psnr, val_ssim, val_dists, val_hfen, config
            )
            is_best = current_score > best_metric
            if is_best:
                best_metric = current_score
                best_snapshot_psnr = val_psnr
                best_ssim = val_ssim
                if val_dists is not None:
                    best_dists = val_dists
                if val_hfen is not None:
                    best_hfen = val_hfen
                log(f'New best model! {score_desc}')
            save_meta_checkpoint(base_model, optimizer, scheduler, epoch, best_metric, save_path, config, is_best=is_best, is_final=(epoch + 1 == num_epochs))

            if (epoch + 1) % vis_interval == 0:
                visualizer.plot_curves(epoch + 1)
                visualizer.create_summary_report(epoch + 1)
                if vis_val_loader is not None:
                    visualizer.visualize_results(base_model, vis_val_loader, epoch + 1, num_samples=4)
        else:
            visualizer.update_metrics(epoch + 1, train_stats['meta_loss'], lr=current_lr)
            if (epoch + 1) % vis_interval == 0:
                visualizer.plot_curves(epoch + 1)
            if epoch + 1 == num_epochs:
                save_meta_checkpoint(base_model, optimizer, scheduler, epoch, best_metric, save_path, config, is_best=False, is_final=True)

    print('==> Meta-Learning Training completed!')
    summary_parts = [
        f'Best selection score: {best_metric:.4f} ({selection_metric})',
        f'Best snapshot metrics: PSNR={best_snapshot_psnr:.4f} dB',
        f'SSIM={best_ssim:.4f}',
    ]
    if config.get('val_compute_dists', False):
        summary_parts.append(f'DISTS={best_dists:.4f}')
    if config.get('val_compute_hfen', False):
        summary_parts.append(f'HFEN={best_hfen:.4f}')
    print(', '.join(summary_parts))


if __name__ == '__main__':
    main()
