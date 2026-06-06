"""
Learning-rate scheduler utilities.

This module builds PyTorch learning-rate schedulers from YAML-style config
dictionaries and provides lightweight custom warmup / polynomial schedulers.
"""

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)


def make_scheduler(optimizer, scheduler_spec):
    """
    Create a learning-rate scheduler from a config dictionary.

    Args:
        optimizer: PyTorch optimizer instance.
        scheduler_spec: Scheduler config, for example:
            {
                'name': 'MultiStepLR',
                'args': {
                    'milestones': [200, 400, 600],
                    'gamma': 0.5,
                },
            }

    Returns:
        A scheduler object, or None when scheduler_spec is None.
    """
    if scheduler_spec is None:
        return None

    scheduler_name = scheduler_spec['name']
    scheduler_args = scheduler_spec.get('args', {})

    if scheduler_name == 'StepLR':
        # Decay the learning rate every step_size epochs.
        return StepLR(
            optimizer,
            step_size=scheduler_args['step_size'],
            gamma=scheduler_args.get('gamma', 0.1),
        )

    elif scheduler_name == 'MultiStepLR':
        # Decay the learning rate at explicitly listed milestone epochs.
        return MultiStepLR(
            optimizer,
            milestones=scheduler_args['milestones'],
            gamma=scheduler_args.get('gamma', 0.1),
        )

    elif scheduler_name == 'ExponentialLR':
        # Apply exponential decay after each scheduler step.
        return ExponentialLR(
            optimizer,
            gamma=scheduler_args['gamma'],
        )

    elif scheduler_name == 'CosineAnnealingLR':
        # Use cosine annealing from the initial LR down to eta_min.
        eta_min = scheduler_args.get('eta_min', 0)
        if isinstance(eta_min, str):
            eta_min = float(eta_min)
        return CosineAnnealingLR(
            optimizer,
            T_max=scheduler_args['T_max'],
            eta_min=eta_min,
        )

    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        # Use cosine annealing with periodic warm restarts.
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_args['T_0'],
            T_mult=scheduler_args.get('T_mult', 1),
            eta_min=scheduler_args.get('eta_min', 0),
        )

    elif scheduler_name == 'ReduceLROnPlateau':
        # Reduce the LR when a monitored validation metric stops improving.
        return ReduceLROnPlateau(
            optimizer,
            mode=scheduler_args.get('mode', 'max'),
            factor=scheduler_args.get('factor', 0.1),
            patience=scheduler_args.get('patience', 10),
            threshold=scheduler_args.get('threshold', 1e-4),
            verbose=True,
        )

    elif scheduler_name == 'Warmup':
        # Linearly warm up the LR before delegating to a base scheduler.
        warmup_epochs = scheduler_args['warmup_epochs']
        base_scheduler_spec = scheduler_args['base_scheduler']
        base_scheduler = make_scheduler(optimizer, base_scheduler_spec)
        return WarmupScheduler(optimizer, warmup_epochs, base_scheduler)

    else:
        raise NotImplementedError(f'Scheduler {scheduler_name} not implemented')


class WarmupScheduler:
    """
    Scheduler wrapper with a linear warmup stage.

    The learning rate increases linearly during warmup_epochs. After warmup, the
    wrapped base scheduler controls the learning rate.
    """

    def __init__(self, optimizer, warmup_epochs, base_scheduler):
        """
        Args:
            optimizer: PyTorch optimizer instance.
            warmup_epochs: Number of warmup epochs.
            base_scheduler: Scheduler used after warmup.
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.current_epoch = 0

        # Store the initial learning rates as warmup targets.
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, metrics=None):
        """
        Advance the scheduler by one epoch.

        Args:
            metrics: Optional validation metric for ReduceLROnPlateau.
        """
        if self.current_epoch < self.warmup_epochs:
            # Warmup stage: increase LR linearly from 0 to the base LR.
            lr_scale = (self.current_epoch + 1) / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * lr_scale
        else:
            # After warmup, delegate stepping to the base scheduler.
            if isinstance(self.base_scheduler, ReduceLROnPlateau):
                if metrics is not None:
                    self.base_scheduler.step(metrics)
            else:
                self.base_scheduler.step()

        self.current_epoch += 1

    def state_dict(self):
        """Return scheduler state for checkpointing."""
        return {
            'current_epoch': self.current_epoch,
            'base_lrs': self.base_lrs,
            'base_scheduler': self.base_scheduler.state_dict() if self.base_scheduler else None,
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state from a checkpoint."""
        self.current_epoch = state_dict['current_epoch']
        self.base_lrs = state_dict['base_lrs']
        if self.base_scheduler and state_dict['base_scheduler']:
            self.base_scheduler.load_state_dict(state_dict['base_scheduler'])


class PolynomialLR:
    """Polynomial learning-rate decay scheduler."""

    def __init__(self, optimizer, max_epochs, power=0.9, min_lr=0):
        """
        Args:
            optimizer: PyTorch optimizer instance.
            max_epochs: Total number of epochs.
            power: Polynomial decay power.
            min_lr: Minimum learning rate.
        """
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.power = power
        self.min_lr = min_lr
        self.current_epoch = 0

        # Store the initial learning rates.
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        """Advance the polynomial scheduler by one epoch."""
        lr_scale = (1 - self.current_epoch / self.max_epochs) ** self.power

        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = (self.base_lrs[i] - self.min_lr) * lr_scale + self.min_lr
            param_group['lr'] = new_lr

        self.current_epoch += 1

    def state_dict(self):
        """Return scheduler state for checkpointing."""
        return {
            'current_epoch': self.current_epoch,
            'base_lrs': self.base_lrs,
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state from a checkpoint."""
        self.current_epoch = state_dict['current_epoch']
        self.base_lrs = state_dict['base_lrs']
