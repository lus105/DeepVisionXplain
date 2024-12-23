import math
import torch
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LearningRateScheduler(_LRScheduler):
    """
    Provides inteface of learning rate scheduler.

    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']


class WarmupLRScheduler(LearningRateScheduler):
    """
    Warmup learning rate until `total_steps`

    Args:
        optimizer (Optimizer): wrapped optimizer.

    """
    def __init__(
            self,
            optimizer: Optimizer,
            init_lr: float,
            peak_lr: float,
            warmup_steps: int,
    ) -> None:
        super().__init__(optimizer, init_lr)
        self.init_lr = init_lr
        if warmup_steps != 0:
            warmup_rate = peak_lr - init_lr
            self.warmup_rate = warmup_rate / warmup_steps
        else:
            self.warmup_rate = 0
        self.update_steps = 1
        self.lr = init_lr
        self.warmup_steps = warmup_steps

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        if self.update_steps < self.warmup_steps:
            lr = self.init_lr + self.warmup_rate * self.update_steps
            self.set_lr(self.optimizer, lr)
            self.lr = lr
        self.update_steps += 1
        return self.lr


class ReduceLROnPlateauScheduler(LearningRateScheduler):
    r"""
    Reduce learning rate when a metric has stopped improving. Models often benefit from reducing the learning rate by
    a factor of 2-10 once learning stagnates. This scheduler reads a metrics quantity and if no improvement is seen
    for a ‘patience’ number of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Optimizer.
        lr (float): Initial learning rate.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
    """
    def __init__(
            self,
            optimizer: Optimizer,
            lr: float,
            patience: int = 1,
            factor: float = 0.3,
    ) -> None:
        super().__init__(optimizer, lr)
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.val_loss = 100.0
        self.count = 0

    def step(self, val_loss: float):
        if val_loss is not None:
            if self.val_loss < val_loss:
                self.count += 1
                self.val_loss = val_loss
            else:
                self.count = 0
                self.val_loss = val_loss

            if self.patience == self.count:
                self.count = 0
                self.lr *= self.factor
                self.set_lr(self.optimizer, self.lr)

        return self.lr


class TriStageLRScheduler(LearningRateScheduler):
    r"""
    Tri-Stage Learning Rate Scheduler. Implement the learning rate scheduler in "SpecAugment"

    Args:
        optimizer (Optimizer): Optimizer.
        init_lr (float): Initial learning rate.
        peak_lr (float): Maximum learning rate.
        final_lr (float): Final learning rate.
        init_lr_scale (float): Initial learning rate scale.
        final_lr_scale (float): Final learning rate scale.
        warmup_steps (int): Warmup the learning rate linearly for the first N updates.
        hold_steps (int): Hold the learning rate for the N updates.
        decay_steps (int): Decay the learning rate linearly for the first N updates.
        total_steps (int): Total steps in training.
    """
    def __init__(
            self,
            optimizer: Optimizer,
            init_lr: float,
            peak_lr: float,
            final_lr: float,
            init_lr_scale: float,
            final_lr_scale: float,
            warmup_steps: int,
            hold_steps: int,
            decay_steps: int,
            total_steps: int,
    ):
        assert isinstance(warmup_steps, int), "warmup_steps should be inteager type"
        assert isinstance(total_steps, int), "total_steps should be inteager type"

        super().__init__(optimizer, init_lr)
        self.init_lr = init_lr
        self.init_lr *= init_lr_scale
        self.final_lr = final_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.decay_steps = decay_steps

        self.warmup_rate = (self.peak_lr - self.init_lr) / self.warmup_steps if self.warmup_steps != 0 else 0
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        self.lr = self.init_lr
        self.update_steps = 0

    def _decide_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps

        offset = self.warmup_steps

        if self.update_steps < offset + self.hold_steps:
            return 1, self.update_steps - offset

        offset += self.hold_steps

        if self.update_steps <= offset + self.decay_steps:
            # decay stage
            return 2, self.update_steps - offset

        offset += self.decay_steps

        return 3, self.update_steps - offset

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)
        self.update_steps += 1

        return self.lr


class WarmupReduceLROnPlateauScheduler(LearningRateScheduler):
    r"""
    Warmup learning rate until `warmup_steps` and reduce learning rate on plateau after.

    Args:
        optimizer (Optimizer): wrapped optimizer.
        init_lr (float): Initial learning rate.
        peak_lr (float): Maximum learning rate.
        warmup_steps (int): Warmup the learning rate linearly for the first N updates.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
    """
    def __init__(
            self,
            optimizer: Optimizer,
            init_lr: float,
            peak_lr: float,
            warmup_steps: int,
            patience: int = 1,
            factor: float = 0.3,
    ) -> None:
        super().__init__(optimizer, init_lr)
        self.warmup_steps = warmup_steps
        self.update_steps = 0
        self.warmup_rate = (peak_lr - init_lr) / self.warmup_steps \
            if self.warmup_steps != 0 else 0
        self.schedulers = [
            WarmupLRScheduler(
                optimizer=optimizer,
                init_lr=init_lr,
                peak_lr=peak_lr,
                warmup_steps=warmup_steps,
            ),
            ReduceLROnPlateauScheduler(
                optimizer=optimizer,
                lr=peak_lr,
                patience=patience,
                factor=factor,
            ),
        ]

    def _decide_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps
        else:
            return 1, None

    def step(self, val_loss: Optional[float] = None):
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.schedulers[0].step()
        elif stage == 1:
            self.schedulers[1].step(val_loss)

        self.update_steps += 1

        return self.get_lr()