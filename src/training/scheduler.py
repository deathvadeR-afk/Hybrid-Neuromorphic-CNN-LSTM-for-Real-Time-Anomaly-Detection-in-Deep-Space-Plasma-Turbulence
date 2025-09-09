# src/training/scheduler.py
"""Custom learning rate schedulers."""

import torch
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupScheduler(_LRScheduler):
    """Cosine learning rate scheduler with warmup."""
    
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            return [base_lr * self.last_epoch / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [base_lr * (1 + math.cos(math.pi * progress)) / 2 for base_lr in self.base_lrs]