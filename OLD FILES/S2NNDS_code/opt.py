from common_header import *

# Custom Warm-Up Scheduler
class WarmUpLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, initial_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.initial_lr * (self.last_epoch + 1) / self.warmup_steps for _ in self.base_lrs]
        return [base_lr for base_lr in self.base_lrs]
