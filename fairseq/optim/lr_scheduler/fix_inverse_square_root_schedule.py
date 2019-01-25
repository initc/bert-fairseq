# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('fix_inverse_sqrt')
class InverseSquareRootSchedule(FairseqLRScheduler):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`--warmup-init-lr`) until the configured
    learning rate (`--lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup:

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup:

      lr = decay_factor / sqrt(update_num)

    where

      decay_factor = args.lr * sqrt(args.warmup_updates)
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with inverse_sqrt.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        warmup_end_lr = args.lr[0]
        if args.warmup_init_lr < 0:
            args.warmup_init_lr = warmup_end_lr

        # linearly warmup for the first args.warmup_updates
        self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * args.warmup_updates**0.5

        # initial learning rate
        self.lr = args.warmup_init_lr
        self.optimizer.set_lr(self.lr)
        
        # XD
        self.high_beta1, self.low_beta1 = 0.95, 0.85
        self.beta2 = 0.99

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument('--warmup-updates', default=4000, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--cycle-updates', default=-1, type=int, metavar='N')  # XD
        parser.add_argument('--total-updates', default=-1, type=int, metavar='N')  # XD
        parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr')

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.args.warmup_updates:
            self.lr = self.args.warmup_init_lr + num_updates*self.lr_step
            if self.args.cycle_updates != -1:  # XD
                beta1 = (self.high_beta1 * (self.args.warmup_updates - num_updates) +
                    self.low_beta1 * num_updates) * 1. / self.args.warmup_updates
        else:
            if self.args.total_updates == -1:
                self.lr = self.decay_factor * num_updates**-0.5
            else:  # XD
                if num_updates < self.args.cycle_updates:
                    self.lr = (self.args.lr[0] * (self.args.cycle_updates - num_updates) +
                        self.args.warmup_init_lr * (num_updates - self.args.warmup_updates)) * 1. / (
                        self.args.cycle_updates - self.args.warmup_updates)
                    beta1 = (self.low_beta1 * (self.args.cycle_updates - num_updates) +
                        self.high_beta1 * (num_updates - self.args.warmup_updates)) * 1. / (
                        self.args.cycle_updates - self.args.warmup_updates)
                elif num_updates < self.args.total_updates:
                    self.lr = (self.args.warmup_init_lr * (self.args.total_updates - num_updates) +
                        self.args.min_lr * (num_updates - self.args.cycle_updates)) * 1. / (
                        self.args.total_updates - self.args.cycle_updates)
                    beta1 = self.high_beta1
                else:
                    self.lr = self.args.min_lr
                    beta1 = self.high_beta1
        self.optimizer.set_lr(self.lr)
        # if self.args.cycle_updates != -1:  # XD
        #     for group in self.optimizer._optimizer.param_groups:
        #         group['betas'] = (beta1, self.beta2)
        return self.lr
