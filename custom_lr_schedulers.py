from torch.optim import lr_scheduler
import math

class ClampedStepLR(lr_scheduler.StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, maximum_lr=-1):
        self.maximum_lr = maximum_lr
        super(ClampedStepLR, self).__init__(optimizer, step_size, gamma, last_epoch)

    def step():
        if((self.maximum_lr > 0 and get_lr()[0]<self.maximum_lr) or self.maximum_lr < 0):
            super().step()


class CosineAnnealingRestartsLR(lr_scheduler._LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule with warm restarts, where :math:`\eta_{max}` is set to the
    initial learning rate, :math:`T_{cur}` is the number of epochs since the
    last restart and :math:`T_i` is the number of epochs in :math:`i`-th run
    (after performing :math:`i` restarts):
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2} \eta_{mult}^i (\eta_{max}-\eta_{min})
        (1 + \cos(\frac{T_{cur}}{T_i - 1}\pi))
        T_i = T T_{mult}^i
    When last_epoch=-1, sets initial lr as lr.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that in the
    paper the :math:`i`-th run takes :math:`T_i + 1` epochs, while in this
    implementation it takes :math:`T_i` epochs only. This implementation
    also enables updating the range of learning rates by multiplicative factor
    :math:`\eta_{mult}` after each restart.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T (int): Length of the initial run (in number of epochs).
        eta_min (float): Minimum learning rate. Default: 0.
        T_mult (float): Multiplicative factor adjusting number of epochs in
            the next run that is applied after each restart. Default: 2.
        eta_mult (float): Multiplicative factor of decay in the range of
            learning rates that is applied after each restart. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T, eta_min=0, T_mult=2.0, eta_mult=1.0, last_epoch=-1):
        self.T = T
        self.eta_min = eta_min
        self.eta_mult = eta_mult

        if T_mult < 1:
            raise ValueError('T_mult should be >= 1.0.')
        self.T_mult = T_mult

        super(CosineAnnealingRestartsLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_mult == 1:
            i_restarts = self.last_epoch // self.T
            last_restart = i_restarts * self.T
            T_i = self.T

        else:
            # computation of the last restarting epoch is based on sum of geometric series:
            # last_restart = T * (1 + T_mult + T_mult ** 2 + ... + T_mult ** i_restarts)
            i_restarts = math.floor(math.log(1 - self.last_epoch * (1 - self.T_mult) / self.T,
                                             self.T_mult))
            last_restart = self.T * (1 - self.T_mult ** i_restarts) / (1 - self.T_mult)
            T_i = self.T * self.T_mult ** i_restarts

        t = (self.last_epoch - last_restart) / T_i
        decay = 0.5 * (self.eta_mult ** i_restarts) * (1 + math.cos(math.pi * t))

        return [decay * base_lr + (1 - decay) * self.eta_min for base_lr in self.base_lrs]

