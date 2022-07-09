# Copyright (c) Facebook, Inc. and its affiliates.
import math
import logging
import torch
from fvcore.common.param_scheduler import CosineParamScheduler, MultiStepParamScheduler
from detectron2.config import CfgNode
from detectron2.solver import LRMultiplier, WarmupParamScheduler

class MyCosineParamScheduler(CosineParamScheduler):
    """
    Cosine decay or cosine warmup schedules based on start and end values.
    The schedule is updated based on the fraction of training progress.
    The schedule was proposed in 'SGDR: Stochastic Gradient Descent with
    Warm Restarts' (https://arxiv.org/abs/1608.03983). Note that this class
    only implements the cosine annealing part of SGDR, and not the restarts.
    Example:
        .. code-block:: python
          CosineParamScheduler(start_value=0.1, end_value=0.0001)
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
        T_max: int,
        max_iter: int,
        last_epoch: int = 0,
    ) -> None:
        self._start_value = start_value
        self._end_value = end_value
        self.T_max = T_max
        self.max_iter = max_iter
        self.last_epoch = last_epoch

    def __call__(self, where: float) -> float:
        t = where * self.max_iter + self.last_epoch
        return self._end_value + 0.5 * (self._start_value - self._end_value) * (
            1 + math.cos(math.pi * (t / self.T_max))
        )

def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME

    if name == "WarmupMultiStepLR":
        steps = [x for x in cfg.SOLVER.STEPS if x <= cfg.SOLVER.MAX_ITER]
        if len(steps) != len(cfg.SOLVER.STEPS):
            logger = logging.getLogger(__name__)
            logger.warning(
                "SOLVER.STEPS contains values larger than SOLVER.MAX_ITER. "
                "These values will be ignored."
            )
        sched = MultiStepParamScheduler(
            values=[cfg.SOLVER.GAMMA ** k for k in range(len(steps) + 1)],
            milestones=steps,
            num_updates=cfg.SOLVER.MAX_ITER,
        )
    elif name == "WarmupCosineLR":
        sched = MyCosineParamScheduler(1., 1e-3, T_max=10000, max_iter=cfg.SOLVER.MAX_ITER, last_epoch=0)
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))

    sched = WarmupParamScheduler(
        sched,
        cfg.SOLVER.WARMUP_FACTOR,
        min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
        cfg.SOLVER.WARMUP_METHOD,
    )
    return LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)
