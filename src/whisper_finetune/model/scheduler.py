import math
import random
from functools import partial
from typing import Dict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def _get_cosine_annealing_with_warmup_restarts_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: int, gamma: float
):
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return 0.0

    cycle_length = num_training_steps / num_cycles

    cycle = current_step // cycle_length
    max_lr = gamma**cycle
    step_in_cyle = current_step % cycle_length

    if step_in_cyle < num_warmup_steps:
        return float(step_in_cyle) / float(max(1, num_warmup_steps)) * max_lr
    else:
        step_in_cyle -= num_warmup_steps

    lr = max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))) * max_lr)
    return lr


def _get_cosine_annealing_with_warmup_restarts_chill_phase_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int,
    gamma: float,
    chill_steps: int,
    chill_range: float = 0.02,
):
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return 0.0

    cycle_length = num_training_steps / num_cycles

    cycle = current_step // cycle_length
    max_lr = gamma**cycle
    step_in_cyle = current_step % cycle_length

    if step_in_cyle < num_warmup_steps:
        return float(step_in_cyle) / float(max(1, num_warmup_steps)) * max_lr
    elif ((cycle_length - step_in_cyle) < chill_steps) and (cycle < num_cycles - 1):
        last_normal_progress = float((cycle_length - chill_steps + 10) - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        last_normal_lr = max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * last_normal_progress) % 1.0))) * max_lr
        )
        return last_normal_lr + random.uniform(-chill_range, chill_range)
    else:
        step_in_cyle -= num_warmup_steps

    lr = max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))) * max_lr)
    return lr


def get_cosine_annealing_with_warmup_restarts(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    last_epoch: int = -1,
    gamma: float = 1.0,
):
    lr_lambda = partial(
        _get_cosine_annealing_with_warmup_restarts_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        gamma=gamma,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_annealing_with_warmup_restarts_chill(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    last_epoch: int = -1,
    gamma: float = 1.0,
    chill_steps: int = 100,
    chill_range: float = 0.02,
):
    lr_lambda = partial(
        _get_cosine_annealing_with_warmup_restarts_chill_phase_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        gamma=gamma,
        chill_steps=chill_steps,
        chill_range=chill_range,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(optimizer: Optimizer, s_conf: dict, train_steps: int) -> LambdaLR:
    if s_conf["type"] == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=s_conf["warmup_steps"], num_training_steps=train_steps
        )
    elif s_conf["type"] == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=s_conf["warmup_steps"], num_training_steps=train_steps
        )
    elif s_conf["type"] == "cosine_with_restarts":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=s_conf["warmup_steps"],
            num_training_steps=train_steps,
            num_cycles=s_conf["lr_num_cycles"],
        )
    elif s_conf["type"] == "cosine_with_warmup_restarts":
        scheduler = get_cosine_annealing_with_warmup_restarts(
            optimizer,
            num_warmup_steps=s_conf["warmup_steps"],
            num_training_steps=train_steps,
            num_cycles=s_conf["lr_num_cycles"],
            gamma=s_conf["lr_gamma"],
        )
    elif s_conf["type"] == "cosine_with_warmup_restarts_chill":
        scheduler = get_cosine_annealing_with_warmup_restarts_chill(
            optimizer,
            num_warmup_steps=s_conf["warmup_steps"],
            num_training_steps=train_steps,
            num_cycles=s_conf["lr_num_cycles"],
            gamma=s_conf["lr_gamma"],
            chill_steps=s_conf["chill_steps"],
            chill_range=s_conf["chill_range"],
        )
    else:
        raise Exception(
            f"Unknown learning rate scheduler: {s_conf['type']}. Must be linear, cosine, cosine_with_restarts or cosine_with_warmup_restarts"
        )

    return scheduler
