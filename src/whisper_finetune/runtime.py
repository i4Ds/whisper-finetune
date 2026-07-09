import os
from contextlib import nullcontext
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist


RANK = 0
LOCAL_RANK = 0
WORLD_SIZE = 1
IS_DISTRIBUTED = False
IS_MAIN = True

_wandb = None


def setup_distributed() -> torch.device:
    global RANK, LOCAL_RANK, WORLD_SIZE, IS_DISTRIBUTED, IS_MAIN

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    IS_DISTRIBUTED = "RANK" in os.environ and env_world_size > 1

    if IS_DISTRIBUTED:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP with the nccl backend requires CUDA.")

        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))

        RANK = int(os.environ["RANK"])
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(LOCAL_RANK)
    else:
        RANK = 0
        LOCAL_RANK = 0
        WORLD_SIZE = 1
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    IS_MAIN = RANK == 0

    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires CUDA.")

    return torch.device("cuda", LOCAL_RANK)


def is_main() -> bool:
    return IS_MAIN


def print_once(*args, **kwargs) -> None:
    if IS_MAIN:
        print(*args, **kwargs)


def setup_wandb(**kwargs) -> None:
    global _wandb

    if IS_MAIN:
        import wandb

        wandb.init(**kwargs)
        _wandb = wandb
    else:
        _wandb = None


def log(data: dict, step: Optional[int] = None) -> None:
    if _wandb is not None:
        _wandb.log(data, step=step)


def watch(model: torch.nn.Module, **kwargs) -> None:
    if _wandb is not None:
        _wandb.watch(unwrap_model(model), **kwargs)


def save_wandb_file(path: str) -> None:
    if _wandb is not None:
        _wandb.save(path)


def update_wandb_config(data: dict, **kwargs) -> None:
    if _wandb is not None:
        _wandb.config.update(data, **kwargs)


def set_wandb_summary(key: str, value) -> None:
    if _wandb is not None:
        _wandb.summary[key] = value


def finish_wandb() -> None:
    if _wandb is not None:
        _wandb.finish()


def barrier() -> None:
    if IS_DISTRIBUTED:
        dist.barrier(device_ids=[LOCAL_RANK])


def cleanup() -> None:
    if IS_DISTRIBUTED and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def maybe_no_sync(model: torch.nn.Module, enabled: bool):
    if enabled and hasattr(model, "no_sync"):
        return model.no_sync()
    return nullcontext()
