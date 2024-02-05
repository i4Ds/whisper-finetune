import random

import numpy as np
import torch
import torch.distributed as dist
import yaml


def read_config(yaml_file_path):
    with open(yaml_file_path, "r") as file:
        return yaml.safe_load(file)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def distributed_setup(rank, world_size):
    if not dist.is_available() or world_size < 2:
        print("Distributed training is not available or world size is less than 2. World Size: ", world_size)
        return False
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if dist.is_initialized():
        print(f"Rank {rank} initialized.")
        return True
    else:
        print(f"Rank {rank} failed to initialize.")
        return False
