import random

import numpy as np
import torch
import torch.distributed as dist
import yaml
import uuid
import os


def read_config(yaml_file_path):
    with open(yaml_file_path, "r") as file:
        return yaml.safe_load(file)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def distributed_setup(rank, world_size, gpus_per_node):
    if not dist.is_available() or world_size < 2:
        print("Distributed training is not available or world size is less than 2. World size:", world_size)
        return
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)

    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    if dist.is_initialized():
        print(f"Rank {rank} initialized.")
    else:
        print(f"Rank {rank} failed to initialize.")


def get_unique_base_path():
    return os.getenv("SLURM_JOB_ID", str(uuid.uuid4()))
