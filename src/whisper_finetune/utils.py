import math
import os
import random
from datetime import datetime
from socket import gethostname
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import yaml


def calculate_training_steps(config: Dict, train_dataset, world_size: int = 1, drop_last: bool = True) -> int:
    # Extract relevant values from config
    samples = len(train_dataset)
    epochs = config["training"]["epochs"]
    batch_size = config["dataset"]["batch_size"]
    accum_grad_steps = config["training"]["accum_grad_steps"]
    world_size = max(int(world_size), 1)

    # Calculate training steps
    if drop_last:
        samples_per_rank = samples // world_size
        microbatches_per_epoch = samples_per_rank // batch_size
        training_steps = math.floor((microbatches_per_epoch * epochs) / accum_grad_steps)
        return max(training_steps, 1)

    training_steps = math.ceil(samples * epochs / (batch_size * world_size * accum_grad_steps))

    return training_steps


def resolve_local_accum_grad_steps(accum_grad_steps: int, world_size: int = 1) -> int:
    """Map a configured global accumulation window to per-rank local accumulation."""
    accum_grad_steps = int(accum_grad_steps)
    world_size = max(int(world_size), 1)

    if accum_grad_steps < 1:
        raise ValueError(f"accum_grad_steps must be >= 1, got {accum_grad_steps}.")

    if accum_grad_steps % world_size != 0:
        raise ValueError(
            "training.accum_grad_steps is interpreted as the global accumulation window and must be "
            f"divisible by WORLD_SIZE. Got accum_grad_steps={accum_grad_steps} and WORLD_SIZE={world_size}."
        )

    return accum_grad_steps // world_size


def calculate_val_steps(config: Dict) -> int:
    val_steps = (config["training"]["train_steps"] / config["training"]["epochs"]) * config["training"]["eval_steps"]
    return max(int(val_steps), 1)


def read_config(yaml_file_path):
    print(f"Reading config {yaml_file_path}")
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
    return os.getenv("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))


def handle_cuda_memory_operations(config: dict) -> None:
    """
    Handles CUDA memory snapshot dumping and stops recording memory history based on the provided config.
    """
    # Construct the file name from config parameters
    file_name_elements = [
        "memory",
        str(config["model"].get("bfloat16", "NA")),
        str(config["model"].get("lora", "NA")),
        str(config["dataset"].get("batch_size", "NA")),
        str(config["training"].get("mixed_precision_training", "NA")),
        str(config["training"].get("mp_dtype", "NA")),
    ]
    file_name = "_".join(file_name_elements) + ".pt"

    # Attempt to dump CUDA memory snapshot
    try:
        torch.cuda.memory._dump_snapshot(f"memory/{file_name}")
    except Exception as e:
        print(f"Failed to dump CUDA memory snapshot: {e}")

    # Attempt to stop recording memory history
    try:
        torch.cuda.memory._record_memory_history(enabled=None)
    except Exception as e:
        # Optionally, you could log this exception if necessary.
        print(f"Failed to stop CUDA memory snapshotting: {e}")


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ", label, " \t", "Size (MB):", size / 1e6)
    os.remove("temp.p")
    return size


def print_trainable_parameters(model):
    # Filter parameters to include only those that require gradients
    parameters_to_optimize = [p for p in model.parameters() if p.requires_grad]

    # Print out the count of parameters being optimized
    num_params_to_optimize = sum(p.numel() for p in parameters_to_optimize)
    total_num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {num_params_to_optimize:,} out of total {total_num_params:,}.")


def disable_all_grads(model):
    for p in model.parameters():
        p.requires_grad = False
