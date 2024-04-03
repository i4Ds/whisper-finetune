import random
from datasets import concatenate_datasets, load_dataset
import numpy as np
import torch
import torch.distributed as dist
import yaml
import uuid
import os
from socket import gethostname


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


def add_fixed_value(batch, col_name, fixed_value):
    batch[col_name] = [fixed_value] * len(batch["text"])
    return batch


# Function to process individual datasets
def process_dataset(dataset_names, select_n_per_ds, split_name):
    """Function to process individual datasets. Mostly, we were not consistent in naming and sometimes did not add all required keys."""
    processed_datasets = []
    for N, dataset_name in zip(select_n_per_ds, dataset_names):
        dataset = load_dataset(dataset_name, split=split_name)
        if N is not None:
            # Ensure N does not exceed dataset size
            N = min(N, len(dataset))
            selected_indices = np.random.choice(len(dataset), size=N, replace=False)
            dataset = dataset.select(selected_indices)
        if "sentence" in dataset.column_names:
            dataset = dataset.rename_column("sentence", "text")

        if "language" not in dataset.column_names:  # Bad hack because we forgot to add language to the dataset.
            dataset = dataset.map(
                add_fixed_value, batched=True, fn_kwargs={"col_name": "language", "fixed_value": "de"}
            )

        processed_datasets.append(dataset)
    return concatenate_datasets(processed_datasets)
