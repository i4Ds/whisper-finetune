from typing import Dict

import torch
from whisper import Whisper as WhisperModel


def get_optimizer(model: WhisperModel, optimizer_conf: Dict):
    # Filter parameters to include only those that require gradients
    parameters_to_optimize = [p for p in model.parameters() if p.requires_grad]

    # Print out the count of parameters being optimized
    num_params_to_optimize = sum(p.numel() for p in parameters_to_optimize)
    total_num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters being optimized: {num_params_to_optimize:,}")
    print(f"Total number of parameters in the model: {total_num_params:,}")
    if optimizer_conf["type"] == "adam":
        if optimizer_conf["8bit"]:
            try:
                import bitsandbytes as bnb

                optimizer = bnb.optim.Adam8bit(parameters_to_optimize, **optimizer_conf["params"])
            except ImportError:
                raise ImportError("For using Adam 8bit optimizer you need to have bitsandbytes installed.")
        else:
            optimizer = torch.optim.Adam(parameters_to_optimize, **optimizer_conf["params"])

    elif optimizer_conf["type"] == "adamw":
        if optimizer_conf["8bit"]:
            try:
                import bitsandbytes as bnb

                optimizer = bnb.optim.AdamW8bit(parameters_to_optimize, **optimizer_conf["params"])
            except ImportError:
                raise ImportError("For using AdamW 8bit optimizer you need to have bitsandbytes installed.")
        else:
            optimizer = torch.optim.AdamW(parameters_to_optimize, **optimizer_conf["params"])

    return optimizer
