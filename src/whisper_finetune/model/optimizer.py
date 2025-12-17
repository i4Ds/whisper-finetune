from typing import Dict

import torch
from whisper import Whisper as WhisperModel

from whisper_finetune.utils import print_trainable_parameters


def get_optimizer(model: WhisperModel, optimizer_conf: Dict, is_lora_run: bool = False):
    """
    Create an optimizer for model training.
    
    Args:
        model: The Whisper model to optimize
        optimizer_conf: Optimizer configuration dictionary
        is_lora_run: Whether this is a LoRA training run (affects optimizer choice warnings)
        
    Returns:
        The configured optimizer
        
    Note on precision:
        - Standard PyTorch optimizers (Adam, AdamW) maintain FP32 states by default.
        - This is crucial for LoRA training where gradients can be small.
        - 8-bit optimizers can be used but may cause issues with very small gradients.
        - With AMP, gradients are accumulated in FP32 regardless of forward pass precision.
    """
    # Filter parameters to include only those that require gradients
    parameters_to_optimize = [p for p in model.parameters() if p.requires_grad]

    print("---OPTIMIZER----")
    print_trainable_parameters(model)
    
    # Warn about 8-bit optimizer with LoRA
    if optimizer_conf["8bit"] and is_lora_run:
        print("WARNING: Using 8-bit optimizer with LoRA training.")
        print("If you observe training instability or zero gradients, try setting optimizer.8bit=False")
        print("8-bit optimizers may quantize small gradient values to zero.")

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
