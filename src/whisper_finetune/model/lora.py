"""
LoRA (Low-Rank Adaptation) utilities for Whisper fine-tuning.

This module provides helper functions for applying LoRA to Whisper models
using the minLoRA library.
"""

from functools import partial

import torch
from torch.nn.utils.parametrize import is_parametrized


def disable_all_but_parametrized_grads(model: torch.nn.Module) -> None:
    """
    Disable gradients for all parameters except LoRA parameters.
    
    This allows training only the LoRA adapters while keeping the base model frozen.
    LoRA parameters are identified by having "lora" in their name.
    
    Args:
        model: The model to modify
    """
    for name, param in model.named_parameters():
        # Keep LoRA parameters trainable, freeze everything else
        if "lora" not in name.lower():
            param.requires_grad = False


def apply_lora(
    model: torch.nn.Module,
    lora_config: dict,
    train_only_decoder: bool = False,
    train_only_encoder: bool = False,
) -> None:
    """
    Apply LoRA adapters to a Whisper model.
    
    Args:
        model: The Whisper model to modify
        lora_config: Configuration dict with LoRA parameters (rank, lora_alpha, lora_dropout)
        train_only_decoder: If True, only apply LoRA to the decoder
        train_only_encoder: If True, only apply LoRA to the encoder
    """
    from minlora import LoRAParametrization, add_lora
    from whisper.model import Linear as WLinear
    
    # Create LoRA config for minLoRA
    minlora_config = {
        WLinear: {
            "weight": partial(LoRAParametrization.from_linear, **lora_config),
        },
    }
    
    if train_only_decoder:
        # Apply LoRA only to decoder
        add_lora(model.decoder, lora_config=minlora_config)
    elif train_only_encoder:
        # Apply LoRA only to encoder
        add_lora(model.encoder, lora_config=minlora_config)
    else:
        # Apply LoRA to entire model
        add_lora(model, lora_config=minlora_config)
    
    # Disable gradients for non-LoRA parameters
    disable_all_but_parametrized_grads(model)


def print_lora_info(model: torch.nn.Module) -> None:
    """
    Print information about LoRA parameters in the model.
    
    Args:
        model: The model with LoRA adapters
    """
    lora_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            lora_params += param.numel()
    
    print(f"LoRA trainable parameters: {lora_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * lora_params / total_params:.4f}%")
