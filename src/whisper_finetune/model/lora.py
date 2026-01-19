"""
LoRA (Low-Rank Adaptation) utilities for Whisper fine-tuning.

This module provides helper functions for applying LoRA to Whisper models
using the minLoRA library.
"""

from functools import partial

import torch
from torch.nn.utils.parametrize import is_parametrized
import torch.nn.utils.parametrize as parametrize

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
    
    # Rename lora_dropout to lora_dropout_p for minlora compatibility
    minlora_lora_config = lora_config.copy()
    if "lora_dropout" in minlora_lora_config:
        minlora_lora_config["lora_dropout_p"] = minlora_lora_config.pop("lora_dropout")
    
    # Create LoRA config for minLoRA
    minlora_config = {
        WLinear: {
            "weight": partial(LoRAParametrization.from_linear, **minlora_lora_config),
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


def merge_lora(model: torch.nn.Module) -> None:  
    """Merge LoRA adapters into the base model weights."""
    def _merge_layer(layer):
        if is_parametrized(layer, "weight"):
            for attr_name in list(layer.parametrizations.keys()):
                parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=False)
    model.apply(_merge_layer)

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


def is_lora_enabled(model: torch.nn.Module) -> bool:
    """
    Check if LoRA is enabled in the model.
    
    Args:
        model: The model to check
        
    Returns:
        True if LoRA parameters are found in the model
    """
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            return True
    return False


def get_lora_debug_stats(model: torch.nn.Module, representative_module_pattern: str = "decoder.blocks.0.cross_attn.query.parametrizations.weight") -> dict:
    """
    Get LoRA debug statistics for a representative module.
    
    Collects parameter norms and gradient norms for LoRA A and B matrices.
    
    Args:
        model: The model with LoRA adapters
        representative_module_pattern: Pattern to match for a representative LoRA module
                                       (default: decoder cross-attn query projection)
    
    Returns:
        Dictionary with debug statistics:
        - lora_A_norm: ||A||_2
        - lora_B_norm: ||B||_2 
        - lora_A_grad_norm: ||A.grad||_2 (if available)
        - lora_B_grad_norm: ||B.grad||_2 (if available)
        - lora_A_grad_abs_max: max(|A.grad|) in FP32 (if available)
        - lora_B_grad_abs_max: max(|B.grad|) in FP32 (if available)
        - param_name: Name of the representative parameter found
    """
    stats = {
        "lora_A_norm": None,
        "lora_B_norm": None,
        "lora_A_grad_norm": None,
        "lora_B_grad_norm": None,
        "lora_A_grad_abs_max": None,
        "lora_B_grad_abs_max": None,
        "param_name": None,
    }
    
    found_A = False
    found_B = False
    
    for name, param in model.named_parameters():
        # Look for a representative LoRA module
        if representative_module_pattern in name or (not found_A and not found_B):
            if "lora_A" in name and not found_A:
                stats["param_name"] = name.replace(".lora_A", "")
                stats["lora_A_norm"] = param.detach().float().norm().item()
                if param.grad is not None:
                    grad_fp32 = param.grad.detach().float()
                    stats["lora_A_grad_norm"] = grad_fp32.norm().item()
                    stats["lora_A_grad_abs_max"] = grad_fp32.abs().max().item()
                found_A = True
            elif "lora_B" in name and not found_B:
                stats["lora_B_norm"] = param.detach().float().norm().item()
                if param.grad is not None:
                    grad_fp32 = param.grad.detach().float()
                    stats["lora_B_grad_norm"] = grad_fp32.norm().item()
                    stats["lora_B_grad_abs_max"] = grad_fp32.abs().max().item()
                found_B = True
        
        if found_A and found_B:
            break
    
    return stats


class LoRAUpdateTracker:
    """
    Track LoRA parameter updates across optimizer steps.
    
    Stores previous parameter values and computes ||ΔB||_2 and ||ΔA||_2
    after each optimizer step.
    """
    
    def __init__(self, model: torch.nn.Module, representative_module_pattern: str = "decoder.blocks.0.cross_attn.query.parametrizations.weight"):
        """
        Initialize the tracker.
        
        Args:
            model: The model with LoRA adapters
            representative_module_pattern: Pattern to match for tracking
        """
        self.model = model
        self.pattern = representative_module_pattern
        self.prev_A = None
        self.prev_B = None
        self.A_name = None
        self.B_name = None
        self._find_params()
    
    def _find_params(self):
        """Find the representative LoRA parameters to track."""
        for name, param in self.model.named_parameters():
            if self.pattern in name or (self.A_name is None and self.B_name is None):
                if "lora_A" in name and self.A_name is None:
                    self.A_name = name
                elif "lora_B" in name and self.B_name is None:
                    self.B_name = name
            if self.A_name is not None and self.B_name is not None:
                break
    
    def snapshot(self):
        """Take a snapshot of current parameter values."""
        for name, param in self.model.named_parameters():
            if name == self.A_name:
                self.prev_A = param.detach().clone().float()
            elif name == self.B_name:
                self.prev_B = param.detach().clone().float()
    
    def get_update_norms(self) -> dict:
        """
        Compute update norms ||ΔA||_2 and ||ΔB||_2.
        
        Call this after optimizer.step() to see how much parameters changed.
        
        Returns:
            Dictionary with delta_A_norm and delta_B_norm
        """
        result = {
            "delta_A_norm": None,
            "delta_B_norm": None,
        }
        
        for name, param in self.model.named_parameters():
            if name == self.A_name and self.prev_A is not None:
                current = param.detach().float()
                result["delta_A_norm"] = (current - self.prev_A).norm().item()
            elif name == self.B_name and self.prev_B is not None:
                current = param.detach().float()
                result["delta_B_norm"] = (current - self.prev_B).norm().item()
        
        return result


def log_lora_debug_info(model: torch.nn.Module, step: int, tracker: LoRAUpdateTracker = None, log_to_wandb: bool = True) -> dict:
    """
    Log comprehensive LoRA debug information.
    
    This should be called after loss.backward() but before optimizer.step()
    to capture gradient information.
    
    Args:
        model: The model with LoRA adapters
        step: Current training step
        tracker: Optional LoRAUpdateTracker for tracking parameter updates
        log_to_wandb: Whether to log to Weights & Biases
        
    Returns:
        Dictionary with all logged statistics
    """
    import wandb
    
    stats = get_lora_debug_stats(model)
    
    # Add update norms if tracker is provided
    if tracker is not None:
        update_norms = tracker.get_update_norms()
        stats.update(update_norms)
    
    # Log to wandb
    if log_to_wandb:
        log_dict = {}
        for key, value in stats.items():
            if value is not None and key != "param_name":
                log_dict[f"lora_debug/{key}"] = value
        if log_dict:
            wandb.log(log_dict, step=step)
    
    return stats
