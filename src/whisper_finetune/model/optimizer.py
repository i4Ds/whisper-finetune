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
    elif optimizer_conf["type"] == "muon":
        if optimizer_conf.get("8bit", False):
            print("WARNING: optimizer.8bit=True is ignored for Muon.")

        try:
            from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
        except ImportError as exc:
            raise ImportError(
                "Muon optimizer requested, but package 'muon' is not installed in the active environment."
            ) from exc

        ndim_threshold = int(optimizer_conf.get("muon_ndim_threshold", 2))
        if ndim_threshold < 1:
            raise ValueError(f"optimizer.muon_ndim_threshold must be >= 1, got {ndim_threshold}")

        muon_compatible_params = [p for p in parameters_to_optimize if p.ndim >= ndim_threshold]
        aux_adam_params = [p for p in parameters_to_optimize if p.ndim < ndim_threshold]

        if len(muon_compatible_params) == 0:
            print(
                "WARNING: No parameters matched Muon criteria "
                f"(ndim >= {ndim_threshold}). Falling back to AdamW."
            )
            optimizer = torch.optim.AdamW(parameters_to_optimize, **optimizer_conf["params"])
            return optimizer

        muon_conf = optimizer_conf.get("muon_params", {})
        adam_conf = optimizer_conf.get("params", {})

        if "amsgrad" in adam_conf:
            print("WARNING: optimizer.params.amsgrad is not used by Muon auxiliary Adam.")

        param_groups = [
            {
                "params": muon_compatible_params,
                "use_muon": True,
                "lr": muon_conf.get("lr", 0.02),
                "momentum": muon_conf.get("momentum", 0.95),
                "weight_decay": muon_conf.get("weight_decay", adam_conf.get("weight_decay", 0.0)),
            }
        ]

        if len(aux_adam_params) > 0:
            param_groups.append(
                {
                    "params": aux_adam_params,
                    "use_muon": False,
                    "lr": adam_conf.get("lr", 3e-4),
                    "betas": tuple(adam_conf.get("betas", (0.9, 0.95))),
                    "eps": adam_conf.get("eps", 1e-10),
                    "weight_decay": adam_conf.get("weight_decay", 0.0),
                }
            )

        use_distributed_muon = torch.distributed.is_available() and torch.distributed.is_initialized()
        optimizer_cls = MuonWithAuxAdam if use_distributed_muon else SingleDeviceMuonWithAuxAdam
        print(
            f"Using {optimizer_cls.__name__} with "
            f"{len(muon_compatible_params)} Muon params and {len(aux_adam_params)} AuxAdam params"
        )
        optimizer = optimizer_cls(param_groups)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_conf['type']}. Must be adam, adamw, or muon.")

    return optimizer
