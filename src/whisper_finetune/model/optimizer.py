from typing import Dict

import torch
from whisper import Whisper as WhisperModel

from whisper_finetune.utils import print_trainable_parameters


def _partition_muon_params(model: WhisperModel, ndim_threshold: int = 2):
    """
    Split Whisper parameters into Muon-eligible hidden weights and AuxAdam params.

    Muon is applied only to parameters inside the transformer block stacks
    (encoder.blocks + decoder.blocks) whose dimensionality is >= ndim_threshold.
    All remaining trainable parameters stay on auxiliary Adam:
    - gains/biases inside transformer blocks
    - embeddings
    - conv/front-end weights
    - final norms / other non-block parameters
    """
    block_param_ids = {
        id(param)
        for block in list(model.encoder.blocks) + list(model.decoder.blocks)
        for param in block.parameters()
    }

    muon_params = []
    aux_adam_params = []
    assigned_ids = set()

    for _, param in model.named_parameters():
        if not param.requires_grad or id(param) in assigned_ids:
            continue

        in_transformer_body = id(param) in block_param_ids
        if in_transformer_body and param.ndim >= ndim_threshold:
            muon_params.append(param)
        else:
            aux_adam_params.append(param)

        assigned_ids.add(id(param))

    expected_ids = {id(param) for param in model.parameters() if param.requires_grad}
    if assigned_ids != expected_ids:
        missing = len(expected_ids - assigned_ids)
        extra = len(assigned_ids - expected_ids)
        raise RuntimeError(
            "Muon parameter partition mismatch: "
            f"missing={missing}, extra={extra}. This should not happen."
        )

    return muon_params, aux_adam_params


def _use_muon_optimizer(optimizer_conf: Dict) -> bool:
    if "muon" in optimizer_conf:
        return bool(optimizer_conf["muon"])
    return optimizer_conf.get("type") == "muon"


def _muon_update_rms_match_scale(param: torch.nn.Parameter, factor: float = 0.2) -> float:
    """
    Emulate the paper's RMS-matched Muon update without editing the upstream package.

    The installed Muon implementation applies an update proportional to:
        lr * O_t * sqrt(max(1, A / B))

    The paper recommends:
        lr * 0.2 * O_t * sqrt(max(A, B))

    For the upstream implementation, the ratio between the desired scaling and the
    built-in scaling simplifies to:
        factor * sqrt(B_effective)

    where B_effective is the last dimension after Muon's internal flattening.
    We absorb this ratio into the Muon group lr, and divide weight decay by the
    same ratio so that lr * weight_decay stays unchanged.
    """
    if param.ndim < 2:
        raise ValueError("Muon RMS matching requires parameters with ndim >= 2.")

    if param.ndim == 4:
        b_effective = param[0].numel()
    else:
        b_effective = param.shape[-1]

    return float(factor) * (float(b_effective) ** 0.5)


def _build_muon_param_groups(
    muon_params,
    base_lr: float,
    base_weight_decay: float,
    momentum: float,
    match_adamw_update_rms: bool,
    match_factor: float,
):
    if not match_adamw_update_rms:
        return [
            {
                "params": muon_params,
                "use_muon": True,
                "lr": base_lr,
                "momentum": momentum,
                "weight_decay": base_weight_decay,
            }
        ]

    # Bucket by effective last dimension so parameters sharing the same RMS-match
    # scaling can stay in one Muon param group.
    grouped = {}
    for param in muon_params:
        scale = _muon_update_rms_match_scale(param, factor=match_factor)
        if scale <= 0:
            raise ValueError(f"Muon RMS match scale must be > 0, got {scale}")

        key = (param.ndim, param[0].numel() if param.ndim == 4 else param.shape[-1])
        if key not in grouped:
            grouped[key] = {
                "params": [],
                "use_muon": True,
                "lr": base_lr * scale,
                "momentum": momentum,
                "weight_decay": base_weight_decay / scale if base_weight_decay != 0 else 0.0,
            }
        grouped[key]["params"].append(param)

    return list(grouped.values())


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

    use_muon = _use_muon_optimizer(optimizer_conf)

    if use_muon:
        if optimizer_conf.get("type") not in (None, "adamw", "muon"):
            print("WARNING: optimizer.type is ignored when optimizer.muon=True. Using MuonWithAuxAdam.")

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

        muon_compatible_params, aux_adam_params = _partition_muon_params(
            model,
            ndim_threshold=ndim_threshold,
        )

        if len(muon_compatible_params) == 0:
            print(
                "WARNING: No transformer block weights matched Muon criteria "
                f"(ndim >= {ndim_threshold}). Falling back to AdamW."
            )
            optimizer = torch.optim.AdamW(parameters_to_optimize, **optimizer_conf["params"])
            return optimizer

        muon_conf = optimizer_conf.get("muon_params", {})
        adamw_conf = optimizer_conf.get("params", {})
        match_adamw_update_rms = bool(optimizer_conf.get("muon_match_adamw_update_rms", True))
        match_factor = float(optimizer_conf.get("muon_match_factor", 0.2))
        if match_factor <= 0:
            raise ValueError(f"optimizer.muon_match_factor must be > 0, got {match_factor}")

        if "amsgrad" in adamw_conf:
            print("WARNING: optimizer.params.amsgrad is not used by Muon auxiliary AdamW.")

        muon_lr = muon_conf.get("lr", 0.02)
        muon_momentum = muon_conf.get("momentum", 0.95)
        muon_weight_decay = muon_conf.get("weight_decay", adamw_conf.get("weight_decay", 0.0))

        param_groups = _build_muon_param_groups(
            muon_compatible_params,
            base_lr=muon_lr,
            base_weight_decay=muon_weight_decay,
            momentum=muon_momentum,
            match_adamw_update_rms=match_adamw_update_rms,
            match_factor=match_factor,
        )

        if len(aux_adam_params) > 0:
            param_groups.append(
                {
                    "params": aux_adam_params,
                    "use_muon": False,
                    "lr": adamw_conf.get("lr", 3e-4),
                    "betas": tuple(adamw_conf.get("betas", (0.9, 0.95))),
                    "eps": adamw_conf.get("eps", 1e-10),
                    "weight_decay": adamw_conf.get("weight_decay", 0.0),
                }
            )

        use_distributed_muon = torch.distributed.is_available() and torch.distributed.is_initialized()
        optimizer_cls = MuonWithAuxAdam if use_distributed_muon else SingleDeviceMuonWithAuxAdam
        if match_adamw_update_rms:
            print(
                "Muon RMS matching active: "
                f"factor={match_factor}, shared base_lr={muon_lr}, shared weight_decay={muon_weight_decay}"
            )
        print(
            f"Using {optimizer_cls.__name__} with "
            f"{len(muon_compatible_params)} Muon params and {len(aux_adam_params)} AuxAdamW params"
        )
        optimizer = optimizer_cls(param_groups)
    elif optimizer_conf["type"] == "adam":
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
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_conf['type']}. Must be adam or adamw.")

    return optimizer
