from whisper.model import MultiHeadAttention
import loralib as lora
import torch
import torch.nn as nn


def replace_attention_layers_with_lora(module, config, parent=None, parent_name=None):
    for name, child in module.named_children():
        # Recursive call for child modules
        replace_attention_layers_with_lora(child, config, parent=module, parent_name=name)

    if isinstance(module, MultiHeadAttention):
        # Replace the specific layers if they match the target names
        if hasattr(module, "query") and "query" in config["target_modules"]:
            setattr(module, "query", lora.Linear(module.query.in_features, module.query.out_features, r=config["r"]))
        if hasattr(module, "key") and "key" in config["target_modules"]:
            setattr(
                module, "key", lora.Linear(module.key.in_features, module.key.out_features, r=config["r"], bias=False)
            )
        if hasattr(module, "value") and "value" in config["target_modules"]:
            setattr(module, "value", lora.Linear(module.value.in_features, module.value.out_features, r=config["r"]))


def print_trainable_params(model: nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Out of {total_params:,} parameters, {trainable_params:,} are trainable, a reduction of {round((1-trainable_params/total_params)*100, 1)} % "
    )


def mark_only_lora_as_trainable(model, bias="none"):
    lora.mark_only_lora_as_trainable(model, bias=bias)


def save_lora_model(model, path, bias="none"):
    torch.save(lora.lora_state_dict(model, bias=bias), path)


def load_lora_model(model, pretrained_path, lora_path, strict=False):
    model.load_state_dict(torch.load(pretrained_path), strict=strict)
    model.load_state_dict(torch.load(lora_path), strict=strict)


def print_model_layers(model, indent=0):
    """
    Recursively prints out the model's layers and their types.
    """
    for name, module in model.named_children():
        print(" " * indent + f"{name}: {type(module).__name__}")
        print_model_layers(module, indent + 2)


def has_lora_layers(model):
    """
    Check if the model has any LoRA layers.1
    """
    for n, _ in model.named_parameters():
        if "lora_" in n:
            return True
    return False
