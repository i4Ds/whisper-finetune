"""
Merge LoRA weights into a Whisper model and save a plain .pt checkpoint.
"""

from __future__ import annotations

import argparse
import os

import torch
import whisper

from whisper_finetune.model.lora import apply_lora, is_lora_enabled, merge_lora
from whisper_finetune.model.model_utils import save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into the base model and save a standard Whisper .pt."
    )
    parser.add_argument("--input", required=True, help="Path to input .pt checkpoint")
    parser.add_argument("--output", help="Path to output merged .pt checkpoint")
    parser.add_argument("--test_merge", action="store_true", help="Test merging before saving")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    output_path = args.output
    if output_path is None:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_merged{ext if ext else '.pt'}"

    model = whisper.load_model("large-v3", device="cpu")

    lora_config = {
        "rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
    }
    apply_lora(model, lora_config=lora_config)

    ckpt = torch.load(args.input, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("missing:", missing[:10], "..." if len(missing) > 10 else "")
        print("unexpected:", unexpected[:10], "..." if len(unexpected) > 10 else "")
        raise ValueError(
            f"State dict keys do not match model keys."
        )

    if is_lora_enabled(model):
        merge_lora(model)
        print("Merged LoRA weights into base model")
        
        if is_lora_enabled(model):
            print("Warning: LoRA parameters still detected after merge.")
    else:
        print("No LoRA parameters found; saving as-is")

    if args.test_merge:
        weights_changed = False
        for name, _ in model.named_parameters():
            if "lora" in name:
                raise ValueError(f"LoRA parameter {name} still present after merge.")
        original_model = whisper.load_model("large-v3", device="cpu")
        for (name1, param1), (_, param2) in zip(
            model.named_parameters(), original_model.named_parameters()
        ):
            if not torch.allclose(param1, param2, atol=1e-5):
                print("Parameter", name1, "differs from original model after merge; LORA did change weights.")
                weights_changed = True
                break
        if not weights_changed:
            raise ValueError("No weights changed after merge; something went wrong.")
        print("Merge test passed: weights successfully merged.")

    save_model(model, output_path)
    print(f"Saved merged checkpoint to {output_path}")


if __name__ == "__main__":
    main()
