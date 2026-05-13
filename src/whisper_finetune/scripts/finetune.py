import argparse
import json
import os
import subprocess
import warnings
from pathlib import Path
from pprint import pprint

import torch
import wandb
import whisper
from torch.utils.data import DataLoader
from tqdm import tqdm
from whisper import Whisper as WhisperModel
from whisper.tokenizer import get_tokenizer

from whisper_finetune.data.data_loader import get_dataloader, WarmupDatasetSampler, get_dataset_boundary_indices
from whisper_finetune.data.utils import process_dataset
from whisper_finetune.eval.evaluator import (
    evaluate_multiple_datasets,
    log_metrics_to_wandb,
)
from whisper_finetune.model.model_utils import (
    CheckpointedStochasticAudioEncoder,
    CheckpointedStochasticTextDecoder,
    infinite_iter,
    resize_whisper_layers,
    register_deep_spec_augment_hooks,
    save_model,
    train_step,
)
from whisper_finetune.model.lora import apply_lora, print_lora_info
from whisper_finetune.model.optimizer import get_optimizer
from whisper_finetune.model.scheduler import get_scheduler
from whisper_finetune.utils import (
    calculate_training_steps,
    calculate_val_steps,
    disable_all_grads,
    get_unique_base_path,
    handle_cuda_memory_operations,
    print_trainable_parameters,
    read_config,
    set_seed,
)

ENABLE_MEMORY_PROFILING = False

MODEL_LAYER_PRESETS = {
    "whisper-4832": {"base_init_name": "large-v3", "encoder_layers": 48, "decoder_layers": 32},
    "whisper-3248": {"base_init_name": "large-v3", "encoder_layers": 32, "decoder_layers": 48},
}


def _first_defined(config: dict, *keys: str):
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return None


def _resolve_model_architecture(model_config: dict) -> tuple[str, int | None, int | None]:
    init_name = model_config["init_name"]
    preset = MODEL_LAYER_PRESETS.get(init_name, {})

    base_init_name = model_config.get("base_init_name", preset.get("base_init_name", init_name))
    encoder_layers = _first_defined(model_config, "encoder_layers", "encoder_layer")
    decoder_layers = _first_defined(model_config, "decoder_layers", "decoder_layer", "deocer_layer")

    if encoder_layers is None:
        encoder_layers = preset.get("encoder_layers")
    if decoder_layers is None:
        decoder_layers = preset.get("decoder_layers")

    if encoder_layers is not None:
        encoder_layers = int(encoder_layers)
    if decoder_layers is not None:
        decoder_layers = int(decoder_layers)

    return base_init_name, encoder_layers, decoder_layers


def _pad_list_with_none(values, target_len, label):
    padded_values = list(values)
    if len(padded_values) < target_len:
        missing = target_len - len(padded_values)
        warnings.warn(
            f"{label} has {len(padded_values)} entries for {target_len} validation datasets; appending {missing} None value(s) to avoid dropping data in zip().",
            stacklevel=2,
        )
        padded_values.extend([None] * missing)
    return padded_values


def main_loop(
    model: WhisperModel,
    train_loader: DataLoader,
    dev_loaders: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    save_dir: str,
    t_config: dict,
) -> None:
    """
    Main loop function that iterates through training steps, evaluates the model, logs training progress, and saves models.

    Parameters:
        model (WhisperModel): The Whisper model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        dev_loaders (dict): Dictionary of DataLoaders for validation datasets (dataset_name -> DataLoader).
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        scheduler (torch.optim.lr_scheduler.LambdaLR): Learning rate scheduler.
        save_dir (str): Directory to save the trained model.
        t_config (dict): Configuration settings for training.

    Returns:
        None
    """
    wandb.watch(model, log="all")

    # Setup LoRA update tracker if this is a LoRA run
    lora_tracker = None
    if t_config.get("is_lora_run", False):
        from whisper_finetune.model.lora import LoRAUpdateTracker
        lora_tracker = LoRAUpdateTracker(model)
        print("LoRA debug logging enabled - tracking parameter norms, gradient norms, and updates")

    # Initial evaluation with new multi-dataset evaluator
    print("\nRunning initial evaluation...")
    dataset_metrics, macro_metrics = evaluate_multiple_datasets(model, dev_loaders, t_config)
    min_wer = macro_metrics["macro_wer"]

    print(f"Initial Macro WER: {min_wer:.4f}")
    log_metrics_to_wandb(dataset_metrics, macro_metrics, step=0, prefix="val")

    pbar = tqdm(range(1, t_config["train_steps"] + 1))
    train_iter = infinite_iter(train_loader)
    for step in pbar:
        # Pass step so train_step can log LoRA debug info at eval steps
        train_loss = train_step(model, train_iter, optimizer, scheduler, t_config, lora_tracker=lora_tracker, step=step)
        pbar.set_postfix({"loss": train_loss})
        # Use consistent step= for all wandb logs to avoid step mismatch warnings
        wandb.log(_build_lr_log_dict(optimizer, scheduler, train_loss), step=step)
        assert (
            train_loss < t_config["max_train_loss"]
        ), f"Train loss is above {t_config['max_train_loss']}, the loss is unable to converge."

        if (step % t_config["val_steps"]) == 0 or step == t_config["train_steps"]:
            # Note: LoRA debug info is logged in train_step at eval steps (captures gradients before optimizer.step)
            
            # Evaluate on all validation datasets
            dataset_metrics, macro_metrics = evaluate_multiple_datasets(model, dev_loaders, t_config)
            eval_wer = macro_metrics["macro_wer"]

            tqdm.write(f"Step {step}: Macro WER={eval_wer:.4f}")
            log_metrics_to_wandb(dataset_metrics, macro_metrics, step=step, prefix="val")

            # Save best model based on macro WER
            if eval_wer < min_wer:
                min_wer = eval_wer
                save_model(model, f"{save_dir}/best_model.pt")
                print(f"  → Saved new best model (WER: {min_wer:.4f})")

            # Always save checkpoint locally (but don't upload to wandb yet)
            if t_config["save_all_checkpoints"]:
                save_model(model, f"{save_dir}/step{step}.pt")

    save_model(model, f"{save_dir}/last_model.pt")

    if t_config.get('upload_models_to_wandb', False):
        # Only upload models to wandb if they are different
        import filecmp

        last_model_path = f"{save_dir}/last_model.pt"
        best_model_path = f"{save_dir}/best_model.pt"

        if os.path.exists(best_model_path) and filecmp.cmp(last_model_path, best_model_path, shallow=False):
            print("Last model and best model are identical. Uploading only best_model.pt to wandb.")
            wandb.save(best_model_path)
        else:
            print("Uploading both last_model.pt and best_model.pt to wandb.")
            wandb.save(last_model_path)
            if os.path.exists(best_model_path):
                wandb.save(best_model_path)


def _build_lr_log_dict(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    train_loss: float,
) -> dict:
    """Log all LR groups so multi-group optimizers like Muon are not misrepresented."""
    current_lrs = scheduler.get_last_lr()
    log_data = {"Train loss": train_loss}

    if len(current_lrs) == 1:
        log_data["Learning rate"] = current_lrs[0]
        return log_data

    log_data["Learning rate/min"] = min(current_lrs)
    log_data["Learning rate/max"] = max(current_lrs)
    log_data["Learning rate/mean"] = sum(current_lrs) / len(current_lrs)

    schedule_factors = []
    grouped_lrs: dict[str, list[float]] = {}
    grouped_base_lrs: dict[str, list[float]] = {}
    lr_group_metadata = getattr(optimizer, "_lr_group_metadata", [])
    for idx, (group, lr) in enumerate(zip(optimizer.param_groups, current_lrs)):
        metadata = lr_group_metadata[idx] if idx < len(lr_group_metadata) else {}
        group_label = str(metadata.get("lr_log_label") or ("muon" if group.get("use_muon") else "aux_adamw"))
        grouped_lrs.setdefault(group_label, []).append(lr)

        base_lr_unscaled = metadata.get("base_lr_unscaled")
        if base_lr_unscaled is not None:
            grouped_base_lrs.setdefault(group_label, []).append(base_lr_unscaled)

        group_name = f"{group_label}_group_{idx}"
        safe_group_name = str(group_name).replace("/", "_")
        log_data[f"Learning rate/{safe_group_name}"] = lr

        initial_lr = group.get("initial_lr")
        if initial_lr is not None and initial_lr != 0:
            schedule_factors.append(lr / initial_lr)

    if schedule_factors:
        shared_schedule_factor = sum(schedule_factors) / len(schedule_factors)
        log_data["Learning rate/schedule_factor"] = shared_schedule_factor
    else:
        shared_schedule_factor = None

    if "muon" in grouped_lrs:
        muon_actual_lrs = grouped_lrs["muon"]
        log_data["Learning rate/muon_actual_min"] = min(muon_actual_lrs)
        log_data["Learning rate/muon_actual_max"] = max(muon_actual_lrs)
        log_data["Learning rate/muon_actual_mean"] = sum(muon_actual_lrs) / len(muon_actual_lrs)

        muon_base_lrs = grouped_base_lrs.get("muon", [])
        if muon_base_lrs:
            muon_base_lr = sum(muon_base_lrs) / len(muon_base_lrs)
            log_data["Learning rate/muon"] = (
                muon_base_lr * shared_schedule_factor if shared_schedule_factor is not None else muon_base_lr
            )

    if "aux_adamw" in grouped_lrs:
        aux_actual_lrs = grouped_lrs["aux_adamw"]
        log_data["Learning rate/aux_adamw_actual"] = sum(aux_actual_lrs) / len(aux_actual_lrs)

        aux_base_lrs = grouped_base_lrs.get("aux_adamw", [])
        if aux_base_lrs:
            aux_base_lr = sum(aux_base_lrs) / len(aux_base_lrs)
            log_data["Learning rate/aux_adamw"] = (
                aux_base_lr * shared_schedule_factor if shared_schedule_factor is not None else aux_base_lr
            )

    if "Learning rate/muon" in log_data:
        log_data["Learning rate"] = log_data["Learning rate/muon"]
    elif "Learning rate/aux_adamw" in log_data:
        log_data["Learning rate"] = log_data["Learning rate/aux_adamw"]
    else:
        log_data["Learning rate"] = current_lrs[0]

    return log_data


def main(config):
    """
    Runs the main training loop for a Whisper model.

    Args:
        config (dict): A dictionary containing the configuration for the training.

    Returns:
        None

    Raises:
        NotImplementedError: If gradient checkpointing is enabled for the encoder and the 'gradient_checkpointing_encoder_last_only' flag is set.

    """
    # Start GPU memory profiling
    torch.cuda.reset_peak_memory_stats("cuda")
    if ENABLE_MEMORY_PROFILING:
        torch.cuda.memory._record_memory_history()

    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    config["save_dir"] = os.path.join(config["save_dir"], get_unique_base_path())

    # Create save directory
    Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)

    if config["model"].get("lora", False):
        lora_config = config["model"].get("lora_config", {})
        lora_config_path = os.path.join(config["save_dir"], "lora_config.json")
        with open(lora_config_path, "w", encoding="utf-8") as handle:
            json.dump(lora_config, handle, indent=2, sort_keys=True)

    # Print SLURM stuff
    # Check if the script is running on a Slurm cluster
    if "SLURM_JOB_ID" in os.environ:
        # Get the current node name
        node_name = os.environ["SLURMD_NODENAME"]
        print(f"Current Node: {node_name}")

        # Fetch general stats about the node using the scontrol command
        stats = subprocess.check_output(["scontrol", "show", "node", node_name]).decode("utf-8")
        print("General Stats:")
        print(stats)

    # Print CUDA version, PyTorch version, and GPU name
    print("CUDA version:", torch.version.cuda)
    print("PyTorch version:", torch.__version__)
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")

    base_init_name, target_encoder_layers, target_decoder_layers = _resolve_model_architecture(config["model"])
    if base_init_name != config["model"]["init_name"]:
        print(f"Model alias '{config['model']['init_name']}' resolved to base model '{base_init_name}'.")

    ## Get model
    whisper_model = whisper.load_model(base_init_name, device="cpu")


    # NOTE ON PRECISION:
    # DO NOT manually cast the model to bfloat16/half!
    # With AMP (automatic mixed precision), model weights stay in FP32 while activations
    # are computed in the specified dtype (fp16 or bf16). This is crucial for LoRA training
    # where optimizer states and gradients should remain in FP32.
    # 
    # If config["model"]["bfloat16"] is True, we print a deprecation warning.
    # The correct way to use bf16 is via mixed_precision_training=True and mp_dtype="bf16"
    if config["model"]["bfloat16"]:
        print("WARNING: config['model']['bfloat16'] is deprecated and will be ignored!")
        print("For bf16 training, use: training.mixed_precision_training=True and training.mp_dtype='bf16'")
        print("Model weights are kept in FP32 and autocast handles precision during forward pass.")
        print("This is REQUIRED for proper LoRA training where optimizer states must be FP32.")
    
    whisper_model.is_bfloat = False  # Always False now - AMP handles precision

    # If gradient checkpointing is enabled, wrap the model with checkpointing
    # Important: When doing encoder/decoder-only training, stochastic depth should only
    # be applied to the part being trained. Set stochastic_depth=0.0 for frozen components.
    encoder_stochastic_depth = 0.0 if config["training"]["train_only_decoder"] else config["training"]["stochastic_depth"]
    decoder_stochastic_depth = 0.0 if config["training"]["train_only_encoder"] else config["training"]["stochastic_depth"]
    use_gradient_checkpointing_encoder = config["training"]["gradient_checkpointing_encoder"]
    use_gradient_checkpointing_decoder = config["training"]["gradient_checkpointing_decoder"]
    checkpoint_reload_state = (
        whisper_model.state_dict()
        if (use_gradient_checkpointing_encoder or use_gradient_checkpointing_decoder)
        else None
    )

    if use_gradient_checkpointing_encoder:
        del whisper_model.encoder
        if config["training"]["gradient_checkpointing_encoder_last_only"]:
            raise ValueError(
                "gradient_checkpointing_encoder_last_only is not supported when gradient_checkpointing_encoder is enabled"
            )
        else:
            whisper_model.encoder = CheckpointedStochasticAudioEncoder(
                whisper_model.dims.n_mels,
                whisper_model.dims.n_audio_ctx,
                whisper_model.dims.n_audio_state,
                whisper_model.dims.n_audio_head,
                whisper_model.dims.n_audio_layer,
                encoder_stochastic_depth,
            )
    if use_gradient_checkpointing_decoder:
        del whisper_model.decoder
        whisper_model.decoder = CheckpointedStochasticTextDecoder(
            whisper_model.dims.n_vocab,
            whisper_model.dims.n_text_ctx,
            whisper_model.dims.n_text_state,
            whisper_model.dims.n_text_head,
            whisper_model.dims.n_text_layer,
            decoder_stochastic_depth,
        )

    # We need to reload weights after replacing encoder/decoder modules.
    if checkpoint_reload_state is not None:
        missing, unexpected = whisper_model.load_state_dict(checkpoint_reload_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Unexpected state-dict mismatch. Missing: {missing}, Unexpected: {unexpected}")

    architecture_changed = resize_whisper_layers(
        whisper_model,
        target_encoder_layers=target_encoder_layers,
        target_decoder_layers=target_decoder_layers,
    )
    if architecture_changed:
        print(
            "Whisper architecture override active: "
            f"encoder={whisper_model.dims.n_audio_layer}, decoder={whisper_model.dims.n_text_layer}"
        )

    if config["training"]["train_only_decoder"]:
        disable_all_grads(whisper_model.encoder)
    if config["training"]["train_only_encoder"]:
        disable_all_grads(whisper_model.decoder)

    # Apply LoRA if enabled
    is_lora_run = config["model"].get("lora", False)
    config["training"]["is_lora_run"] = is_lora_run  # Pass to training loop for debug logging
    
    if is_lora_run:        
        print("Applying LoRA adapters...")
        print("Before LoRA:")
        print_trainable_parameters(whisper_model)

        apply_lora(
            whisper_model,
            lora_config=config["model"]["lora_config"],
            train_only_decoder=config["training"]["train_only_decoder"],
            train_only_encoder=config["training"]["train_only_encoder"],
        )
        
        print("After LoRA:")
        print_lora_info(whisper_model)
    else:
        print_trainable_parameters(whisper_model)
    whisper_model.to("cuda")

    if config["augmentation"].get("deep_spec_augment", {}).get("apply", False):
        # SpecAugment applied inside the encoder as in SpecAugment++
        # https://arxiv.org/abs/2103.16858
        print(
            "Depth check before DeepSpecAugment: "
            f"encoder_blocks={len(whisper_model.encoder.blocks)}, "
            f"decoder_blocks={len(whisper_model.decoder.blocks)}, "
            f"dims.encoder={whisper_model.dims.n_audio_layer}, "
            f"dims.decoder={whisper_model.dims.n_text_layer}"
        )
        dconf = config["augmentation"]["deep_spec_augment"]
        register_deep_spec_augment_hooks(
            whisper_model,
            time_mask_param=dconf["time_mask_param"],
            freq_mask_param=dconf["freq_mask_param"],
            layer_indices=dconf["layer_indices"],
        )

    # Get datasets
    ds_config = config["dataset"]
    
    # Check if warmup dataset sampling is enabled
    warmup_dataset_idx = ds_config.get("warmup_dataset_idx", None)  # Index of the warmup dataset in train_datasets
    
    # Process datasets - get sizes if we need warmup sampling
    if warmup_dataset_idx is not None:
        train_dataset, dataset_sizes = process_dataset(
            ds_config["train_datasets"],
            ds_config["select_n_per_t_ds"],
            ds_config["train_split_name"],
            ds_config["groupby_col"],
            return_sizes=True,
        )
        print(f"\nDataset sizes: {dataset_sizes}")
        print(f"Warmup will use dataset index {warmup_dataset_idx}: {ds_config['train_datasets'][warmup_dataset_idx]}")
    else:
        train_dataset = process_dataset(
            ds_config["train_datasets"],
            ds_config["select_n_per_t_ds"],
            ds_config["train_split_name"],
            ds_config["groupby_col"],
        )
        dataset_sizes = None

    # Process validation datasets - now supports multiple named datasets
    # Ensure val_datasets is always a list (wrap single string in list)
    val_datasets_config = ds_config.get("val_datasets", [])
    if isinstance(val_datasets_config, str):
        val_datasets_config = [val_datasets_config]
    
    val_dataset_names = ds_config.get("val_dataset_names", None)

    # Auto-generate names if not specified
    if val_dataset_names is None:
        val_dataset_names = []
        for val_ds in val_datasets_config:
            # If dataset has a /, split and take the part after the last /
            if "/" in val_ds:
                name = val_ds.split("/")[-1]
            else:
                name = val_ds
            val_dataset_names.append(name)
    else:
        val_dataset_names = _pad_list_with_none(val_dataset_names, len(val_datasets_config), "val_dataset_names")

    # Create a dictionary of validation datasets
    val_datasets_dict = {}
    # Process each validation dataset separately
    for i, (val_ds, val_name) in enumerate(zip(val_datasets_config, val_dataset_names)):
        select_n = ds_config["select_n_per_v_ds"][i] if i < len(ds_config["select_n_per_v_ds"]) else None
        groupby = ds_config["groupby_col"][i] if i < len(ds_config.get("groupby_col", [])) else None

        val_dataset = process_dataset(
            [val_ds],
            [select_n] if select_n is not None else [None],
            ds_config["valid_split_name"],
            [groupby] if groupby is not None else [None],
        )
        val_datasets_dict[val_name] = val_dataset

    # Calculate steps
    config["training"]["train_steps"] = calculate_training_steps(config, train_dataset)
    config["training"]["val_steps"] = calculate_val_steps(config)
    if config["lr_scheduler"]["warmup_steps"] < 1.0:  # If smaller than one, assume it's a ratio.
        config["lr_scheduler"]["warmup_steps"] = int(config["lr_scheduler"]["warmup_steps"] * config["training"]["train_steps"])

    # Get tokenizer
    tokenizer = get_tokenizer(multilingual=True, language="de", task="transcribe")

    # Create warmup sampler if configured
    warmup_sampler = None
    use_shuffle = True
    if warmup_dataset_idx is not None and dataset_sizes is not None:
        # Calculate boundaries for each dataset in the concatenated dataset
        boundaries = get_dataset_boundary_indices(dataset_sizes)
        warmup_start, warmup_end = boundaries[warmup_dataset_idx]
        warmup_indices = list(range(warmup_start, warmup_end))
        all_indices = list(range(len(train_dataset)))
        
        warmup_sampler = WarmupDatasetSampler(
            warmup_indices=warmup_indices,
            all_indices=all_indices,
            warmup_steps=config["lr_scheduler"]["warmup_steps"],
            batch_size=config["dataset"]["batch_size"],
            shuffle=True,
        )
        use_shuffle = False  # Sampler handles shuffling
        print(f"\nWarmup sampling enabled:")
        print(f"  - Dataset: {ds_config['train_datasets'][warmup_dataset_idx]}")
        print(f"  - Warmup indices: {warmup_start} to {warmup_end} ({warmup_end - warmup_start} samples)")
        print(f"  - Warmup steps: {config['lr_scheduler']['warmup_steps']}")

    train_num_workers = config["dataset"].get("train_num_workers", min(os.cpu_count() or 1, 8))
    # Validation runs inside the training loop, so default to single-process loading.
    # This avoids worker respawn/fork stalls once the training loader and CUDA context are active.
    eval_num_workers = config["dataset"].get("eval_num_workers", 0)

    print(f"Train DataLoader workers: {train_num_workers}")
    print(f"Eval DataLoader workers: {eval_num_workers}")

    # Get dataloaders
    train_loader = get_dataloader(
        hu_dataset=train_dataset,
        tokenizer=tokenizer,
        n_mels=whisper_model.dims.n_mels,
        batch_size=config["dataset"]["batch_size"],
        sampler=warmup_sampler,
        shuffle=use_shuffle,
        no_timestamp_training=config["dataset"]["no_timestamp_training"],
        max_prompt_length=config["dataset"]["max_prompt_length"],
        prompt_use_rate=config["dataset"]["prompt_use_rate"],
        no_timestamps_rate=config["dataset"]["no_timestamp_rate"],
        num_workers=train_num_workers,
        spec_augment=config["augmentation"]["spec_augment"]["apply"],
        spec_augment_params=config["augmentation"]["spec_augment"],
        extremes_spec_augment=config["augmentation"]["extremes_spec_augment"]["apply"],
        extremes_spec_augment_params=config["augmentation"]["extremes_spec_augment"],
        apply_baseline_aug=config["augmentation"]["audio_augment"]["apply_baseline_aug"],
        apply_office_aug=config["augmentation"]["audio_augment"]["apply_office_aug"],
        apply_advanced_aug=config["augmentation"]["audio_augment"].get("apply_advanced_aug", False),
        time_stretch_min_rate=config["augmentation"]["audio_augment"].get("time_stretch", {}).get("min_rate", 0.8),
        time_stretch_max_rate=config["augmentation"]["audio_augment"].get("time_stretch", {}).get("max_rate", 1.25),
        bpe_dropout=config["augmentation"]["bpe_dropout"],
    )

    # Create multiple validation dataloaders
    val_loaders = {}
    for val_name, val_ds in val_datasets_dict.items():
        val_loaders[val_name] = get_dataloader(
            hu_dataset=val_ds,
            tokenizer=tokenizer,
            n_mels=whisper_model.dims.n_mels,
            batch_size=config["dataset"]["batch_size_eval"],
            no_timestamp_training=True,
            prompt_use_rate=0,
            no_timestamps_rate=0,
            num_workers=eval_num_workers,
        )

    # Load optimizer
    optimizer = get_optimizer(whisper_model, config["optimizer"], is_lora_run=is_lora_run)

    # Get Scheduler
    scheduler = get_scheduler(optimizer, config["lr_scheduler"], config["training"]["train_steps"])

    # Print out final config
    pprint(config)

    # Wandb
    wandb.init(config=config)

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        # Make the SLURM job ID visible in the run metadata for traceability
        wandb.config.update({"slurm_job_id": slurm_job_id}, allow_val_change=True)
        wandb.summary["slurm_job_id"] = slurm_job_id

    # Train
    main_loop(whisper_model, train_loader, val_loaders, optimizer, scheduler, config["save_dir"], config["training"])

    # Print out peak memory stats
    peak_memory_mb = torch.cuda.max_memory_allocated("cuda") / (1024**2)  # Convert to megabytes

    print(f"Peak memory usage: {peak_memory_mb:.2f} MB")

    # Save memory log
    if ENABLE_MEMORY_PROFILING:
        handle_cuda_memory_operations(config)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Script Configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()
    config = read_config(args.config)
    config['path_to_config'] = args.config

    # Ensure deterministic behavior across runs
    set_seed(config["seed"])

    main(config)
