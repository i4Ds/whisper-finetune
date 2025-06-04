import argparse
import logging
import os
import subprocess
from pathlib import Path
from pprint import pprint

import torch
import wandb
import whisper
from torch.utils.data import DataLoader
from tqdm import tqdm
from whisper import Whisper as WhisperModel
from whisper.tokenizer import get_tokenizer

from whisper_finetune.data.data_loader import get_dataloader
from whisper_finetune.data.utils import process_dataset
from whisper_finetune.model.model_utils import (
    CheckpointedStochasticAudioEncoder,
    CheckpointedStochasticTextDecoder,
    evaluate,
    infinite_iter,
    load_model_and_set_heads,
    register_deep_spec_augment_hooks,
    save_model,
    train_step,
)
from whisper_finetune.model.optimizer import get_optimizer
from whisper_finetune.model.scheduler import get_scheduler
from whisper_finetune.utils import (
    calculate_training_steps,
    calculate_val_steps,
    disable_all_grads,
    get_unique_base_path,
    handle_cuda_memory_operations,
    print_size_of_model,
    print_trainable_parameters,
    read_config,
    set_seed,
)

ENABLE_MEMORY_PROFILING = False


def main_loop(
    model: WhisperModel,
    train_loader: DataLoader,
    dev_loader: DataLoader,
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
        dev_loader (DataLoader): DataLoader for the development set.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        scheduler (torch.optim.lr_scheduler.LambdaLR): Learning rate scheduler.
        save_dir (str): Directory to save the trained model.
        t_config (dict): Configuration settings for training.

    Returns:
        None
    """
    wandb.watch(model, log="all")

    min_loss, min_wer = evaluate(model, dev_loader, t_config)
    print(f"Initial loss: {min_loss}. Initial WER: {min_wer}")
    logging.info(f"eval\t0\t{min_loss}\t{scheduler.get_last_lr()[0]}")
    wandb.log({"Initial loss": min_loss, "Initial WER": min_wer})

    pbar = tqdm(range(1, t_config["train_steps"] + 1))
    train_iter = infinite_iter(train_loader)
    for step in pbar:
        train_loss = train_step(model, train_iter, optimizer, scheduler, t_config)
        pbar.set_postfix({"loss": train_loss})
        logging.info(f"train\t{step}\t{train_loss}\t{scheduler.get_last_lr()[0]}")
        wandb.log({"Learning rate": scheduler.get_last_lr()[0]})
        wandb.log({"Train loss": train_loss})  # Log training loss
        assert (
            train_loss < t_config["max_train_loss"]
        ), f"Train loss is above {t_config['max_train_loss']}, the loss is unable to converge."

        if (step % t_config["val_steps"]) == 0 or step == t_config["train_steps"] + 1:
            eval_loss, eval_wer = evaluate(model, dev_loader, t_config)
            tqdm.write(f"Step {step}: validation loss={eval_loss}")
            wandb.log({"Validation loss": eval_loss, "Validation WER": eval_wer})  # Log validation loss

            if eval_wer < min_wer:
                min_wer = eval_wer
                save_model(model, f"{save_dir}/best_model.pt")

            if t_config["save_all_checkpoints"]:
                save_model(model, f"{save_dir}/step{step}.pt")

            logging.info(f"eval\t{step}\t{eval_loss}\t{scheduler.get_last_lr()[0]}")

    save_model(model, f"{save_dir}/last_model.pt")
    wandb.save(f"{save_dir}/last_model.pt")  # Save last model to wandb
    wandb.save(f"{save_dir}/best_model.pt")  # Save best model to wandb


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

    config["save_dir"] = get_unique_base_path() + "_" + config["save_dir"]

    # Create save directory
    Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)

    # Save config
    logging.basicConfig(
        filename=f"{config['save_dir']}/model.log",
        encoding="utf-8",
        level=logging.DEBUG,
        format="%(asctime)s\t%(message)s",
    )
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

    ## Get model
    whisper_model = whisper.load_model(config["model"]["init_name"], device="cpu")

    # bfloat16 training?
    if config["model"]["bfloat16"]:
        f = print_size_of_model(whisper_model)
        whisper_model = whisper_model.bfloat16()
        q = print_size_of_model(whisper_model)
        print("Bfloat16 is {0:.2f} times smaller".format(f / q))
        whisper_model.is_bfloat = True
    else:
        whisper_model.is_bfloat = False

    # If gradient checkpointing is enabled, wrap the model with checkpointing
    if config["training"]["gradient_checkpointing_encoder"]:
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
                config["training"]["stochastic_depth"],
            )
    if config["training"]["gradient_checkpointing_decoder"]:
        del whisper_model.decoder
        whisper_model.decoder = CheckpointedStochasticTextDecoder(
            whisper_model.dims.n_vocab,
            whisper_model.dims.n_text_ctx,
            whisper_model.dims.n_text_state,
            whisper_model.dims.n_text_head,
            whisper_model.dims.n_text_layer,
            config["training"]["stochastic_depth"],
        )

    # We need to reload weights for deletected Decoder and Encoder because we need to set the heads.
    if config["training"]["gradient_checkpointing_decoder"] or config["training"]["gradient_checkpointing_encoder"]:
        whisper_model = load_model_and_set_heads(whisper_model, config["model"]["init_name"])

    if config["training"]["train_only_decoder"]:
        disable_all_grads(whisper_model.encoder)
    if config["training"]["train_only_encoder"]:
        disable_all_grads(whisper_model.decoder)

    whisper_model.to("cuda")

    if config["augmentation"].get("deep_spec_augment", {}).get("apply", False):
        dconf = config["augmentation"]["deep_spec_augment"]
        register_deep_spec_augment_hooks(
            whisper_model,
            time_mask_param=dconf.get("time_mask_param", 20),
            freq_mask_param=dconf.get("freq_mask_param", 10),
        )

    # Get datasets
    ds_config = config["dataset"]
    train_dataset = process_dataset(
        ds_config["train_datasets"],
        ds_config["select_n_per_t_ds"],
        ds_config["train_split_name"],
        ds_config["groupby_col"],
    )

    # Process validation datasets
    val_dataset = process_dataset(
        ds_config["val_datasets"],
        ds_config["select_n_per_v_ds"],
        ds_config["valid_split_name"],
        ds_config["groupby_col"],
    )

    # Calculate steps
    config["training"]["train_steps"] = calculate_training_steps(config, train_dataset)
    config["training"]["val_steps"] = calculate_val_steps(config)
    if config["lr_scheduler"]["warmup_steps"] < 1.0:  # If smaller than one, assume it's a ratio.
        config["lr_scheduler"]["warmup_steps"] = int(config["lr_scheduler"]["warmup_steps"] * len(train_dataset))

    # Get tokenizer
    tokenizer = get_tokenizer(multilingual=True, language="de", task="transcribe")

    # Get dataloaders
    train_loader = get_dataloader(
        hu_dataset=train_dataset,
        tokenizer=tokenizer,
        n_mels=128 if "v3" in config["model"]["init_name"] else 80,
        batch_size=config["dataset"]["batch_size"],
        no_timestamp_training=config["dataset"]["no_timestamp_training"],
        max_prompt_length=config["dataset"]["max_prompt_length"],
        prompt_use_rate=config["dataset"]["prompt_use_rate"],
        no_timestamps_rate=config["dataset"]["no_timestamp_rate"],
        num_workers=min(os.cpu_count(), 8),
        spec_augment=config["augmentation"]["spec_augment"]["apply"],
        spec_augment_params=config["augmentation"]["spec_augment"],
        audio_aug=config["augmentation"]["audio_augment"]["apply"],
        audio_augment_params=config["augmentation"]["audio_augment"],
    )
    val_loader = get_dataloader(
        hu_dataset=val_dataset,
        tokenizer=tokenizer,
        n_mels=128 if "v3" in config["model"]["init_name"] else 80,
        batch_size=config["dataset"]["batch_size_eval"],
        no_timestamp_training=True,
        prompt_use_rate=0,
        no_timestamps_rate=0,
        num_workers=min(os.cpu_count(), 8),
        spec_augment=False,
        audio_aug=False,
    )

    # Load optimizer
    optimizer = get_optimizer(whisper_model, config["optimizer"])

    # Get Scheduler
    scheduler = get_scheduler(optimizer, config["lr_scheduler"], config["training"]["train_steps"])

    # Print out final config
    pprint(config)

    # Wandb
    wandb.init(config=config)

    # Train
    main_loop(whisper_model, train_loader, val_loader, optimizer, scheduler, config["save_dir"], config["training"])

    # Print out peak memory stats
    peak_memory_mb = torch.cuda.max_memory_allocated("cuda") / (1024**2)  # Convert to megabytes

    print(f"Peak memory usage: {peak_memory_mb:.2f} MB")

    # Save memory log
    if ENABLE_MEMORY_PROFILING:
        handle_cuda_memory_operations(config)

    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Script Configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()
    config = read_config(args.config)

    # Ensure deterministic behavior across runs
    set_seed(config["seed"])

    main(config)
