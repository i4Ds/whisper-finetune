import argparse
import logging
import os
from pathlib import Path
from socket import gethostname

import torch
import wandb
import whisper
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from whisper import Whisper as WhisperModel
from whisper.tokenizer import get_tokenizer

from whisper_finetune.data.data_loader import get_dataloader
from whisper_finetune.model.model_utils import (
    evaluate,
    infinite_iter,
    save_model,
    train_step,
)
from whisper_finetune.model.optimizer import get_optimizer
from whisper_finetune.model.scheduler import get_scheduler
from whisper_finetune.utils import distributed_setup, read_config, set_seed


def main_loop(
    model: WhisperModel,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: dict,
) -> None:
    wandb.init(project="my_project", config=config)  # Initialize a new wandb run
    wandb.watch(model, log="all")  # Log all gradients and model parameters

    min_loss = evaluate(model, dev_loader)
    print(f"Initial loss: {min_loss}")
    logging.info(f"eval\t0\t{min_loss}\t{scheduler.get_last_lr()[0]}")
    wandb.log({"Initial loss": min_loss})  # Log initial loss

    pbar = tqdm(range(1, config["train_steps"] + 1))
    train_iter = infinite_iter(train_loader)
    for step in pbar:
        train_loss = train_step(
            model,
            train_iter,
            optimizer,
            scheduler,
            config["accum_grad_steps"],
            config["train_only_decoder"],
            config["max_grad_norm"],
        )
        pbar.set_postfix({"loss": train_loss})
        logging.info(f"train\t{step}\t{train_loss}\t{scheduler.get_last_lr()[0]}")
        wandb.log({"Train loss": train_loss})  # Log training loss

        if ((step <= config["eval_warmup"]) and (step % config["eval_steps_early"] == 0)) or (
            (step > config["eval_warmup"]) and (step % config["eval_steps"] == 0)
        ):
            eval_loss = evaluate(model, dev_loader)
            tqdm.write(f"Step {step}: validation loss={eval_loss}")
            wandb.log({"Validation loss": eval_loss})  # Log validation loss

            if eval_loss < min_loss:
                min_loss = eval_loss
                save_model(model, f"{config['save_dir']}/best_model.pt")

            if config["save_all_checkpoints"]:
                save_model(model, f"{config['save_dir']}/step{step}.pt")

            logging.info(f"eval\t{step}\t{eval_loss}\t{scheduler.get_last_lr()[0]}")
            save_model(model, f"{config['save_dir']}/last_model.pt")

    wandb.finish()  # End the wandb run


def main(config):
    set_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=f"{config['save_dir']}/model.log",
        encoding="utf-8",
        level=logging.DEBUG,
        format="%(asctime)s\t%(message)s",
    )

    # Setup distributed training
    dist_is_init = distributed_setup(rank=config["rank"], world_size=config["world_size"])

    # Get datasets
    ds_config = config["dataset"]
    train_datasets = []
    for dataset_name in ds_config["train_datasets"]:
        train_datasets.append(load_dataset(dataset_name, split="train"))
    train_dataset = concatenate_datasets(train_datasets)
    val_datasets = []
    for dataset_name in ds_config["validation_datasets"]:
        val_datasets.append(load_dataset(dataset_name, split="validation"))
    val_dataset = concatenate_datasets(val_datasets)

    # Get tokenizer
    tokenizer = get_tokenizer(multilingual=True, language="de", task="transcribe")

    # Create distributed sampler
    if dist_is_init:
        # Add distributed sampler
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=config["world_size"], rank=config["rank"]
        )
        # Assign GPUS
        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(local_rank)
        print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")
    else:
        sampler = None

    # Get dataloaders
    train_loader = get_dataloader(
        hu_dataset=train_dataset,
        tokenizer=tokenizer,
        sampler=sampler,
        batch_size=config["dataset"]["batch_size"],
        fp16=config["model"]["fp16"],
        no_timestamps_training=config["dataset"]["no_timestamps_training"],
        max_prompt_length=config["dataset"]["max_prompt_length"],
        prompt_use_rate=config["dataset"]["prompt_use_rate"],
        no_timestamps_rate=config["dataset"]["no_timestamps_rate"],
        num_workers=os.cpu_count(),
        spec_augment=config["augmentation"]["spec_augment"]["apply"],
        time_mask_param=config["augmentation"]["spec_augment"]["time_mask_param"],
        freq_mask_param=config["augmentation"]["spec_augment"]["freq_mask_param"],
        p=config["augmentation"]["spec_augment"]["p"],
    )
    val_loader = get_dataloader(
        hu_dataset=val_dataset,
        tokenizer=tokenizer,
        batch_size=16,
        fp16=config["model"]["fp16"],
        no_timestamps_training=True,
        prompt_use_rate=0,
        no_timestamps_rate=0,
        num_workers=os.cpu_count(),
        spec_augment=False,
    )
    # Load model
    whisper_model = whisper.load_model(config["model"]["init_name"], "cuda")  # No point to train on CPU
    whisper_model = whisper_model.half() if config["model"]["fp16"] else whisper_model

    # Load optimizer
    optimizer = get_optimizer(whisper_model, config["optimizer"])

    # Get Scheduler
    scheduler = get_scheduler(optimizer, config["lr_scheduler"])

    # Train
    main_loop(whisper_model, train_loader, val_loader, optimizer, scheduler, config["training"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Script Configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()
    config = read_config(args.config)
    # Get env varaibles
    world_size = os.environ.get("WORLD_SIZE", 1)
    rank = os.environ.get("RANK", 0)
    gpus_per_node = os.environ.get("GPUS_PER_NODE", 1)
    assert gpus_per_node == torch.cuda.device_count()
    # Add env variables to config
    config["world_size"] = world_size
    config["rank"] = rank
    config["gpus_per_node"] = gpus_per_node

    main(config)
