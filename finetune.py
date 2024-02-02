import argparse
import copy
import json
import logging
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import whisper
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from whisper import Whisper as WhisperModel
from whisper.tokenizer import get_tokenizer

from whisper_finetune.data.data_loader import AudioDataset, get_dataloader
from whisper_finetune.model.model_utils import (
    evaluate,
    infinite_iter,
    save_model,
    train_step,
)
from whisper_finetune.model.optimizer import get_optimizer
from whisper_finetune.model.scheduler import get_scheduler
from whisper_finetune.utils import read_config, set_seed


def dataloader():
    print("In main.")
    hf_dataset = load_dataset("i4ds/stt4sg-350_train_all_fold_4", split="train")
    tokenizer = get_tokenizer(multilingual=True, task="transcribe")

    max_prompt_length = 256

    debug_loader = get_dataloader(
        hu_dataset=hf_dataset,
        tokenizer=tokenizer,
        batch_size=1,
        fp16=True,
        no_timestamps_training=False,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=0.5,
        no_timestamps_rate=0.5,
        shuffle=True,
        num_workers=1,
        spec_augment=True,
    )

    print("Dataset loaded.")
    # Time to iterate over the dataset.
    for i, batch in enumerate(tqdm(debug_loader)):
        assert len(batch) == 3
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[1][0].tolist())

        prompt = torch.tensor(batch[1][0], dtype=torch.int32)
        decode = torch.tensor(batch[2][0], dtype=torch.int32)
        decode = decode[decode != -100]
        print("Prompt length:", len(prompt))
        print(tokenizer.decode(prompt.tolist()))
        print("Decode length:", len(decode))
        print(tokenizer.decode(decode.tolist()))
        break


def list_to_int(l):
    return [l.int() for l in l]


def audiodataset():
    hf_dataset = load_dataset("i4ds/stt4sg-350_train_all_fold_4", split="train")
    tokenizer = get_tokenizer(multilingual=True, task="transcribe", language="de")

    dataset = AudioDataset(
        hf_dataset,
        tokenizer,
        fp16=True,
        no_timestamps_training=False,
        max_prompt_length=223,
        prompt_use_rate=1,
        no_timestamps_rate=0.5,
        spec_augment=False,
    )
    for i in range(5, 9):
        t_prompt = hf_dataset[i]["prompt"]
        print(t_prompt)
        encoded = dataset._get_prompt_tokens(t_prompt, True)
        print(encoded)

        print(tokenizer.decode_with_timestamps(encoded))
        print(tokenizer.decode_with_timestamps([879, 482]))


def whisper():
    import whisper

    hf_dataset = load_dataset("i4ds/stt4sg-350_train_all_fold_4", split="train")
    hf_dataset = hf_dataset.with_format(type="torch")
    model = whisper.load_model("large-v3")
    print(hf_dataset[8])
    print(hf_dataset[8]["text"])
    result = model.transcribe(hf_dataset[8]["audio"]["array"])
    tokens = [x["tokens"] for x in result["segments"]]
    tokens = [item for sublist in tokens for item in sublist]
    tokenizer = get_tokenizer(multilingual=True, task="transcribe")
    print(tokenizer.decode_with_timestamps(tokens))


def main_loop(
    model: WhisperModel,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: dict,
) -> None:
    min_loss = evaluate(model, dev_loader)
    print(f"Initial loss: {min_loss}")
    logging.info(f"eval\t0\t{min_loss}\t{scheduler.get_last_lr()[0]}")
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

        if ((step <= config["eval_warmup"]) and (step % config["eval_steps_early"] == 0)) or (
            (step > config["eval_warmup"]) and (step % config["eval_steps"] == 0)
        ):
            eval_loss = evaluate(model, dev_loader)
            tqdm.write(f"Step {step}: validation loss={eval_loss}")
            if eval_loss < min_loss:
                min_loss = eval_loss
                save_model(model, f"{config['save_dir']}/best_model.pt")

            if config["save_all_checkpoints"]:
                save_model(model, f"{config['save_dir']}/step{step}.pt")

            logging.info(f"eval\t{step}\t{eval_loss}\t{scheduler.get_last_lr()[0]}")
            save_model(model, f"{config['save_dir']}/last_model.pt")


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
    # Get tokenizer
    tokenizer = get_tokenizer(multilingual=True, language="de", task="transcribe")
    # Get dataloader
    ds_config = config["dataset"]
    train_datasets = []
    for dataset_name in ds_config["train_datasets"]:
        train_datasets.append(load_dataset(dataset_name, split="train"))
    train_dataset = concatenate_datasets(train_datasets)
    val_datasets = []
    for dataset_name in ds_config["validation_datasets"]:
        val_datasets.append(load_dataset(dataset_name, split="validation"))
    val_dataset = concatenate_datasets(val_datasets)
    """
    model:
  init_name: large-v3
  fp16: True
dataset:
  dataset_name: i4ds/stt
  no_timestamp_training: False
  prompt_use_rate: 0.5
  no_timestamp_rate: 0.5
  batch_size: 4
    """
    train_loader = get_dataloader(
        hu_dataset=train_dataset,
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        fp16=config["fp16"],
        no_timestamps_training=config["no_timestamps_training"],
        max_prompt_length=config["max_prompt_length"],
        prompt_use_rate=config["prompt_use_rate"],
        no_timestamps_rate=config["no_timestamps_rate"],
        num_workers=os.cpu_count(),
        num_workers=config["num_workers"],
        spec_augment=config["spec_augment"],
    )
    val_loader = get_dataloader(
        hu_dataset=val_dataset,
        tokenizer=tokenizer,
        batch_size=16,
        fp16=config["fp16"],
        no_timestamps_training=True,
        prompt_use_rate=0,
        no_timestamps_rate=0,
        num_workers=os.cpu_count(),
        spec_augment=config["spec_augment"],
    )
    # Load model
    whisper_model = whisper.load_model(config["model"]["init_name"], "cuda")  # No point to train on CPU
    whisper_model = whisper_model.half() if config["model"]["fp16"] else whisper_model
    # Load optimizer
    optimizer = get_optimizer(whisper_model, config["optimizer"])

    # Get Scheduler
    scheduler = get_scheduler(config["lr_scheduler"], optimizer)

    # Further adjustments to use `config` instead of `args` throughout your code
    main_loop(
        whisper_model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        config,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Script Configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()
    config = read_config(args.config)
    main(config)
