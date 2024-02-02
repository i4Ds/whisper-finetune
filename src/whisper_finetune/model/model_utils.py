import argparse
import copy
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import whisper
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from whisper import Whisper
from whisper.tokenizer import get_tokenizer

from whisper_finetune.model.scheduler import (
    get_cosine_annealing_with_warmup_restarts,
    get_cosine_annealing_with_warmup_restarts_chill,
)


def train_step(
    model: Whisper,
    train_iter: Iterator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    accum_grad_steps: int,
    train_only_decoder: bool,
    max_grad_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    for _ in range(accum_grad_steps):
        x, y_in, y_out = next(train_iter)
        x, y_in, y_out = x.to(model.device), y_in.to(model.device), y_out.to(model.device)

        if train_only_decoder:
            with torch.no_grad():
                audio_features = model.embed_audio(x)
        else:
            audio_features = model.embed_audio(x)
        logits = model.logits(y_in, audio_features=audio_features)
        loss = F.cross_entropy(logits.transpose(1, 2), y_out)

        loss = loss / accum_grad_steps
        loss.backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return total_loss


@torch.no_grad()
def evaluate(model: Whisper, dev_loader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    for x, y_in, y_out in tqdm(dev_loader):
        x, y_in, y_out = x.to(model.device), y_in.to(model.device), y_out.to(model.device)
        logits = model(x, y_in)

        if torch.isnan(logits).any():
            print("Warning: logits nan")

        loss = F.cross_entropy(logits.transpose(1, 2), y_out)

        if torch.isnan(loss).any():
            print("Warning: loss nan")
        else:
            total_loss += loss.item()
    return total_loss / len(dev_loader)


def save_model(model: Whisper, save_path: str) -> None:
    # save model in half precision to save space
    model = copy.deepcopy(model).half()
    # save model weights and config in a dictionary that can be loaded with `whisper.load_model`
    torch.save({"model_state_dict": model.state_dict(), "dims": asdict(model.dims)}, save_path)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_args(args: argparse.Namespace, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(vars(args), indent=4, ensure_ascii=False))


def infinite_iter(data_loader: DataLoader) -> Iterator:
    while True:
        for batch in data_loader:
            yield batch


def main_loop(
    model: Whisper,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    args: argparse.Namespace,
) -> None:
    min_loss = evaluate(model, dev_loader)
    print(f"Initial loss: {min_loss}")
    logging.info(f"eval\t0\t{min_loss}\t{scheduler.get_last_lr()[0]}")
    pbar = tqdm(range(1, args.train_steps + 1))
    train_iter = infinite_iter(train_loader)
    for step in pbar:
        train_loss = train_step(
            model,
            train_iter,
            optimizer,
            scheduler,
            args.accum_grad_steps,
            args.train_only_decoder,
            args.max_grad_norm,
        )
        pbar.set_postfix({"loss": train_loss})
        logging.info(f"train\t{step}\t{train_loss}\t{scheduler.get_last_lr()[0]}")

        if ((step <= args.eval_warmup) and (step % args.eval_steps_early == 0)) or (
            (step > args.eval_warmup) and (step % args.eval_steps == 0)
        ):
            eval_loss = evaluate(model, dev_loader)
            tqdm.write(f"Step {step}: validation loss={eval_loss}")
            if eval_loss < min_loss:
                min_loss = eval_loss
                save_model(model, f"{args.save_dir}/best_model.pt")

            if args.save_all_checkpoints:
                save_model(model, f"{args.save_dir}/step{step}.pt")

            logging.info(f"eval\t{step}\t{eval_loss}\t{scheduler.get_last_lr()[0]}")
            save_model(model, f"{args.save_dir}/last_model.pt")


def main():
    args = get_parser().parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_args(args, f"{args.save_dir}/args.json")

    logging.basicConfig(
        filename=f"{args.save_dir}/model.log",
        encoding="utf-8",
        level=logging.DEBUG,
        format="%(asctime)s\t%(message)s",
    )

    tokenizer = get_tokenizer(multilingual=".en" not in args.model, task="transcribe")
    model = whisper.load_model(args.model, args.device)
    #  -1 is for the special token `sot_prev` and the other half is for the transcribed tokens
    max_prompt_length = model.dims.n_text_ctx // 2 - 1

    fp16 = args.device == "cuda"
    train_loader = get_dataloader(
        json=args.train_json,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        fp16=fp16,
        no_timestamps_training=args.no_timestamps_training,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=args.prompt_use_rate,
        no_timestamps_rate=args.no_timestamps_rate,
        shuffle=True,
        num_workers=args.data_loader_workers,
        spec_augment=args.spec_augment,
    )
    dev_loader = get_dataloader(
        json=args.dev_json,
        tokenizer=tokenizer,
        batch_size=args.dev_batch_size,
        fp16=fp16,
        no_timestamps_training=args.no_timestamps_training,
        max_prompt_length=max_prompt_length,
        # always use prompts and timestamps for validation to make it deterministic
        prompt_use_rate=1.0,
        no_timestamps_rate=0.0,
        shuffle=False,
        num_workers=args.data_loader_workers,
    )
    if args.use_adam_8bit:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("For using Adam 8bit optimizer you need to have bitsandbytes installed.")
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr, betas=[0.9, 0.98], eps=1e-6)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.lr_scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.train_steps
        )
    elif args.lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.train_steps
        )
    elif args.lr_scheduler_type == "cosine_with_restarts":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.train_steps,
            num_cycles=args.lr_num_cycles,
        )
    elif args.lr_scheduler_type == "cosine_with_warmup_restarts":
        scheduler = get_cosine_annealing_with_warmup_restarts(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.train_steps,
            num_cycles=args.lr_num_cycles,
            gamma=args.lr_gamma,
        )
    elif args.lr_scheduler_type == "cosine_with_warmup_restarts_chill":
        scheduler = get_cosine_annealing_with_warmup_restarts_chill(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.train_steps,
            num_cycles=args.lr_num_cycles,
            gamma=args.lr_gamma,
            chill_steps=args.chill_steps,
            chill_range=args.chill_range,
        )
    else:
        raise Exception(
            f"Unknown learning rate scheduler: {args.lr_scheduler_type}. Must be linear, cosine, cosine_with_restarts or cosine_with_warmup_restarts"
        )

    main_loop(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
    )


if __name__ == "__main__":
    main()
