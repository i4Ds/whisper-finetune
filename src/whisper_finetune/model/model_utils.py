import argparse
import copy
import json
import random
from dataclasses import asdict
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from whisper import Whisper


def train_step(
    model: Whisper,
    train_iter: Iterator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    accum_grad_steps: int,
    train_only_decoder: bool,
    max_grad_norm: float,
    fp16: bool,
) -> float:
    model.train()
    total_loss = 0.0

    # Setup grad scaler, if using fp16
    if fp16:
        print("Detected fp16 training. Using torch.cuda.amp.GradScaler.")
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for _ in range(accum_grad_steps):
        x, y_in, y_out = next(train_iter)
        x, y_in, y_out = x.to(model.device), y_in.to(model.device), y_out.to(model.device)
        with torch.cuda.amp.autocast(enabled=fp16):
            if train_only_decoder:
                with torch.no_grad():
                    audio_features = model.embed_audio(x)
            else:
                audio_features = model.embed_audio(x)
            logits = model.logits(y_in, audio_features=audio_features)
            loss = F.cross_entropy(logits.transpose(1, 2), y_out)

        loss = loss / accum_grad_steps
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return total_loss


@torch.no_grad()
def evaluate(model: Whisper, dev_loader: DataLoader, fp16: bool) -> float:
    model.eval()
    total_loss = 0.0
    for x, y_in, y_out in tqdm(dev_loader):
        x, y_in, y_out = x.to(model.device), y_in.to(model.device), y_out.to(model.device)
        with torch.cuda.amp.autocast(enabled=fp16):
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
