import argparse
import copy
import io
import json
import os
import random
from dataclasses import asdict
from functools import partial
from typing import Iterator, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from whisper import _ALIGNMENT_HEADS, _MODELS, _download, available_models
from whisper.model import AudioEncoder, TextDecoder, Whisper
from whisper.version import __version__
from whisper.tokenizer import get_tokenizer
from whisper_finetune.eval.utils import normalize_text, VOCAB_SPECS
from whisper_finetune.eval.wer import WER

def train_step(
    model: Whisper,
    train_iter: Iterator,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    t_config: dict,
) -> float:
    model.train()
    total_loss = 0.0

    # Read variables from t_config
    mixed_precision = t_config["mixed_precision"]
    accum_grad_steps = t_config["accum_grad_steps"]
    train_only_decoder = t_config["train_only_decoder"]
    max_grad_norm = t_config["max_grad_norm"]
    mp_dtype = torch.float16 if t_config["mp_dtype"] == "fp16" else torch.bfloat16

    # Setup grad scaler, if using fp16
    # bfloat16 is not supported by torch.cuda.amp.GradScaler: RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
    # Unsure what the solution to this problem ? is
    if mixed_precision and not model.is_bfloat:
        print("Detected fp16 training. Using torch.cuda.amp.GradScaler.")
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    max_retries = 3  # Set the maximum number of retries for a training step

    for _ in range(accum_grad_steps):
        retry_count = 0
        while retry_count < max_retries:
            try: # Illegal memory access happens sometime. 
                x, y_in, y_out = next(train_iter)
                x, y_in, y_out = x.to(model.device), y_in.to(model.device), y_out.to(model.device)
                with torch.autocast(device_type="cuda", enabled=mixed_precision, dtype=mp_dtype):
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
                break  # Exit retry loop if no error occurs
            except RuntimeError as e:
                if "CUDA error: an illegal memory" in str(e):
                    print(f"Caught illegal memory access, retry {retry_count + 1}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        print("Max retries reached. Something is wrong.")
                        raise
                else:
                    raise

    if scaler:
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    if scaler:
        scaler.step(optimizer)
        lr_scheduler.step()
        scaler.update()
    else:
        optimizer.step()
        lr_scheduler.step()

    optimizer.zero_grad()

    return total_loss


@torch.no_grad()
def evaluate(model: Whisper, dev_loader: DataLoader, t_config: dict) -> float:
    model.eval()
    total_loss = 0.0
    pred_sentences, true_sentences = [], []

    # Read variables from t_config
    mixed_precision = t_config["mixed_precision"]
    mp_dtype = torch.float16 if t_config["mp_dtype"] == "fp16" else torch.bfloat16

    # Get tokenizer & eval metric
    tokenizer = get_tokenizer(multilingual=True, language="de", task="transcribe")

    wer = WER()

    for x, y_in, y_out in tqdm(dev_loader):
        x, y_in, y_out = x.to(model.device), y_in.to(model.device), y_out.to(model.device)
        with torch.autocast(device_type="cuda", enabled=mixed_precision, dtype=mp_dtype):
            logits = model(x, y_in)

            if torch.isnan(logits).any():
                print("Warning: logits nan")

            loss = F.cross_entropy(logits.transpose(1, 2), y_out)

            # Convert logits to token IDs
            pred_token_ids = torch.argmax(logits, dim=-1) 

            # Filter out -100 values, special tokens and decode.
            batch_pred = [tokenizer.decode([id for id in ids.cpu().tolist() if id not in tokenizer.special_tokens.values() and id != -100]) for ids in pred_token_ids]
            batch_true = [tokenizer.decode([id for id in ids.cpu().tolist() if id not in tokenizer.special_tokens.values() and id != -100]) for ids in y_out]

            # Normalize and filter out empty sentences in the reference
            mask = [True if x != '' else False for x in batch_true]
            batch_pred = [normalize_text(x, **VOCAB_SPECS["v0"]) for x, m in zip(batch_pred, mask) if m]
            batch_true = [normalize_text(x, **VOCAB_SPECS["v0"]) for x, m in zip(batch_true, mask) if m]

            # Append 
            pred_sentences.extend(batch_pred)
            true_sentences.extend(batch_true)

        if torch.isnan(loss).any():
            print("Warning: loss nan")
        else:
            total_loss += loss.item()

    wer = wer._compute(
            predictions=pred_sentences,
            references=true_sentences,
        )

    del x, y_in, y_out, pred_sentences, true_sentences, batch_pred, batch_true
    return total_loss / len(dev_loader), wer


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


class CheckpointedAudioEncoder(AudioEncoder):
    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)

        x = self.ln_post(x)
        return x


class CheckpointedTextDecoder(TextDecoder):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        # Call the initializer of the parent class (TextDecoder)
        super().__init__(n_vocab, n_ctx, n_state, n_head, n_layer)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        x = x.to(xa.dtype)

        for block in self.blocks:
            block_p = partial(block, xa=xa, mask=self.mask, kv_cache=kv_cache)
            x = checkpoint(block_p, x, use_reentrant=False)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits


def load_model_and_set_heads(
    model: Whisper,
    name: str,
    device: Union[str, torch.device],
    download_root: Optional[str] = None,
    in_memory: bool = False,
) -> Whisper:
    """
    Load a Whisper ASR model, set the alignment heads, and move the model to a device.

    Parameters
    ----------
    name : str
        One of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        The PyTorch device to move the model to.
    download_root: str, optional
        Path to download the model files; by default, it uses "~/.cache/whisper".
    in_memory: bool, optional
        Whether to preload the model weights into host memory.

    Returns
    -------
    model : Whisper
        The updated Whisper model instance.
    """
    if download_root is None:
        default = os.path.join(os.path.expanduser("~"), ".cache")
        download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")

    if name in _MODELS:
        checkpoint_file = _download(_MODELS[name], download_root, in_memory)
        alignment_heads = _ALIGNMENT_HEADS[name]
    elif os.path.isfile(name):
        checkpoint_file = open(name, "rb").read() if in_memory else name
        alignment_heads = None
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb") as fp:
        checkpoint = torch.load(fp, map_location=device)
    del checkpoint_file

    model.load_state_dict(checkpoint["model_state_dict"])

    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)

    return model.to(device)
