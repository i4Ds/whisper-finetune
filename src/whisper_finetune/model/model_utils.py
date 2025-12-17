import copy
import io
import os
from dataclasses import asdict
from functools import partial
from typing import Callable, Iterator, Optional, Union

import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import wandb
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from whisper import _ALIGNMENT_HEADS, _MODELS, _download, available_models
from whisper.model import AudioEncoder, TextDecoder, Whisper
from whisper.tokenizer import get_tokenizer

from whisper_finetune.eval.utils import VOCAB_SPECS, normalize_text


def train_step(
    model: Whisper,
    train_iter: Iterator,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    t_config: dict,
    lora_tracker=None,
    step: int = None,
) -> float:
    model.train()
    total_loss = 0.0

    # Read variables from t_config
    mixed_precision_training = t_config["mixed_precision_training"]
    accum_grad_steps = t_config["accum_grad_steps"]
    max_grad_norm = t_config["max_grad_norm"]
    mp_dtype = torch.float16 if t_config["mp_dtype"] == "fp16" else torch.bfloat16
    label_smoothing = t_config.get("label_smoothing", 0.0)  # Label smoothing for cross-entropy
    is_lora_run = t_config.get("is_lora_run", False)

    # Setup grad scaler ONLY for fp16 (NOT for bfloat16!)
    # bfloat16 does NOT need loss scaling - it has sufficient dynamic range
    # GradScaler with bf16 causes: RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
    use_fp16 = mixed_precision_training and t_config["mp_dtype"] == "fp16"
    if use_fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None  # No scaler for bf16 or fp32

    max_retries = 3  # Set the maximum number of retries for a training step

    for _ in range(accum_grad_steps):
        retry_count = 0
        while retry_count < max_retries:
            try:  # Illegal memory access happens sometimes.
                x, y_in, y_out = next(train_iter)
                x, y_in, y_out = x.to(model.device), y_in.to(model.device), y_out.to(model.device)
                with torch.autocast(device_type="cuda", enabled=mixed_precision_training, dtype=mp_dtype):
                    audio_features = model.embed_audio(x)
                    logits = model.logits(y_in, audio_features=audio_features)
                    loss = F.cross_entropy(logits.transpose(1, 2), y_out, label_smoothing=label_smoothing)

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

    # Log LoRA debug info AFTER backward() but BEFORE optimizer.step()
    # This captures gradient information before gradients are zeroed
    # Only log at eval steps (when step is provided and is an eval step)
    val_steps = t_config.get("val_steps", None)
    train_steps = t_config.get("train_steps", None)
    should_log_lora = (
        is_lora_run and 
        step is not None and 
        val_steps is not None and 
        ((step % val_steps == 0) or step == train_steps)
    )
    if should_log_lora:
        from whisper_finetune.model.lora import log_lora_debug_info
        log_lora_debug_info(model, step=step, tracker=lora_tracker, log_to_wandb=True)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Take snapshot of LoRA params BEFORE optimizer.step() to track updates
    if is_lora_run and lora_tracker is not None:
        lora_tracker.snapshot()

    if scaler:
        # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/8
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        if (
            scale_before <= scaler.get_scale()
        ):  # If the scale has stayed the same or increased (which is good, means it's stable), then step the scheduler.
            # If this is not the case, scaler.step(optimizer) also didn't happen.
            lr_scheduler.step()
        scaler.update()
    else:
        optimizer.step()
        lr_scheduler.step()

    optimizer.zero_grad()

    return total_loss


def save_model(model: Whisper, save_path: str) -> None:
    # save model in half precision to save space
    model = copy.deepcopy(model).half()
    # save model weights and config in a dictionary that can be loaded with `whisper.load_model`
    torch.save({"model_state_dict": model.state_dict(), "dims": asdict(model.dims)}, save_path)


def infinite_iter(data_loader: DataLoader) -> Iterator:
    while True:
        for batch in data_loader:
            yield batch


class StochasticDepthMixin:
    """
    Mixin class providing stochastic depth functionality with gradient checkpointing.
    See: https://arxiv.org/abs/1603.09382
    """

    def stochastic_depth(self, x: Tensor, layer: Callable[[Tensor], Tensor], p: float) -> Tensor:
        """
        Apply stochastic depth: randomly skip a layer during training with probability p.
        Uses gradient checkpointing to save memory.

        Args:
            x: Input tensor
            layer: Layer function to apply
            p: Probability of skipping the layer during training

        Returns:
            Output tensor (either skipped or after applying the layer)
        """
        if self.training and torch.rand(1).item() < p:
            return x  # Skip the layer
        return checkpoint(layer, x, use_reentrant=False)  # Apply the layer with checkpointing


class CheckpointedStochasticAudioEncoder(StochasticDepthMixin, AudioEncoder):
    """
    CheckpointedStochasticAudioEncoder, which contains stochastic depth and checkpointing, also used in the original Whisper model.
    See: https://arxiv.org/abs/1603.09382
    """

    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, stochastic_depth_prob: float):
        super().__init__(n_mels, n_ctx, n_state, n_head, n_layer)
        self.stochastic_depth_prob = stochastic_depth_prob

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
            block_p = partial(block)
            x = self.stochastic_depth(x, block_p, self.stochastic_depth_prob)

        x = self.ln_post(x)
        return x


class CheckpointedStochasticTextDecoder(StochasticDepthMixin, TextDecoder):
    """
    CheckpointedStochasticTextDecoder, which contains stochastic depth and checkpointing, also used in the original Whisper model.
    See: https://arxiv.org/abs/1603.09382
    """

    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, stochastic_depth_prob: float):
        super().__init__(n_vocab, n_ctx, n_state, n_head, n_layer)
        self.stochastic_depth_prob = stochastic_depth_prob

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
            x = self.stochastic_depth(x, block_p, self.stochastic_depth_prob)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits


def load_model_and_set_heads(
    model: Whisper,
    name: str,
    device: Union[str, torch.device] = "cpu",
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


def register_deep_spec_augment_hooks(
    model: Whisper,
    time_mask_param: int,
    freq_mask_param: int,
    layer_indices: Optional[list] = None,
) -> None:
    time_mask = T.TimeMasking(time_mask_param=time_mask_param)
    freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)

    def _norm_hook(module, input, output):
        if module.training:
            # output here is normalized: shape (batch, seq_len, embed_dim)
            # Convert to (batch, embed_dim, seq_len) for masking
            x = output.permute(0, 2, 1)
            x = time_mask(x)  # time masking on normalized features
            x = freq_mask(x)  # frequency masking on normalized features
            return x.permute(0, 2, 1)
        return output

    if layer_indices is None:
        # Skip the final encoder block to allow the model to recover from the
        # augmentation. ``range`` would include the last index, so we remove it
        # here to avoid raising an error below.
        layer_indices = range(len(model.encoder.blocks) - 1)

    for idx in layer_indices:
        if idx >= len(model.encoder.blocks):
            raise ValueError(f"Layer index {idx} out of range")

        if idx == len(model.encoder.blocks) - 1:
            # Skip the last layer entirely
            continue

        # Register hook to the attention layer norm of each block
        layer_norm = model.encoder.blocks[idx].attn_ln
        layer_norm.register_forward_hook(_norm_hook)
