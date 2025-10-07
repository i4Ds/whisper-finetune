from collections import defaultdict
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset
from librosa.feature.inverse import mel_to_audio
from whisper.audio import HOP_LENGTH, N_FFT, N_SAMPLES


class TimeWarpAugmenter:
    def __init__(self, W=50):
        """
        Initialize the TimeWarpAugmenter with the strength of warp (W).
        """
        self.W = W

    def __call__(self, specs):
        """
        Apply time warp augmentation when the class is called.

        param:
        specs: spectrogram of size (batch, channel, freq_bin, length)
        """
        if not torch.is_tensor(specs):
            specs = torch.from_numpy(specs)
        if specs.dim() < 2 or specs.dim() > 3:
            raise ValueError("You sure it's a Spectrogram?")
        if specs.dim() == 2:
            # Add dummy batch.
            specs = torch.unsqueeze(specs, dim=0)
        warped = self.time_warp(specs, self.W)
        return warped.squeeze(0)

    @staticmethod
    def h_poly(t):
        tt = t.unsqueeze(-2) ** torch.arange(4, device=t.device).view(-1, 1)
        A = torch.tensor([[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]], dtype=t.dtype, device=t.device)
        return A @ tt

    @staticmethod
    def hspline_interpolate_1D(x, y, xs):
        """
        Input x and y must be of shape (batch, n) or (n)
        """
        m = (y[..., 1:] - y[..., :-1]) / (x[..., 1:] - x[..., :-1])
        m = torch.cat([m[..., [0]], (m[..., 1:] + m[..., :-1]) / 2, m[..., [-1]]], -1)
        idxs = torch.searchsorted(x[..., 1:], xs)
        # print(torch.abs(x.take_along_dim(idxs+1, dim=-1) - x.gather(dim=-1, index=idxs+1)))
        dx = x.gather(dim=-1, index=idxs + 1) - x.gather(dim=-1, index=idxs)
        hh = TimeWarpAugmenter.h_poly((xs - x.gather(dim=-1, index=idxs)) / dx)
        return (
            hh[..., 0, :] * y.gather(dim=-1, index=idxs)
            + hh[..., 1, :] * m.gather(dim=-1, index=idxs) * dx
            + hh[..., 2, :] * y.gather(dim=-1, index=idxs + 1)
            + hh[..., 3, :] * m.gather(dim=-1, index=idxs + 1) * dx
        )
        # dx = (x.take_along_dim(idxs+1, dim=-1) - x.take_along_dim(idxs, dim=-1))
        # hh = h_poly((xs - x.take_along_dim(idxs, dim=-1)) / dx)
        # return hh[...,0,:] * y.take_along_dim(idxs, dim=-1) \
        #     + hh[...,1,:] * m.take_along_dim(idxs, dim=-1) * dx \
        #     + hh[...,2,:] * y.take_along_dim(idxs+1, dim=-1) \
        #     + hh[...,3,:] * m.take_along_dim(idxs+1, dim=-1) * dx

    def time_warp(self, specs, W=80):
        """
        Timewarp augmentation by https://github.com/IMLHF/SpecAugmentPyTorch/blob/master/spec_augment_pytorch.py

        param:
        specs: spectrogram of size (batch, channel, freq_bin, length)
        W: strength of warp
        """
        device = specs.device
        specs = specs.unsqueeze(0)  # Add dim for channels
        batch_size, _, num_rows, spec_len = specs.shape

        warp_p = torch.randint(W, spec_len - W, (batch_size,), device=device)

        # Uniform distribution from (0,W) with chance to be up to W negative
        # warp_d = torch.randn(1)*W # Not using this since the paper author make random number with uniform distribution
        warp_d = torch.randint(-W, W, (batch_size,), device=device)
        # print("warp_d", warp_d)
        x = torch.stack(
            [
                torch.tensor([0], device=device).expand(batch_size),
                warp_p,
                torch.tensor([spec_len - 1], device=device).expand(batch_size),
            ],
            1,
        )
        y = torch.stack(
            [
                torch.tensor([-1.0], device=device).expand(batch_size),
                (warp_p - warp_d) * 2 / (spec_len - 1.0) - 1.0,
                torch.tensor([1.0], device=device).expand(batch_size),
            ],
            1,
        )
        # print((warp_p-warp_d)*2/(spec_len-1.)-1.)

        # Interpolate from 3 points to spec_len
        xs = torch.linspace(0, spec_len - 1, spec_len, device=device).unsqueeze(0).expand(batch_size, -1)
        ys = TimeWarpAugmenter.hspline_interpolate_1D(x, y, xs)

        grid = torch.cat(
            (
                ys.view(batch_size, 1, -1, 1).expand(-1, num_rows, -1, -1),
                torch.linspace(-1, 1, num_rows, device=device).view(-1, 1, 1).expand(batch_size, -1, spec_len, -1),
            ),
            -1,
        )

        return torch.nn.functional.grid_sample(specs, grid, align_corners=True).squeeze(0)  # Remove dim for channels


class ExtremesFrequencyMasking:
    """Mask frequency bins starting from both spectrum extremes."""

    def __init__(self, low_freq_range: int = 10, high_freq_range: int = 10):
        """Create the transform.

        Args:
            low_freq_range: Maximum number of lowest bins eligible for masking.
            high_freq_range: Maximum number of highest bins eligible for masking.
        """
        self.low_freq_range = low_freq_range
        self.high_freq_range = high_freq_range

    def __call__(self, specs: torch.Tensor) -> torch.Tensor:
        """Apply the masking.

        A single random ratio is drawn for each sample. The same ratio is used
        for both the low and high frequency ranges. Masking always starts from
        the edges of the spectrogram.
        """
        if not torch.is_tensor(specs):
            specs = torch.tensor(specs)
        single = False
        if specs.dim() == 2:
            specs = specs.unsqueeze(0)
            single = True

        batch, n_mels, _ = specs.shape

        for b in range(batch):
            r = torch.rand(1).item()

            low_mask_len = int(round(r * self.low_freq_range))
            if low_mask_len > 0:
                max_end = min(low_mask_len, n_mels)
                specs[b, :max_end, :] = 0

            high_mask_len = int(round(r * self.high_freq_range))
            if high_mask_len > 0:
                start = max(n_mels - high_mask_len, 0)
                specs[b, start:, :] = 0

        if single:
            specs = specs.squeeze(0)
        return specs


def process_dataset(dataset_names, select_n_per_ds, split_name, groupby_col, print_examples=False, example_count=5):
    """
    Function to process individual datasets with optional groupby sampling,
    filtering out entries with text < 30 chars or audio < 6 seconds, and optionally printing examples.

    Args:
    - dataset_names (list): List of dataset names to process.
    - select_n_per_ds (list): List of N values for sampling from each dataset.
    - split_name (str): The split of the dataset to use.
    - groupby_col (list): Column name to use for groupby sampling.
    - print_examples (bool): If True, print a few filtered examples.
    - example_count (int): Number of examples to print.

    Returns:
    - concatenated_dataset: A concatenated dataset of all processed datasets.
    """
    processed_datasets = []

    # Validate lengths
    if not (len(select_n_per_ds) == len(groupby_col) or len(select_n_per_ds) == len(dataset_names)):
        print(f"Warning! Check length of select_n_per_ds, groupby_col and dataset_names")

    for N, GROUPBYCOL, dataset_name in zip(select_n_per_ds, groupby_col, dataset_names):
        # Load
        dataset = load_dataset(dataset_name)
        if split_name not in dataset:
            print(f"Split name {split_name} not found in dataset {dataset_name}. Available splits: {list(dataset.keys())}")
            if 'train' in dataset:
                split_name = 'train'
                print(f"Defaulting to 'train' split.")
            else:
                split_name = list(dataset.keys())[0]
                print(f"Defaulting to first available split: {split_name}")
        dataset = dataset[split_name]

        print(f"Processing dataset: {dataset_name}")
        print(f"Original dataset size: {len(dataset)}")

        # Rename 'sentence' to 'text'
        if "sentence" in dataset.column_names:
            dataset = dataset.rename_column("sentence", "text")

        # Ensure 'language' column exists
        if "language" not in dataset.column_names:
            dataset = dataset.map(
                add_fixed_value, batched=True, fn_kwargs={"col_name": "language", "fixed_value": "de"}
            )

        # Sampling
        if N is not None:
            if GROUPBYCOL and GROUPBYCOL in dataset.column_names:
                print(f"Performing groupby sampling on column: {GROUPBYCOL}")
                groups = defaultdict(list)
                for idx, item in enumerate(dataset[GROUPBYCOL]):
                    groups[item].append(idx)
                selected_indices = []
                for group_indices in groups.values():
                    replace = len(group_indices) < N
                    selected_indices.extend(np.random.choice(group_indices, size=N, replace=replace))
            else:
                print("Performing regular random sampling")
                count = min(N, len(dataset))
                selected_indices = np.arange(count)
            dataset = dataset.select(selected_indices)
            print(f"Number of samples selected: {len(dataset)}")
        else:
            print("No sampling performed (N is None)")
            dataset = dataset

        processed_datasets.append(dataset)

    concatenated = concatenate_datasets(processed_datasets)
    print(f"Total rows in concatenated dataset: {len(concatenated)}")
    return concatenated


def add_fixed_value(batch, col_name, fixed_value):
    batch[col_name] = [fixed_value] * len(batch["text"])
    return batch


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to `length`, using the minimum value in the array for padding.
    This is particularly useful for spectrograms where padding with the minimum value (silence) is desired.
    As mentioned by Openai, we should zeropad in Audio, which is the minimum value of the melspectrogram
    in the melspectrogram-dimension.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            min_value = torch.min(array).item()
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes], value=min_value)
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            min_value = np.min(array)
            array = np.pad(array, pad_widths, constant_values=min_value)

    return array


def inverse_mel_to_audio(
    mel_spec: Union[torch.Tensor, np.array],
    sr: int = 16000,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    power: float = 10,
) -> np.ndarray:
    """
    Inverse a Mel spectrogram back to an audio waveform using the Griffin-Lim algorithm.
    The parameters are working for the data generation for whisper-large v2.

    Parameters:
    - mel_spec : torch.Tensor
        The Mel spectrogram as a PyTorch tensor.
    - sr : int, optional
        The sample rate of the audio (default is 16000 Hz).
    - n_fft : int, optional
        The number of FFT components (default is 400).
    - hop_length : int, optional
        The number of samples between successive frames (default is 160).
    - power : float, optional
        The power to raise the Mel spectrogram before inversion (default is 10).

    Returns:
    - audio : np.ndarray
        The reconstructed audio signal as a NumPy array.
    """
    # Convert the Mel spectrogram to a NumPy array and apply power
    if torch.is_tensor(mel_spec):
        mel_spec_np = mel_spec.numpy()
    mel_spec_power = np.power(mel_spec_np, power)

    # Invert the Mel spectrogram to audio using librosa
    audio = mel_to_audio(mel_spec_power, sr=sr, n_fft=n_fft, hop_length=hop_length)

    return audio
