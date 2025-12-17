import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio.transforms as T
from audiomentations import Compose
from datasets import Dataset as HU_Dataset
from numpy import ndarray
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from whisper.audio import CHUNK_LENGTH, N_FRAMES, N_SAMPLES, log_mel_spectrogram
from whisper.tokenizer import Tokenizer
from tqdm import tqdm

from whisper_finetune.data.utils import (
    ExtremesFrequencyMasking,
    TimeWarpAugmenter,
    pad_or_trim,
)
from whisper_finetune.model.augment import (
    get_audio_augments_advanced,
    get_audio_augments_baseline,
    get_audio_augments_office,
)


@dataclass
class Record:
    """
    A single training instance for Whisper.
    `text` can include timestamps in the format of <|0.00|>.
    """

    audio_array: ndarray
    text: str  # text including timestamps
    language: str = ""
    prompt: str = ""  # previous text including timestamps


class AudioDataset(Dataset):
    def __init__(
        self,
        hu_dataset: HU_Dataset,
        tokenizer: Tokenizer,
        device: Optional[torch.device] = None,  # CUDA does not allow for multiprocessing.
        no_timestamp_training: bool = False,
        n_mels: int = 80,
        max_prompt_length: int = 223,  # The maximum number of tokens to use for the prompt
        prompt_use_rate: float = 0.5,
        no_timestamps_rate: float = 0.5,
        spec_augment: bool = False,
        spec_augment_params: Optional[dict] = None,
        extremes_spec_augment: bool = False,
        extremes_spec_augment_params: Optional[dict] = None,
        apply_baseline_aug: bool = False,
        apply_office_aug: bool = False,
        apply_advanced_aug: bool = False,
        time_stretch_min_rate: float = 0.8,
        time_stretch_max_rate: float = 1.25,
        bpe_dropout: float = 0.0,
    ) -> None:
        """
        Initializes the class with the given parameters.

        Args:
            hu_dataset (HU_Dataset): The dataset to use.
            tokenizer (Tokenizer): The tokenizer to use.
            device (Optional[torch.device], optional): The device to use. Defaults to None.
            no_timestamp_training (bool, optional): Whether to use no timestamps for training. Defaults to False.
            n_mels (int, optional): The number of mel filters to use. Defaults to 80.
            max_prompt_length (int, optional): The maximum number of tokens to use for the prompt. Defaults to 223.
            prompt_use_rate (float, optional): The rate at which to use prompts. Defaults to 0.5.
            no_timestamps_rate (float, optional): The rate at which to use no timestamps. Defaults to 0.5.
            spec_augment (bool, optional): Whether to use spectrogram augmentation. Defaults to False.
            spec_augment_params (Optional[dict], optional): The parameters for spectrogram augmentation. Defaults to None.
            extremes_spec_augment (bool, optional): Whether to apply masking on extreme frequency ranges. Defaults to False.
            extremes_spec_augment_params (Optional[dict], optional): Parameters for extreme frequency masking (``low_freq_range`` and ``high_freq_range``). Defaults to None.
            apply_office_aug (bool, optional): Whether to apply office augmentation. Defaults to False.
            apply_baseline_aug (bool, optional): Whether to apply baseline augmentation. Defaults to False.
            apply_advanced_aug (bool, optional): Whether to apply advanced augmentation (filters, pitch, gain). Defaults to False.
            time_stretch_min_rate (float, optional): Minimum time-stretch rate. Defaults to 0.8.
            time_stretch_max_rate (float, optional): Maximum time-stretch rate. Defaults to 1.25.
            bpe_dropout (float, optional): The rate at which to drop BPE tokens. Defaults to 0.0.

        Returns:
            None

        Raises:
            ValueError: If the dataset does not contain the required columns.

        """
        self.hu_dataset = hu_dataset
        self.tokenizer = tokenizer
        self.n_mels = n_mels
        self.device = device
        self.no_timestamp_training = no_timestamp_training
        self.max_prompt_length = max_prompt_length
        self.prompt_use_rate = prompt_use_rate
        self.no_timestamps_rate = no_timestamps_rate
        self.spec_augment = spec_augment
        self.extremes_spec_augment = extremes_spec_augment
        self.apply_baseline_aug = apply_baseline_aug
        self.apply_office_aug = apply_office_aug
        self.apply_advanced_aug = apply_advanced_aug
        self.time_stretch_min_rate = time_stretch_min_rate
        self.time_stretch_max_rate = time_stretch_max_rate
        self.bpe_dropout = bpe_dropout
        if spec_augment:
            self.time_masking = T.TimeMasking(time_mask_param=spec_augment_params["time_mask_param"])
            self.freq_masking = T.FrequencyMasking(freq_mask_param=spec_augment_params["freq_mask_param"])
            self.time_warping = TimeWarpAugmenter(W=spec_augment_params["time_warp_w"])
        else:
            self.time_masking = None
            self.freq_masking = None
            self.time_warping = None

        if extremes_spec_augment:
            self.extreme_freq_masking = ExtremesFrequencyMasking(
                low_freq_range=extremes_spec_augment_params["low_freq_range"],
                high_freq_range=extremes_spec_augment_params["high_freq_range"],
            )
        else:
            self.extreme_freq_masking = None
        if self.apply_office_aug or self.apply_baseline_aug or self.apply_advanced_aug:
            aud_aug = []
            if self.apply_baseline_aug:
                aud_aug.append(get_audio_augments_baseline(
                    min_rate=self.time_stretch_min_rate,
                    max_rate=self.time_stretch_max_rate
                ))
            if self.apply_office_aug:
                aud_aug.append(get_audio_augments_office())
            if self.apply_advanced_aug:
                aud_aug.append(get_audio_augments_advanced())
            self.aud_augment = Compose(aud_aug)
        else:
            self.aud_augment = None

        self.num_frames_per_second = N_FRAMES / CHUNK_LENGTH
        # timestamps tokens are from <|0.00|> to <|30.00|> with a step of 0.02
        self.timestamp_pattern = re.compile(r"(<\|[123]?[0-9]\.[0-9][0-9]\|>)")
        self.model_n_text_ctx = 448
        self.hu_dataset = self.hu_dataset.with_format(type="torch")

        # Some checks
        assert np.intersect1d(self.hu_dataset.column_names, ["audio", "text", "language"]).size == 3

        # Some more data checks!
        # Often, some data is corrupted, 1-2 corrupted examples out of TB of data should not hurt training.
        self.valid_indices = []
        for i in tqdm(range(len(self.hu_dataset)), desc="Validating dataset"):
            try:
                record = self.hu_dataset[i]
                array = record["audio"]["array"]
                x = torch.as_tensor(array)
                # Check if text is valid
                assert isinstance(record["text"], str), f"Text is not a string: {record['text']}"
                # Append valid index
                self.valid_indices.append(i)
            except Exception as e:
                print(f"Skipping index {i} due to error: {e}")

        # Print the number of valid records
        if len(self.valid_indices) != len(self.hu_dataset):
            print(f"Filtered out {len(self.hu_dataset) - len(self.valid_indices)} invalid records from the dataset.")
            print(f"Invalid indices: {[i for i in range(len(self.hu_dataset)) if i not in self.valid_indices]}")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _get_prompt_tokens(self, record: Record, no_timestamps: bool) -> List[int]:
        if torch.rand(1).item() < self.prompt_use_rate and len(record["prompt"]) > 0:
            if no_timestamps:
                prompt_tokens = self._encode_text_without_timestamps(record["prompt"])[-self.max_prompt_length :]
            else:
                prompt_tokens = self._encode_text_with_timestamps(record["prompt"])[-self.max_prompt_length :]
            prompt_tokens = [self.tokenizer.sot_prev] + prompt_tokens
        else:
            prompt_tokens = []

        return prompt_tokens

    def _get_special_tokens(self, is_text_empty: bool, language: str, no_timestamps: bool) -> List[int]:
        if is_text_empty:
            special_tokens = [self.tokenizer.sot, self.tokenizer.no_speech]
        else:
            special_tokens = [
                self.tokenizer.sot,
                self.tokenizer.special_tokens[f"<|{language}|>"],
                self.tokenizer.special_tokens["<|transcribe|>"],
            ]
            if no_timestamps:
                special_tokens.append(self.tokenizer.no_timestamps)

        return special_tokens

    def _encode_text_without_timestamps(self, text: str) -> List[int]:
        """Encode text without timestamps by removing timestamps."""
        parts = self.timestamp_pattern.split(text)
        parts = [token for token in parts if token != ""]
        tokens = []
        for part in parts:
            if self.timestamp_pattern.fullmatch(part) is not None:
                timestamp = float(part[2:-2])

                # timestamp must be in the range [0, 30] and be a multiple of 0.02 seconds
                if timestamp < 0 or timestamp > 30 or round(timestamp * 100) % 2 != 0:
                    raise ValueError(f"Invalid timestamp: {timestamp}")
                continue
            else:
                tokens.extend(self.tokenizer.encode(part, dropout_prob=self.bpe_dropout))

        return tokens

    def _encode_text_with_timestamps(self, text: str) -> List[int]:
        parts = self.timestamp_pattern.split(text)
        parts = [token for token in parts if token != ""]
        tokens = []
        for part in parts:
            if self.timestamp_pattern.fullmatch(part) is not None:
                timestamp = float(part[2:-2])

                # timestamp must be in the range [0, 30] and be a multiple of 0.02 seconds
                if timestamp < 0 or timestamp > 30 or round(timestamp * 100) % 2 != 0:
                    raise ValueError(f"Invalid timestamp: {timestamp}")

                token = self.tokenizer.timestamp_begin + round(timestamp * 100) // 2
                tokens.append(token)
            else:
                tokens.extend(self.tokenizer.encode(part, dropout_prob=self.bpe_dropout))

        return tokens

    def _get_partial_segment_start(self, tokens: List[int]) -> Optional[float]:
        """If at the end there are two timestamps, use the last one to cut the audio.
        And then zero pad it in the audio-dimension, so that the model learns about silence."""
        if (
            len(tokens) >= 2
            and tokens[-2] >= self.tokenizer.timestamp_begin
            and tokens[-1] >= self.tokenizer.timestamp_begin
        ):  # if the last token is a start time token
            return (tokens[-1] - self.tokenizer.timestamp_begin) * 0.02
        else:
            return None

    def _get_text_tokens(self, text: str, no_timestamps: bool) -> Tuple[List[int], Optional[float]]:
        text_tokens = self._encode_text_with_timestamps(text)
        next_partial_segment_start = self._get_partial_segment_start(text_tokens)
        if no_timestamps:
            text_tokens = list(filter(lambda x: x < self.tokenizer.timestamp_begin, text_tokens))

        return text_tokens, next_partial_segment_start

    def _calculate_mel(
        self, audio_array: ndarray, next_partial_segment_start: Optional[float], no_timestamps: bool
    ) -> torch.Tensor:
        if self.aud_augment is not None:
            audio_array = self.aud_augment(audio_array, sample_rate=16000)
        mel = log_mel_spectrogram(audio_array, n_mels=self.n_mels, device=self.device)
        if no_timestamps and next_partial_segment_start is not None:
            mel = mel[:, : int(next_partial_segment_start * self.num_frames_per_second)]
        if mel.shape[1] != N_FRAMES:
            mel = pad_or_trim(mel, N_FRAMES)

        if self.spec_augment:
            mel = self.time_warping(mel)
            mel = self.time_masking(mel)
            mel = self.freq_masking(mel)

        if self.extreme_freq_masking:
            mel = self.extreme_freq_masking(mel)

        return mel

    def _construct_decoder_output(
        self, prompt_tokens: List[int], special_tokens: List[int], text_tokens: List[int]
    ) -> List[int]:
        if len(prompt_tokens) == 0:
            decoder_output = special_tokens[1:] + text_tokens + [self.tokenizer.eot]
        else:
            decoder_output = (
                # Mask out the training loss for predicting the prompt tokens. We use "-100" as the
                # default value for the `ignore_index` parameter in
                # `torch.nn.functional.cross_entropy()`. However, we do not mask out the loss for
                # predicting the sot token because our experiment indicates that the original
                # Whisper model assigns a high probability to the sot token after prompt tokens.
                [-100] * (len(prompt_tokens) - 1)
                + special_tokens
                + text_tokens
                + [self.tokenizer.eot]
            )
        return decoder_output

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = self.valid_indices[index]
        record = self.hu_dataset[index]
        no_timestamps = self.no_timestamp_training or torch.rand(1).item() < self.no_timestamps_rate

        prompt_tokens = self._get_prompt_tokens(record, no_timestamps)
        text_tokens, next_partial_segment_start = self._get_text_tokens(record["text"], no_timestamps)
        is_text_empty = len(text_tokens) == 0
        special_tokens = self._get_special_tokens(is_text_empty, record["language"], no_timestamps)

        decoder_input = prompt_tokens + special_tokens + text_tokens
        if len(decoder_input) > self.model_n_text_ctx:
            print(f"Input is too long: {record} (length: {len(decoder_input)}). Shortening... the prompt")
            too_long_by = len(decoder_input) - self.model_n_text_ctx
            prompt_tokens = prompt_tokens[:-too_long_by]
            decoder_input = prompt_tokens + special_tokens + text_tokens
            if len(decoder_input) > self.model_n_text_ctx:
                print(f"Input is still too long: {record} (length: {len(decoder_input)}).")

        decoder_output = self._construct_decoder_output(prompt_tokens, special_tokens, text_tokens)
        audio_arr = record["audio"]["array"]
        del record

        # Pad in audio domain, not spectrogram domain.
        # https://github.com/openai/whisper/discussions/838#discussioncomment-5233715
        audio_arr = np.pad(audio_arr, (0, N_SAMPLES - audio_arr.shape[0]), "constant")

        mel = self._calculate_mel(audio_arr, next_partial_segment_start, no_timestamps)
        if index == 0:
            print(f"Mel shape: {mel.shape}")
            print(
                f"Mel Spectrogram stats: mean={mel.mean()}, std={mel.std()}, min={mel.min()}, max={mel.max()}. N NANs: {torch.isnan(mel).sum()}"
            )

        return (
            mel,
            torch.tensor(decoder_input, dtype=torch.int64),
            torch.tensor(decoder_output, dtype=torch.int64),
        )


def collate_fn(data):
    x, y_in, y_out = zip(*data)
    x = pad_sequence(x, batch_first=True, padding_value=0)
    y_in = pad_sequence(y_in, batch_first=True, padding_value=0)
    y_out = pad_sequence(y_out, batch_first=True, padding_value=-100)
    return x, y_in, y_out


class WarmupDatasetSampler(torch.utils.data.Sampler):
    """
    A sampler that uses only warmup dataset indices during warmup phase,
    then switches to sampling from all datasets.
    
    This is useful for curriculum learning where you want to start training
    on a specific "easier" or "cleaner" dataset before mixing in other data.
    
    Args:
        warmup_indices: Indices of samples belonging to the warmup dataset
        all_indices: All valid indices in the dataset
        warmup_steps: Number of warmup steps (in batches, not samples)
        batch_size: Batch size for calculating when warmup ends
        shuffle: Whether to shuffle indices
    """
    
    def __init__(
        self,
        warmup_indices: List[int],
        all_indices: List[int],
        warmup_steps: int,
        batch_size: int,
        shuffle: bool = True,
    ):
        self.warmup_indices = list(warmup_indices)
        self.all_indices = list(all_indices)

        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if len(self.all_indices) == 0:
            raise ValueError("all_indices must be non-empty")
        if len(self.warmup_indices) == 0 and warmup_steps > 0:
            raise ValueError("warmup_indices must be non-empty when warmup_steps > 0")

        self.warmup_steps = int(warmup_steps)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        
        # How many *samples* to draw before switching from warmup-only sampling.
        self.warmup_samples = self.warmup_steps * self.batch_size
        
        # Current position tracker
        self._current_sample = 0
        self._warmup_exhausted = False
        
        print(f"WarmupDatasetSampler initialized:")
        print(f"  - Warmup indices: {len(self.warmup_indices)}")
        print(f"  - All indices: {len(self.all_indices)}")
        print(f"  - Warmup steps: {warmup_steps} ({self.warmup_samples} samples)")
    
    def __iter__(self):
        # Reset sample counter at start of each epoch
        self._current_sample = 0
        self._warmup_exhausted = False
        
        while True:
            in_warmup = self._current_sample < self.warmup_samples
            indices = self.warmup_indices if in_warmup else self.all_indices

            indices = indices.copy()
            if self.shuffle:
                np.random.shuffle(indices)

            for idx in indices:
                # Switch exactly after emitting warmup_samples items.
                if (not self._warmup_exhausted) and self._current_sample >= self.warmup_samples:
                    print(
                        f"\n>>> Warmup complete after {self.warmup_samples} samples. Now sampling from all datasets.\n"
                    )
                    self._warmup_exhausted = True

                yield idx
                self._current_sample += 1
    
    def __len__(self):
        # Return total dataset size (for progress bars etc)
        return len(self.all_indices)


def get_dataset_boundary_indices(
    dataset_sizes: List[int],
) -> List[Tuple[int, int]]:
    """
    Given sizes of concatenated datasets, return (start, end) index tuples for each.
    
    Example:
        dataset_sizes = [1000, 500, 2000]
        returns: [(0, 1000), (1000, 1500), (1500, 3500)]
    """
    boundaries = []
    start = 0
    for size in dataset_sizes:
        boundaries.append((start, start + size))
        start += size
    return boundaries


def get_dataloader(
    hu_dataset: HU_Dataset,
    tokenizer: Tokenizer,
    batch_size: int = 1,
    n_mels: int = 80,
    sampler: Optional[torch.utils.data.sampler.Sampler] = None,
    device: Optional[torch.device] = None,  # Does not allow for multiprocessing.
    no_timestamp_training: bool = False,
    max_prompt_length: int = 223,
    prompt_use_rate: float = 0.5,
    no_timestamps_rate: float = 0.5,
    shuffle: bool = True,
    num_workers: int = 0,
    spec_augment: bool = False,
    spec_augment_params: Optional[dict] = None,
    extremes_spec_augment: bool = False,
    extremes_spec_augment_params: Optional[dict] = None,
    apply_baseline_aug: bool = False,
    apply_office_aug: bool = False,
    apply_advanced_aug: bool = False,
    time_stretch_min_rate: float = 0.8,
    time_stretch_max_rate: float = 1.25,
    bpe_dropout: float = 0.0,
) -> DataLoader:
    print(f"Found {len(hu_dataset)} records in the dataset.")
    dataset = AudioDataset(
        hu_dataset,
        tokenizer,
        device=device,
        no_timestamp_training=no_timestamp_training,
        n_mels=n_mels,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=prompt_use_rate,
        no_timestamps_rate=no_timestamps_rate,
        spec_augment=spec_augment,
        spec_augment_params=spec_augment_params,
        extremes_spec_augment=extremes_spec_augment,
        extremes_spec_augment_params=extremes_spec_augment_params,
        apply_baseline_aug=apply_baseline_aug,
        apply_office_aug=apply_office_aug,
        apply_advanced_aug=apply_advanced_aug,
        time_stretch_min_rate=time_stretch_min_rate,
        time_stretch_max_rate=time_stretch_max_rate,
        bpe_dropout=bpe_dropout,
    )
    
    # PyTorch DataLoader doesn't allow both sampler and shuffle=True
    if sampler is not None:
        shuffle = False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
