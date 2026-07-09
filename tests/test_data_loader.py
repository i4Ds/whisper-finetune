"""
Tests for Whisper data loading and decoder target construction.
"""

import sys
import types

import pytest
import torch


def _install_whisper_stubs():
    """Provide minimal whisper modules so data_loader can be imported in unit tests."""
    if "whisper" in sys.modules:
        return

    try:
        import whisper.audio  # noqa: F401
        import whisper.tokenizer  # noqa: F401
        return
    except ImportError:
        pass

    whisper_module = types.ModuleType("whisper")

    audio_module = types.ModuleType("whisper.audio")
    audio_module.CHUNK_LENGTH = 30
    audio_module.HOP_LENGTH = 160
    audio_module.N_FFT = 400
    audio_module.N_FRAMES = 3000
    audio_module.N_SAMPLES = 480000
    audio_module.log_mel_spectrogram = lambda *args, **kwargs: None

    tokenizer_module = types.ModuleType("whisper.tokenizer")
    tokenizer_module.LANGUAGES = {"de": "german"}
    tokenizer_module.TO_LANGUAGE_CODE = {"german": "de"}
    tokenizer_module.Tokenizer = object

    whisper_module.audio = audio_module
    whisper_module.tokenizer = tokenizer_module

    sys.modules["whisper"] = whisper_module
    sys.modules["whisper.audio"] = audio_module
    sys.modules["whisper.tokenizer"] = tokenizer_module


_install_whisper_stubs()

from whisper_finetune.data import data_loader as data_loader_module
from whisper_finetune.data.data_loader import AudioDataset


class DummyTokenizer:
    def __init__(self):
        self.sot = 1
        self.no_speech = 2
        self.eot = 3
        self.sot_prev = 4
        self.no_timestamps = 5
        self.timestamp_begin = 100
        self.special_tokens = {
            "<|de|>": 6,
            "<|transcribe|>": 7,
        }


class DummyHFDataset:
    def __init__(self, records):
        self.records = records
        self.column_names = ["audio", "text", "language"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if isinstance(record, Exception):
            raise record
        return record

    def with_format(self, *args, **kwargs):
        return self


class TestNoSpeechTargets:
    def test_empty_text_uses_no_speech_special_tokens(self):
        dataset = AudioDataset.__new__(AudioDataset)
        dataset.tokenizer = DummyTokenizer()

        special_tokens = dataset._get_special_tokens(is_text_empty=True, language="de", no_timestamps=False)

        assert special_tokens == [
            dataset.tokenizer.sot,
            dataset.tokenizer.special_tokens["<|de|>"],
            dataset.tokenizer.special_tokens["<|transcribe|>"],
            dataset.tokenizer.no_speech,
        ]

    def test_empty_text_without_prompt_trains_no_speech_then_eot(self):
        dataset = AudioDataset.__new__(AudioDataset)
        dataset.tokenizer = DummyTokenizer()

        special_tokens = dataset._get_special_tokens(is_text_empty=True, language="de", no_timestamps=False)
        decoder_output = dataset._construct_decoder_output([], special_tokens, [])

        assert decoder_output == [
            dataset.tokenizer.special_tokens["<|de|>"],
            dataset.tokenizer.special_tokens["<|transcribe|>"],
            dataset.tokenizer.no_speech,
            dataset.tokenizer.eot,
        ]

    def test_empty_text_with_prompt_still_trains_no_speech_then_eot(self):
        dataset = AudioDataset.__new__(AudioDataset)
        dataset.tokenizer = DummyTokenizer()

        prompt_tokens = [dataset.tokenizer.sot_prev, 42, 43]
        special_tokens = dataset._get_special_tokens(is_text_empty=True, language="de", no_timestamps=False)
        decoder_output = dataset._construct_decoder_output(prompt_tokens, special_tokens, [])

        assert decoder_output == [
            -100,
            -100,
            dataset.tokenizer.sot,
            dataset.tokenizer.special_tokens["<|de|>"],
            dataset.tokenizer.special_tokens["<|transcribe|>"],
            dataset.tokenizer.no_speech,
            dataset.tokenizer.eot,
        ]

    def test_empty_text_with_no_timestamps_keeps_no_timestamps_prefix(self):
        dataset = AudioDataset.__new__(AudioDataset)
        dataset.tokenizer = DummyTokenizer()

        special_tokens = dataset._get_special_tokens(is_text_empty=True, language="de", no_timestamps=True)
        decoder_output = dataset._construct_decoder_output([], special_tokens, [])

        assert special_tokens == [
            dataset.tokenizer.sot,
            dataset.tokenizer.special_tokens["<|de|>"],
            dataset.tokenizer.special_tokens["<|transcribe|>"],
            dataset.tokenizer.no_timestamps,
            dataset.tokenizer.no_speech,
        ]
        assert decoder_output == [
            dataset.tokenizer.special_tokens["<|de|>"],
            dataset.tokenizer.special_tokens["<|transcribe|>"],
            dataset.tokenizer.no_timestamps,
            dataset.tokenizer.no_speech,
            dataset.tokenizer.eot,
        ]


class TestLazyInvalidRecordHandling:
    def test_load_valid_record_skips_invalid_examples_without_prescan(self):
        dataset = AudioDataset.__new__(AudioDataset)
        dataset.hu_dataset = DummyHFDataset(
            [
                {"audio": {"array": object()}, "text": "bad tensor conversion", "language": "de"},
                {"audio": {"array": [0.1, 0.2]}, "text": 123, "language": "de"},
                {"audio": {"array": [0.1, 0.2]}, "text": "ok", "language": "de"},
            ]
        )
        dataset.invalid_indices = set()
        dataset._logged_invalid_count = 0

        index, record = dataset._load_valid_record(0)

        assert index == 2
        assert record["text"] == "ok"
        assert dataset.invalid_indices == {0, 1}


class AdditiveTransform:
    def __init__(self, value, calls):
        self.value = value
        self.calls = calls

    def __call__(self, mel):
        self.calls.append(self.value)
        return mel + self.value


class TestSpecAugmentProbability:
    def _build_dataset(self, p):
        calls = []
        dataset = AudioDataset.__new__(AudioDataset)
        dataset.aud_augment = None
        dataset.n_mels = 80
        dataset.device = None
        dataset.num_frames_per_second = data_loader_module.N_FRAMES / data_loader_module.CHUNK_LENGTH
        dataset.spec_augment = True
        dataset.spec_augment_p = p
        dataset.time_warping = AdditiveTransform(1, calls)
        dataset.time_masking = AdditiveTransform(2, calls)
        dataset.freq_masking = AdditiveTransform(4, calls)
        dataset.extreme_freq_masking = None
        return dataset, calls

    def test_spec_augment_p_zero_skips_transforms(self, monkeypatch):
        dataset, calls = self._build_dataset(p=0.0)
        base_mel = torch.zeros(80, data_loader_module.N_FRAMES)
        monkeypatch.setattr(
            data_loader_module,
            "log_mel_spectrogram",
            lambda *args, **kwargs: base_mel.clone(),
        )

        mel = dataset._calculate_mel(
            torch.zeros(16000),
            next_partial_segment_start=None,
            no_timestamps=False,
        )

        assert calls == []
        assert torch.equal(mel, base_mel)

    def test_spec_augment_p_one_applies_all_transforms(self, monkeypatch):
        dataset, calls = self._build_dataset(p=1.0)
        base_mel = torch.zeros(80, data_loader_module.N_FRAMES)
        monkeypatch.setattr(
            data_loader_module,
            "log_mel_spectrogram",
            lambda *args, **kwargs: base_mel.clone(),
        )

        mel = dataset._calculate_mel(
            torch.zeros(16000),
            next_partial_segment_start=None,
            no_timestamps=False,
        )

        assert calls == [1, 2, 4]
        assert torch.equal(mel, base_mel + 7)

    def test_spec_augment_p_uses_probability_threshold(self, monkeypatch):
        dataset, _ = self._build_dataset(p=0.5)

        monkeypatch.setattr(
            data_loader_module.torch,
            "rand",
            lambda *args, **kwargs: torch.tensor([0.25]),
        )
        assert dataset._should_apply_spec_augment()

        monkeypatch.setattr(
            data_loader_module.torch,
            "rand",
            lambda *args, **kwargs: torch.tensor([0.75]),
        )
        assert not dataset._should_apply_spec_augment()

    def test_missing_spec_augment_p_defaults_to_always_apply(self):
        dataset = AudioDataset(
            DummyHFDataset([]),
            DummyTokenizer(),
            spec_augment=True,
            spec_augment_params={
                "time_mask_param": 100,
                "freq_mask_param": 43,
                "time_warp_w": 80,
            },
        )

        assert dataset.spec_augment_p == 1.0

    def test_invalid_spec_augment_p_raises(self):
        with pytest.raises(ValueError, match="spec_augment p must be between 0 and 1"):
            AudioDataset(
                DummyHFDataset([]),
                DummyTokenizer(),
                spec_augment=True,
                spec_augment_params={
                    "time_mask_param": 100,
                    "freq_mask_param": 43,
                    "time_warp_w": 80,
                    "p": 1.1,
                },
            )
