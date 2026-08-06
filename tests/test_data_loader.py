"""
Tests for Whisper data loading and decoder target construction.
"""

from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

FIXTURE_AUDIO_PATH = Path(__file__).parent / "fixtures" / "28210.mp3"
FIXTURE_TRANSCRIPT = (
    '<|0.00|> Für die Glaubwürdigkeit der Union ist es unverzichtbar, dass Wort und Tat übereinstimmen.'
    '<|4.60|><|7.18|> Zweifellos ist Integrität im Online-spielesektor extrem wichtig.'
    '<|10.58|><|11.30|> Die "Sir Tristram" war eines von sechs Landungsschiffen der Round-Table-Klasse.'
    "<|16.10|><|16.12|> Nordöstlich befindet sich der Stadtteil Opladen."
    "<|18.40|><|20.08|> Entscheidend sind letztendlich die Aktivitäten, die auf den Wegen durchgeführt werden."
    "<|25.58|>"
)
FIXTURE_TRANSCRIPT_WITH_PARTIAL_SEGMENT_START = FIXTURE_TRANSCRIPT + "<|26.00|>"


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

    def encode(self, text, dropout_prob=0.0):
        return [10 + (ord(char) % 80) for char in text]


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


class TestTimestampAudioPaddingBehavior:
    def _build_dataset(self):
        dataset = AudioDataset(
            DummyHFDataset([]),
            DummyTokenizer(),
        )
        return dataset

    def test_no_timestamps_strips_single_final_timestamp_without_partial_cut(self):
        dataset = self._build_dataset()

        text_tokens, next_partial_segment_start = dataset._get_text_tokens(
            "<|0.00|> Text.<|3.64|><|3.66|> More text.<|25.58|>",
            no_timestamps=True,
        )

        assert next_partial_segment_start is None
        assert text_tokens
        assert all(token < dataset.tokenizer.timestamp_begin for token in text_tokens)

    def test_no_timestamps_detects_double_final_timestamp_for_partial_cut(self):
        dataset = self._build_dataset()

        text_tokens, next_partial_segment_start = dataset._get_text_tokens(
            "<|0.00|> Text.<|24.28|><|24.94|>",
            no_timestamps=True,
        )

        assert next_partial_segment_start == pytest.approx(24.94)
        assert text_tokens
        assert all(token < dataset.tokenizer.timestamp_begin for token in text_tokens)

    def test_timestamp_training_keeps_timestamps_but_does_not_cut_audio(self, monkeypatch):
        dataset = self._build_dataset()
        base_mel = torch.arange(data_loader_module.N_FRAMES).repeat(80, 1).float()
        monkeypatch.setattr(
            data_loader_module,
            "log_mel_spectrogram",
            lambda *args, **kwargs: base_mel.clone(),
        )

        text_tokens, next_partial_segment_start = dataset._get_text_tokens(
            "<|0.00|> Text.<|24.28|><|24.94|>",
            no_timestamps=False,
        )
        mel = dataset._calculate_mel(
            torch.zeros(16000),
            next_partial_segment_start=next_partial_segment_start,
            no_timestamps=False,
        )

        assert next_partial_segment_start == pytest.approx(24.94)
        assert any(token >= dataset.tokenizer.timestamp_begin for token in text_tokens)
        assert torch.equal(mel, base_mel)

    def test_no_timestamps_cuts_and_pads_when_partial_segment_start_is_set(self, monkeypatch):
        dataset = self._build_dataset()
        base_mel = torch.arange(data_loader_module.N_FRAMES).repeat(80, 1).float()
        monkeypatch.setattr(
            data_loader_module,
            "log_mel_spectrogram",
            lambda *args, **kwargs: base_mel.clone(),
        )

        mel = dataset._calculate_mel(
            torch.zeros(16000),
            next_partial_segment_start=24.94,
            no_timestamps=True,
        )

        cut_frame = int(24.94 * dataset.num_frames_per_second)
        assert mel.shape == base_mel.shape
        assert torch.equal(mel[:, :cut_frame], base_mel[:, :cut_frame])
        assert torch.equal(mel[:, cut_frame:], torch.zeros_like(mel[:, cut_frame:]))

    def test_no_timestamps_with_single_final_timestamp_uses_normal_audio_path(self, monkeypatch):
        dataset = self._build_dataset()
        base_mel = torch.arange(data_loader_module.N_FRAMES).repeat(80, 1).float()
        monkeypatch.setattr(
            data_loader_module,
            "log_mel_spectrogram",
            lambda *args, **kwargs: base_mel.clone(),
        )

        mel = dataset._calculate_mel(
            torch.zeros(16000),
            next_partial_segment_start=None,
            no_timestamps=True,
        )

        assert torch.equal(mel, base_mel)


@pytest.mark.slow
@pytest.mark.integration
class TestFixtureAudioIntegration:
    def _load_fixture_audio_array(self):
        torchaudio = pytest.importorskip("torchaudio")
        waveform, sample_rate = torchaudio.load(FIXTURE_AUDIO_PATH)

        assert sample_rate == 16000
        assert waveform.shape[0] == 1
        return waveform.squeeze(0).numpy()

    def _build_dataset(self, audio_array, no_timestamp_training, text=FIXTURE_TRANSCRIPT):
        return AudioDataset(
            DummyHFDataset(
                [
                    {
                        "audio": {"array": audio_array},
                        "text": text,
                        "language": "de",
                    }
                ]
            ),
            DummyTokenizer(),
            no_timestamp_training=no_timestamp_training,
            no_timestamps_rate=0.0,
            prompt_use_rate=0.0,
        )

    def test_single_final_timestamp_no_timestamp_training_keeps_normal_audio_path(self, monkeypatch):
        audio_array = self._load_fixture_audio_array()
        dataset = self._build_dataset(audio_array, no_timestamp_training=True)
        base_mel = torch.arange(data_loader_module.N_FRAMES).repeat(80, 1).float()
        captured = {}

        def fake_log_mel_spectrogram(audio, *args, **kwargs):
            captured["audio"] = np.asarray(audio).copy()
            return base_mel.clone()

        monkeypatch.setattr(data_loader_module, "log_mel_spectrogram", fake_log_mel_spectrogram)

        mel, decoder_input, decoder_output = dataset[0]

        assert mel.shape == (80, data_loader_module.N_FRAMES)
        assert torch.equal(mel, base_mel)
        assert captured["audio"].shape[0] == data_loader_module.N_SAMPLES
        np.testing.assert_allclose(captured["audio"][: audio_array.shape[0]], audio_array)
        assert dataset.tokenizer.no_timestamps in decoder_input.tolist()
        assert all(token < dataset.tokenizer.timestamp_begin for token in decoder_input.tolist())
        assert all(token < dataset.tokenizer.timestamp_begin for token in decoder_output.tolist())

    def test_double_final_timestamp_no_timestamp_training_cuts_and_pads_mel(self, monkeypatch):
        audio_array = self._load_fixture_audio_array()
        dataset = self._build_dataset(
            audio_array,
            no_timestamp_training=True,
            text=FIXTURE_TRANSCRIPT_WITH_PARTIAL_SEGMENT_START,
        )
        base_mel = torch.arange(data_loader_module.N_FRAMES).repeat(80, 1).float()

        monkeypatch.setattr(
            data_loader_module,
            "log_mel_spectrogram",
            lambda *args, **kwargs: base_mel.clone(),
        )

        mel, decoder_input, decoder_output = dataset[0]

        cut_frame = int(26.00 * dataset.num_frames_per_second)
        assert mel.shape == (80, data_loader_module.N_FRAMES)
        assert torch.equal(mel[:, :cut_frame], base_mel[:, :cut_frame])
        assert torch.equal(mel[:, cut_frame:], torch.zeros_like(mel[:, cut_frame:]))
        assert dataset.tokenizer.no_timestamps in decoder_input.tolist()
        assert all(token < dataset.tokenizer.timestamp_begin for token in decoder_input.tolist())
        assert all(token < dataset.tokenizer.timestamp_begin for token in decoder_output.tolist())

    def test_single_final_timestamp_timestamp_training_keeps_timestamps_and_normal_audio_path(self, monkeypatch):
        audio_array = self._load_fixture_audio_array()
        dataset = self._build_dataset(audio_array, no_timestamp_training=False)
        base_mel = torch.arange(data_loader_module.N_FRAMES).repeat(80, 1).float()
        captured = {}

        def fake_log_mel_spectrogram(audio, *args, **kwargs):
            captured["audio"] = np.asarray(audio).copy()
            return base_mel.clone()

        monkeypatch.setattr(data_loader_module, "log_mel_spectrogram", fake_log_mel_spectrogram)

        mel, decoder_input, decoder_output = dataset[0]

        assert mel.shape == (80, data_loader_module.N_FRAMES)
        assert torch.equal(mel, base_mel)
        assert captured["audio"].shape[0] == data_loader_module.N_SAMPLES
        np.testing.assert_allclose(captured["audio"][: audio_array.shape[0]], audio_array)
        assert dataset.tokenizer.no_timestamps not in decoder_input.tolist()
        assert any(token >= dataset.tokenizer.timestamp_begin for token in decoder_input.tolist())
        assert any(token >= dataset.tokenizer.timestamp_begin for token in decoder_output.tolist())


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
