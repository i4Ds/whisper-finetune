"""
Tests for Whisper data loading and decoder target construction.
"""

import sys
import types


def _install_whisper_stubs():
    """Provide minimal whisper modules so data_loader can be imported in unit tests."""
    if "whisper" in sys.modules:
        return

    whisper_module = types.ModuleType("whisper")

    audio_module = types.ModuleType("whisper.audio")
    audio_module.CHUNK_LENGTH = 30
    audio_module.N_FRAMES = 3000
    audio_module.N_SAMPLES = 480000
    audio_module.log_mel_spectrogram = lambda *args, **kwargs: None

    tokenizer_module = types.ModuleType("whisper.tokenizer")
    tokenizer_module.Tokenizer = object

    whisper_module.audio = audio_module
    whisper_module.tokenizer = tokenizer_module

    sys.modules["whisper"] = whisper_module
    sys.modules["whisper.audio"] = audio_module
    sys.modules["whisper.tokenizer"] = tokenizer_module


_install_whisper_stubs()

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
