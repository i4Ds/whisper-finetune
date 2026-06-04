"""
Tests for model utility helpers.
"""

import pytest
import torch

from whisper_finetune.model import model_utils


class AdditiveMask:
    def __init__(self, value, calls):
        self.value = value
        self.calls = calls

    def __call__(self, x):
        self.calls.append(self.value)
        return x + self.value


class FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_ln = torch.nn.Identity()


class FakeEncoder(torch.nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        self.blocks = torch.nn.ModuleList(FakeBlock() for _ in range(num_blocks))

    def forward(self, x):
        for block in self.blocks:
            x = block.attn_ln(x)
        return x


class FakeWhisper(torch.nn.Module):
    def __init__(self, num_blocks=3):
        super().__init__()
        self.encoder = FakeEncoder(num_blocks)


class TestDeepSpecAugmentProbability:
    def _patch_masks(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            model_utils.T,
            "TimeMasking",
            lambda *args, **kwargs: AdditiveMask(1, calls),
        )
        monkeypatch.setattr(
            model_utils.T,
            "FrequencyMasking",
            lambda *args, **kwargs: AdditiveMask(2, calls),
        )
        return calls

    def test_deep_spec_augment_p_zero_skips_masks(self, monkeypatch):
        calls = self._patch_masks(monkeypatch)
        model = FakeWhisper()
        model_utils.register_deep_spec_augment_hooks(
            model,
            time_mask_param=100,
            freq_mask_param=27,
            p=0.0,
            layer_indices=[0],
        )

        x = torch.zeros(1, 5, 4)
        output = model.encoder(x)

        assert calls == []
        assert torch.equal(output, x)

    def test_deep_spec_augment_p_one_applies_masks(self, monkeypatch):
        calls = self._patch_masks(monkeypatch)
        model = FakeWhisper()
        model_utils.register_deep_spec_augment_hooks(
            model,
            time_mask_param=100,
            freq_mask_param=27,
            p=1.0,
            layer_indices=[0],
        )

        x = torch.zeros(1, 5, 4)
        output = model.encoder(x)

        assert calls == [1, 2]
        assert torch.equal(output, x + 3)

    def test_deep_spec_augment_p_uses_probability_threshold(self, monkeypatch):
        self._patch_masks(monkeypatch)
        model = FakeWhisper()
        model_utils.register_deep_spec_augment_hooks(
            model,
            time_mask_param=100,
            freq_mask_param=27,
            p=0.5,
            layer_indices=[0],
        )

        x = torch.zeros(1, 5, 4)
        monkeypatch.setattr(
            model_utils.torch,
            "rand",
            lambda *args, **kwargs: torch.tensor([0.25]),
        )
        assert torch.equal(model.encoder(x), x + 3)

        monkeypatch.setattr(
            model_utils.torch,
            "rand",
            lambda *args, **kwargs: torch.tensor([0.75]),
        )
        assert torch.equal(model.encoder(x), x)

    def test_invalid_deep_spec_augment_p_raises(self, monkeypatch):
        self._patch_masks(monkeypatch)

        with pytest.raises(ValueError, match="deep_spec_augment p must be between 0 and 1"):
            model_utils.register_deep_spec_augment_hooks(
                FakeWhisper(),
                time_mask_param=100,
                freq_mask_param=27,
                p=-0.1,
                layer_indices=[0],
            )
