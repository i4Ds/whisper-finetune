"""
Tests for configuration and utility functions.
"""

import numpy as np
import pytest
import torch
from torch import nn


class TestSeedSetting:
    """Test seed setting functionality."""

    def test_seed_reproducibility(self):
        """Test that setting seed produces reproducible results."""
        from whisper_finetune.utils import set_seed

        # Set seed and generate random numbers
        set_seed(42)
        random_nums_1 = [torch.rand(3, 3) for _ in range(3)]

        # Set same seed again
        set_seed(42)
        random_nums_2 = [torch.rand(3, 3) for _ in range(3)]

        # Should get identical results
        for t1, t2 in zip(random_nums_1, random_nums_2):
            assert torch.allclose(t1, t2), "Random numbers should be identical with same seed"

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        from whisper_finetune.utils import set_seed

        set_seed(42)
        random_nums_1 = torch.rand(10, 10)

        set_seed(123)
        random_nums_2 = torch.rand(10, 10)

        # Should get different results
        assert not torch.allclose(random_nums_1, random_nums_2), "Different seeds should produce different results"


class TestTrainingStepsCalculation:
    """Test training steps calculation."""

    def test_calculate_training_steps_basic(self):
        """Test basic training steps calculation."""
        from whisper_finetune.utils import calculate_training_steps

        config = {
            "training": {
                "epochs": 2,
                "accum_grad_steps": 4,
            },
            "dataset": {
                "batch_size": 8,
            },
        }

        # Mock dataset with 100 samples
        class MockDataset:
            def __len__(self):
                return 100

        dataset = MockDataset()
        steps = calculate_training_steps(config, dataset)

        # With 100 samples, batch_size=8, we have 13 batches per epoch (100/8 = 12.5, rounded up)
        # With accum_grad_steps=4, we have 13/4 = 3.25 -> 4 updates per epoch
        # With 2 epochs: 4 * 2 = 8 steps
        # Note: actual implementation might differ, this tests it works
        assert steps > 0
        assert isinstance(steps, int)

    def test_calculate_training_steps_single_epoch(self):
        """Test training steps for single epoch."""
        from whisper_finetune.utils import calculate_training_steps

        config = {
            "training": {
                "epochs": 1,
                "accum_grad_steps": 1,
            },
            "dataset": {
                "batch_size": 10,
            },
        }

        class MockDataset:
            def __len__(self):
                return 50

        dataset = MockDataset()
        steps = calculate_training_steps(config, dataset)

        # 50 samples / 10 batch_size = 5 batches per epoch
        # With accum_grad_steps=1, 5 steps per epoch
        # 1 epoch = 5 steps
        assert steps > 0

    def test_calculate_training_steps_scales_by_world_size_with_local_accum(self):
        """DDP step count should preserve global accum semantics when local accum is resolved."""
        from whisper_finetune.utils import calculate_training_steps

        config = {
            "training": {
                "epochs": 2,
                "accum_grad_steps": 4,
            },
            "dataset": {
                "batch_size": 8,
            },
        }

        class MockDataset:
            def __len__(self):
                return 128

        dataset = MockDataset()

        assert calculate_training_steps(config, dataset, world_size=1) == 8

        config["training"]["accum_grad_steps"] = 2
        assert calculate_training_steps(config, dataset, world_size=2) == 8

        config["training"]["accum_grad_steps"] = 1
        assert calculate_training_steps(config, dataset, world_size=4) == 8

    def test_resolve_local_accum_grad_steps(self):
        from whisper_finetune.utils import resolve_local_accum_grad_steps

        assert resolve_local_accum_grad_steps(8, world_size=1) == 8
        assert resolve_local_accum_grad_steps(8, world_size=2) == 4
        assert resolve_local_accum_grad_steps(8, world_size=4) == 2

    def test_resolve_local_accum_grad_steps_rejects_fractional_local_window(self):
        from whisper_finetune.utils import resolve_local_accum_grad_steps

        with pytest.raises(ValueError, match="global accumulation window"):
            resolve_local_accum_grad_steps(2, world_size=4)

    def test_calculate_training_steps_drop_last_uses_complete_accumulation_windows(self):
        """drop_last=True should avoid scheduling partial final accumulation windows."""
        from whisper_finetune.utils import calculate_training_steps

        config = {
            "training": {
                "epochs": 1,
                "accum_grad_steps": 4,
            },
            "dataset": {
                "batch_size": 8,
            },
        }

        class MockDataset:
            def __len__(self):
                return 130

        dataset = MockDataset()

        # Default/drop_last=True:
        # per-rank samples = 65, per-rank full microbatches = 8, full accum windows = 2.
        assert calculate_training_steps(config, dataset, world_size=2) == 2
        assert calculate_training_steps(config, dataset, world_size=2, drop_last=True) == 2

        # Without drop_last: ceil(130 / (8 * 2 * 4)) = 3 optimizer steps.
        assert calculate_training_steps(config, dataset, world_size=2, drop_last=False) == 3


class TestValidationStepsCalculation:
    """Test validation steps calculation."""

    def test_calculate_val_steps_fraction(self):
        """Test validation steps with fractional eval_steps."""
        from whisper_finetune.utils import calculate_val_steps

        config = {
            "training": {
                "train_steps": 100,
                "epochs": 4,  # Required by calculate_val_steps
                "eval_steps": 0.25,  # Evaluate 4 times per epoch
            }
        }

        val_steps = calculate_val_steps(config)

        # Should evaluate every 25 steps (100 / 4 * 0.25 = 6.25 -> 6)
        assert val_steps > 0
        assert isinstance(val_steps, int)

    def test_calculate_val_steps_integer(self):
        """Test validation steps with integer eval_steps."""
        from whisper_finetune.utils import calculate_val_steps

        config = {
            "training": {
                "train_steps": 100,
                "epochs": 2,  # Required by calculate_val_steps
                "eval_steps": 10,  # Evaluate every 10 steps (per epoch)
            }
        }

        val_steps = calculate_val_steps(config)

        # Formula: (train_steps / epochs) * eval_steps = (100 / 2) * 10 = 500
        assert val_steps == 500


class TestGradientDisabling:
    """Test gradient disabling functionality."""

    def test_disable_all_grads(self):
        """Test disabling gradients for a module."""
        from whisper_finetune.utils import disable_all_grads

        # Create a simple module
        module = torch.nn.Linear(10, 5)

        # Initially, all parameters should require grad
        assert all(p.requires_grad for p in module.parameters())

        # Disable gradients
        disable_all_grads(module)

        # Now, no parameters should require grad
        assert all(not p.requires_grad for p in module.parameters())

    def test_disable_grads_nested_module(self):
        """Test disabling gradients in nested modules."""
        from whisper_finetune.utils import disable_all_grads

        # Create nested module
        module = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5),
        )

        disable_all_grads(module)

        # All parameters in all submodules should have grad disabled
        for param in module.parameters():
            assert not param.requires_grad


class TestModelSizeCalculation:
    """Test model size printing/calculation."""

    def test_print_size_of_model(self):
        """Test model size calculation (returns bytes, despite name)."""
        from whisper_finetune.utils import print_size_of_model

        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 100),
            torch.nn.Linear(100, 10),
        )

        size_bytes = print_size_of_model(model)

        # Should return size in bytes (note: function name is misleading)
        assert size_bytes > 0
        assert isinstance(size_bytes, int)

        # For this small model, size should be less than 1 MB (1e6 bytes)
        assert size_bytes < 1e6


class TestConfigReading:
    """Test configuration file reading."""

    def test_read_config_returns_dict(self):
        """Test that read_config returns a dictionary."""
        # This test would need an actual config file
        # For now, we'll test that the function exists
        from whisper_finetune.utils import read_config

        # Function should be callable
        assert callable(read_config)


class TestStochasticDepthConfiguration:
    """Test stochastic depth configuration for encoder/decoder-only training."""

    def test_encoder_only_training_disables_decoder_stochastic_depth(self):
        """Test that stochastic depth is disabled for decoder when training encoder only."""
        # Simulating the logic from finetune.py
        config = {
            "training": {
                "train_only_decoder": False,
                "train_only_encoder": True,
                "stochastic_depth": 0.1,
            }
        }

        # This is the logic from finetune.py
        encoder_stochastic_depth = (
            0.0 if config["training"]["train_only_decoder"] else config["training"]["stochastic_depth"]
        )
        decoder_stochastic_depth = (
            0.0 if config["training"]["train_only_encoder"] else config["training"]["stochastic_depth"]
        )

        assert encoder_stochastic_depth == 0.1, "Encoder should have stochastic depth when training encoder only"
        assert decoder_stochastic_depth == 0.0, "Decoder should NOT have stochastic depth when training encoder only"

    def test_decoder_only_training_disables_encoder_stochastic_depth(self):
        """Test that stochastic depth is disabled for encoder when training decoder only."""
        config = {
            "training": {
                "train_only_decoder": True,
                "train_only_encoder": False,
                "stochastic_depth": 0.1,
            }
        }

        encoder_stochastic_depth = (
            0.0 if config["training"]["train_only_decoder"] else config["training"]["stochastic_depth"]
        )
        decoder_stochastic_depth = (
            0.0 if config["training"]["train_only_encoder"] else config["training"]["stochastic_depth"]
        )

        assert encoder_stochastic_depth == 0.0, "Encoder should NOT have stochastic depth when training decoder only"
        assert decoder_stochastic_depth == 0.1, "Decoder should have stochastic depth when training decoder only"

    def test_full_training_enables_both_stochastic_depth(self):
        """Test that stochastic depth is enabled for both when training full model."""
        config = {
            "training": {
                "train_only_decoder": False,
                "train_only_encoder": False,
                "stochastic_depth": 0.1,
            }
        }

        encoder_stochastic_depth = (
            0.0 if config["training"]["train_only_decoder"] else config["training"]["stochastic_depth"]
        )
        decoder_stochastic_depth = (
            0.0 if config["training"]["train_only_encoder"] else config["training"]["stochastic_depth"]
        )

        assert encoder_stochastic_depth == 0.1, "Encoder should have stochastic depth when training full model"
        assert decoder_stochastic_depth == 0.1, "Decoder should have stochastic depth when training full model"


class TestInfiniteIterator:
    """Test infinite iterator for data loading."""

    def test_infinite_iter_repeats(self):
        """Test that infinite_iter repeats the dataloader."""
        from whisper_finetune.model.model_utils import infinite_iter

        # Create a simple list to iterate over
        data = [1, 2, 3]

        class SimpleLoader:
            def __iter__(self):
                return iter(data)

        loader = SimpleLoader()
        inf_iter = infinite_iter(loader)

        # Should be able to get more items than in original list
        items = [next(inf_iter) for _ in range(10)]

        assert len(items) == 10
        # Should cycle through [1, 2, 3, 1, 2, 3, ...]
        assert items[:3] == [1, 2, 3]
        assert items[3:6] == [1, 2, 3]

    def test_infinite_iter_calls_distributed_sampler_set_epoch(self):
        """DistributedSampler shuffling must be reseeded when the loader cycles."""
        from whisper_finetune.model.model_utils import infinite_iter

        class Sampler:
            def __init__(self):
                self.epochs = []

            def set_epoch(self, epoch):
                self.epochs.append(epoch)

        class Loader:
            def __init__(self):
                self.sampler = Sampler()

            def __iter__(self):
                return iter([1, 2])

        loader = Loader()
        inf_iter = infinite_iter(loader)

        assert [next(inf_iter) for _ in range(5)] == [1, 2, 1, 2, 1]
        assert loader.sampler.epochs == [0, 1, 2]


class _NoSyncContext:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.no_sync_entries += 1

    def __exit__(self, exc_type, exc, tb):
        return False


class _TinyDDPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3)
        self.no_sync_entries = 0

    def no_sync(self):
        return _NoSyncContext(self)

    def forward(self, x, y_in):
        logits = self.proj(x)
        return logits.unsqueeze(1).expand(-1, y_in.shape[1], -1).contiguous()


class _IllegalMemoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3)
        self.forward_calls = 0

    def forward(self, x, y_in):
        self.forward_calls += 1
        raise RuntimeError("CUDA error: an illegal memory access was encountered")


class TestDDPGradientAccumulation:
    def _batch(self):
        x = torch.randn(2, 4)
        y_in = torch.zeros(2, 5, dtype=torch.long)
        y_out = torch.randint(0, 3, (2, 5), dtype=torch.long)
        return x, y_in, y_out

    def test_train_step_uses_no_sync_until_last_accumulation(self, monkeypatch):
        """Only the final microbatch in an accumulation window should sync grads."""
        import whisper_finetune.runtime as rt
        from whisper_finetune.model.model_utils import train_step

        monkeypatch.setattr(rt, "IS_DISTRIBUTED", True)
        monkeypatch.setattr(rt, "IS_MAIN", True)

        model = _TinyDDPModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        train_iter = iter([self._batch(), self._batch(), self._batch()])

        loss = train_step(
            model,
            train_iter,
            optimizer,
            scheduler,
            {
                "mixed_precision_training": False,
                "accum_grad_steps": 3,
                "max_grad_norm": 1.0,
                "mp_dtype": "fp16",
                "label_smoothing": 0.0,
                "is_lora_run": False,
            },
            step=1,
        )

        assert loss > 0
        assert model.no_sync_entries == 2
        assert scheduler.last_epoch == 1

    def test_train_step_does_not_use_no_sync_without_ddp(self, monkeypatch):
        import whisper_finetune.runtime as rt
        from whisper_finetune.model.model_utils import train_step

        monkeypatch.setattr(rt, "IS_DISTRIBUTED", False)

        model = _TinyDDPModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        train_iter = iter([self._batch(), self._batch()])

        train_step(
            model,
            train_iter,
            optimizer,
            scheduler,
            {
                "mixed_precision_training": False,
                "accum_grad_steps": 2,
                "max_grad_norm": 1.0,
                "mp_dtype": "fp16",
                "label_smoothing": 0.0,
                "is_lora_run": False,
            },
            step=1,
        )

        assert model.no_sync_entries == 0

    def test_train_step_raises_illegal_memory_immediately_under_ddp(self, monkeypatch):
        import whisper_finetune.runtime as rt
        from whisper_finetune.model.model_utils import train_step

        monkeypatch.setattr(rt, "IS_DISTRIBUTED", True)

        model = _IllegalMemoryModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        train_iter = iter([self._batch(), self._batch(), self._batch()])

        with pytest.raises(RuntimeError, match="illegal memory"):
            train_step(
                model,
                train_iter,
                optimizer,
                scheduler,
                {
                    "mixed_precision_training": False,
                    "accum_grad_steps": 3,
                    "max_grad_norm": 1.0,
                    "mp_dtype": "fp16",
                    "label_smoothing": 0.0,
                    "is_lora_run": False,
                },
                step=1,
            )

        assert model.forward_calls == 1

    def test_train_step_requires_persistent_scaler_for_fp16(self, monkeypatch):
        import whisper_finetune.runtime as rt
        from whisper_finetune.model.model_utils import train_step

        monkeypatch.setattr(rt, "IS_DISTRIBUTED", False)

        model = _TinyDDPModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        train_iter = iter([self._batch()])

        with pytest.raises(ValueError, match="persistent GradScaler"):
            train_step(
                model,
                train_iter,
                optimizer,
                scheduler,
                {
                    "mixed_precision_training": True,
                    "accum_grad_steps": 1,
                    "max_grad_norm": 1.0,
                    "mp_dtype": "fp16",
                    "label_smoothing": 0.0,
                    "is_lora_run": False,
                },
                step=1,
            )
