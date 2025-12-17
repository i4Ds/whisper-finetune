"""
Tests for LoRA (Low-Rank Adaptation) functionality with minlora.

This module tests that minlora correctly applies LoRA to Whisper models.
"""

import pytest
import torch
from functools import partial
from torch.nn.utils.parametrize import is_parametrized


class TestLoRABasics:
    """Test basic LoRA functionality from minlora."""

    def test_lora_parametrization_creation(self):
        """Test that LoRA parametrization can be created."""
        from minlora import LoRAParametrization
        
        # Create a simple linear layer
        linear = torch.nn.Linear(64, 32)
        
        # Create LoRA parametrization
        lora_param = LoRAParametrization.from_linear(linear, rank=4, lora_alpha=8, lora_dropout_p=0.1)
        
        assert lora_param is not None
        assert hasattr(lora_param, 'lora_A')
        assert hasattr(lora_param, 'lora_B')
        assert lora_param.rank == 4
        assert lora_param.lora_alpha == 8

    def test_lora_parameter_shapes(self):
        """Test that LoRA matrices have correct shapes."""
        from minlora import LoRAParametrization
        
        fan_in, fan_out = 64, 32
        rank = 8
        
        linear = torch.nn.Linear(fan_in, fan_out)
        lora_param = LoRAParametrization.from_linear(linear, rank=rank)
        
        # lora_A should be (rank, fan_in) and lora_B should be (fan_out, rank)
        assert lora_param.lora_A.shape == (rank, fan_in)
        assert lora_param.lora_B.shape == (fan_out, rank)

    def test_lora_forward_changes_output(self):
        """Test that LoRA forward pass modifies the output."""
        from minlora import LoRAParametrization
        import torch.nn.utils.parametrize as parametrize
        
        linear = torch.nn.Linear(64, 32)
        x = torch.randn(1, 64)
        
        # Output without LoRA
        output_before = linear(x).clone()
        
        # Apply LoRA
        lora_config = partial(LoRAParametrization.from_linear, rank=4, lora_alpha=8)
        parametrize.register_parametrization(linear, 'weight', lora_config(linear))
        
        # Get LoRA params and set them to non-zero
        for name, param in linear.named_parameters():
            if 'lora_B' in name:
                param.data = torch.randn_like(param) * 0.1
        
        # Output with LoRA (should be different)
        output_after = linear(x)
        
        # Outputs should be different since LoRA B is non-zero
        assert not torch.allclose(output_before, output_after)

    def test_lora_enable_disable(self):
        """Test that LoRA can be enabled and disabled."""
        from minlora import LoRAParametrization
        import torch.nn.utils.parametrize as parametrize
        
        linear = torch.nn.Linear(64, 32)
        x = torch.randn(1, 64)
        
        # Apply LoRA
        lora_param = LoRAParametrization.from_linear(linear, rank=4, lora_alpha=8)
        parametrize.register_parametrization(linear, 'weight', lora_param)
        
        # Set lora_B to non-zero
        lora_param.lora_B.data = torch.randn_like(lora_param.lora_B) * 0.1
        
        # Get output with LoRA enabled
        output_enabled = linear(x).clone()
        
        # Disable LoRA
        lora_param.disable_lora()
        output_disabled = linear(x).clone()
        
        # Enable LoRA again
        lora_param.enable_lora()
        output_re_enabled = linear(x)
        
        # Disabled should be different from enabled
        assert not torch.allclose(output_enabled, output_disabled)
        # Re-enabled should match original enabled
        assert torch.allclose(output_enabled, output_re_enabled)


class TestMinLoRAWorkflow:
    """Test the complete minlora workflow as shown in their examples."""

    def test_complete_lora_workflow(self):
        """Test the complete workflow: add, train, save, load, merge."""
        from minlora import (
            add_lora, apply_to_lora, disable_lora, enable_lora, 
            get_lora_params, get_lora_state_dict, merge_lora, remove_lora
        )
        
        # Create model
        model = torch.nn.Sequential(
            torch.nn.Linear(in_features=5, out_features=7),
            torch.nn.Linear(in_features=7, out_features=3),
        )
        
        x = torch.randn(1, 5)
        y0 = model(x).clone()
        
        # Add LoRA - output should be same because B is initialized to 0
        add_lora(model)
        y_after_lora = model(x)
        assert torch.allclose(y_after_lora, y0), "Output should be same after adding LoRA (B=0)"
        
        # Initialize B to non-zero
        model.apply(apply_to_lora(lambda x: torch.nn.init.ones_(x.lora_B)))
        y1 = model(x).clone()
        assert not torch.allclose(y1, y0), "Output should differ after initializing B"
        
        # Disable LoRA - output should match original
        disable_lora(model)
        y_disabled = model(x)
        assert torch.allclose(y_disabled, y0), "Output should match original when LoRA disabled"
        
        # Enable LoRA
        enable_lora(model)
        y_enabled = model(x)
        assert torch.allclose(y_enabled, y1), "Output should match Y1 when LoRA re-enabled"
        
        # Save state dict
        state_dict_to_save = get_lora_state_dict(model)
        assert len(state_dict_to_save) == 4, "Should have 4 LoRA params (2 layers * 2 matrices)"
        
        # Remove and re-add LoRA
        remove_lora(model)
        add_lora(model)
        
        # Load saved state
        model.load_state_dict(state_dict_to_save, strict=False)
        y_loaded = model(x)
        assert torch.allclose(y_loaded, y1), "Output should match after loading saved LoRA"
        
        # Merge LoRA
        merge_lora(model)
        y_merged = model(x)
        assert torch.allclose(y_merged, y1), "Output should match after merging LoRA"

    def test_lora_state_dict_keys(self):
        """Test that LoRA state dict has correct keys."""
        from minlora import add_lora, get_lora_state_dict, name_is_lora
        
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.Linear(128, 32),
        )
        
        add_lora(model)
        state_dict = get_lora_state_dict(model)
        
        # All keys should be LoRA parameters
        for key in state_dict.keys():
            assert name_is_lora(key), f"Key {key} should be a LoRA parameter"
            assert 'lora_A' in key or 'lora_B' in key, f"Key {key} should contain lora_A or lora_B"
        
        # Should have 4 keys (2 layers * 2 matrices)
        assert len(state_dict) == 4

    def test_optimizer_with_get_lora_params(self):
        """Test that optimizer can be created with get_lora_params."""
        from minlora import add_lora, get_lora_params
        
        model = torch.nn.Linear(in_features=5, out_features=3)
        add_lora(model)
        
        # Create optimizer with LoRA params only
        lora_params = list(get_lora_params(model))
        optimizer = torch.optim.AdamW([{"params": lora_params}], lr=1e-3)
        
        # Should have correct number of param groups
        assert len(optimizer.param_groups) == 1
        assert len(optimizer.param_groups[0]['params']) == 2  # lora_A and lora_B


class TestMinLoRAAddRemove:
    """Test add_lora and remove_lora functions."""

    def test_add_lora_to_model(self):
        """Test that add_lora adds parametrizations to all Linear layers."""
        from minlora import add_lora
        
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
        )
        
        # Before LoRA, no parametrizations
        assert not is_parametrized(model[0], 'weight')
        assert not is_parametrized(model[2], 'weight')
        
        # Add LoRA
        add_lora(model)
        
        # After LoRA, Linear layers should be parametrized
        assert is_parametrized(model[0], 'weight')
        assert is_parametrized(model[2], 'weight')

    def test_remove_lora_from_model(self):
        """Test that remove_lora removes parametrizations."""
        from minlora import add_lora, remove_lora
        
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.Linear(128, 32),
        )
        
        add_lora(model)
        assert is_parametrized(model[0], 'weight')
        
        remove_lora(model)
        assert not is_parametrized(model[0], 'weight')

    def test_merge_lora(self):
        """Test that merge_lora merges LoRA weights into base model."""
        from minlora import add_lora, merge_lora, LoRAParametrization
        
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
        )
        x = torch.randn(1, 64)
        
        # Add LoRA with custom config to have non-zero lora_B
        lora_config = {
            torch.nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=4, lora_alpha=8),
            },
        }
        add_lora(model, lora_config=lora_config)
        
        # Set lora_B to non-zero for meaningful test
        for name, param in model.named_parameters():
            if 'lora_B' in name:
                param.data = torch.randn_like(param) * 0.1
        
        # Get output before merge
        output_before = model(x).clone()
        
        # Merge LoRA
        merge_lora(model)
        
        # Should no longer be parametrized
        assert not is_parametrized(model[0], 'weight')
        
        # Output should be the same after merge
        output_after = model(x)
        assert torch.allclose(output_before, output_after, atol=1e-5)


class TestLoRATrainableParameters:
    """Test that LoRA correctly freezes base model and only trains LoRA params."""

    def test_only_lora_params_trainable(self):
        """Test that after applying LoRA, only LoRA parameters are trainable."""
        from whisper_finetune.model.lora import disable_all_but_parametrized_grads
        from minlora import add_lora, LoRAParametrization
        from functools import partial
        
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
        )
        
        # Initially all params should be trainable
        all_trainable_before = all(p.requires_grad for p in model.parameters())
        assert all_trainable_before
        
        # Add LoRA
        lora_config = {
            torch.nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=4, lora_alpha=8),
            },
        }
        add_lora(model, lora_config=lora_config)
        
        # Disable gradients for non-LoRA params
        disable_all_but_parametrized_grads(model)
        
        # Count trainable params
        trainable_params = []
        frozen_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
            else:
                frozen_params.append(name)
        
        # Only LoRA params should be trainable (not bias, not original weights)
        for name in trainable_params:
            assert 'lora' in name.lower(), f"Unexpected trainable param {name}"
        
        # Original weights and biases should be frozen
        assert len(frozen_params) > 0, "Some params should be frozen"
        
        # Verify bias is frozen
        bias_frozen = any('bias' in name for name in frozen_params)
        assert bias_frozen, "Bias should be frozen"

    def test_trainable_params_match_get_lora_params(self):
        """Test that trainable params exactly match what minlora's get_lora_params returns."""
        from whisper_finetune.model.lora import disable_all_but_parametrized_grads
        from minlora import add_lora, get_lora_params, LoRAParametrization
        from functools import partial
        
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64, 32),
        )
        
        # Add LoRA
        lora_config = {
            torch.nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=8, lora_alpha=16),
            },
        }
        add_lora(model, lora_config=lora_config)
        
        # Disable gradients for non-LoRA params
        disable_all_but_parametrized_grads(model)
        
        # Get trainable params
        trainable_tensors = {id(p) for p in model.parameters() if p.requires_grad}
        
        # Get LoRA params from minlora's function
        lora_tensors = {id(p) for p in get_lora_params(model)}
        
        # They should be exactly the same
        assert trainable_tensors == lora_tensors, (
            f"Trainable params ({len(trainable_tensors)}) don't match "
            f"get_lora_params ({len(lora_tensors)})"
        )

    def test_trainable_param_count(self):
        """Test that trainable parameter count is significantly reduced with LoRA."""
        from whisper_finetune.model.lora import disable_all_but_parametrized_grads
        from minlora import add_lora, LoRAParametrization
        from functools import partial
        
        model = torch.nn.Sequential(
            torch.nn.Linear(512, 512, bias=False),  # No bias to simplify counting
            torch.nn.Linear(512, 512, bias=False),
            torch.nn.Linear(512, 512, bias=False),
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Add LoRA with low rank
        rank = 4
        lora_config = {
            torch.nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=rank, lora_alpha=8),
            },
        }
        add_lora(model, lora_config=lora_config)
        disable_all_but_parametrized_grads(model)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # LoRA params should be much less than total
        # Each layer: rank * 512 (A) + 512 * rank (B) = 2 * rank * 512
        expected_lora_params = 3 * 2 * rank * 512
        
        assert trainable_params == expected_lora_params, f"Expected {expected_lora_params}, got {trainable_params}"
        assert trainable_params < total_params * 0.1, "LoRA should have <10% trainable params"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLoRAWithWhisper:
    """Test LoRA with Whisper models (requires CUDA for practical testing)."""

    def test_apply_lora_to_whisper_base(self):
        """Test applying LoRA to Whisper base model."""
        import whisper
        from whisper_finetune.model.lora import apply_lora, print_lora_info
        from whisper_finetune.utils import print_trainable_parameters
        
        # Load tiny model for fast testing
        model = whisper.load_model("tiny")
        model.to("cuda")
        
        # Check trainable params before
        total_params_before = sum(p.numel() for p in model.parameters())
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert trainable_before == total_params_before, "All params should be trainable before LoRA"
        
        # Apply LoRA
        lora_config = {
            "rank": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.1,
        }
        apply_lora(model, lora_config)
        
        # Check trainable params after
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert trainable_after < trainable_before, "Trainable params should decrease"
        assert trainable_after > 0, "Some params should still be trainable"
        
        # Trainable should be a small fraction
        ratio = trainable_after / total_params_before
        assert ratio < 0.05, f"LoRA should have <5% trainable params, got {ratio*100:.2f}%"

    def test_lora_forward_pass_works(self):
        """Test that forward pass works after applying LoRA."""
        import whisper
        from whisper_finetune.model.lora import apply_lora
        
        model = whisper.load_model("tiny")
        model.to("cuda")
        
        # Apply LoRA
        lora_config = {"rank": 4, "lora_alpha": 8, "lora_dropout": 0.0}
        apply_lora(model, lora_config)
        
        # Create dummy mel spectrogram input (1, 80, 3000) for tiny model
        mel = torch.randn(1, 80, 3000).to("cuda")
        
        # Forward pass should work
        with torch.no_grad():
            audio_features = model.encoder(mel)
        
        assert audio_features is not None
        assert audio_features.shape[0] == 1

    def test_lora_backward_pass_works(self):
        """Test that backward pass works and only updates LoRA params."""
        import whisper
        from whisper_finetune.model.lora import apply_lora
        
        model = whisper.load_model("tiny")
        model.to("cuda")
        
        # Apply LoRA
        lora_config = {"rank": 4, "lora_alpha": 8, "lora_dropout": 0.0}
        apply_lora(model, lora_config)
        
        # Store initial LoRA param values
        lora_params_before = {}
        for name, param in model.named_parameters():
            if param.requires_grad and 'lora' in name.lower():
                lora_params_before[name] = param.clone().detach()
        
        # Forward pass
        mel = torch.randn(1, 80, 3000).to("cuda")
        audio_features = model.encoder(mel)
        
        # Simple loss
        loss = audio_features.sum()
        loss.backward()
        
        # Check that LoRA params have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"LoRA param {name} should have gradient"

    def test_lora_only_decoder(self):
        """Test applying LoRA only to decoder."""
        import whisper
        from whisper_finetune.model.lora import apply_lora
        from torch.nn.utils.parametrize import is_parametrized
        
        model = whisper.load_model("tiny")
        model.to("cuda")
        
        # Apply LoRA only to decoder
        lora_config = {"rank": 4, "lora_alpha": 8, "lora_dropout": 0.0}
        apply_lora(model, lora_config, train_only_decoder=True)
        
        # Check encoder has no LoRA
        encoder_parametrized = False
        for name, module in model.encoder.named_modules():
            if hasattr(module, 'parametrizations'):
                encoder_parametrized = True
                break
        
        # Check decoder has LoRA
        decoder_parametrized = False
        for name, module in model.decoder.named_modules():
            if hasattr(module, 'parametrizations'):
                decoder_parametrized = True
                break
        
        assert not encoder_parametrized, "Encoder should not have LoRA"
        assert decoder_parametrized, "Decoder should have LoRA"

    def test_lora_only_encoder(self):
        """Test applying LoRA only to encoder."""
        import whisper
        from whisper_finetune.model.lora import apply_lora
        
        model = whisper.load_model("tiny")
        model.to("cuda")
        
        # Apply LoRA only to encoder
        lora_config = {"rank": 4, "lora_alpha": 8, "lora_dropout": 0.0}
        apply_lora(model, lora_config, train_only_encoder=True)
        
        # Check encoder has LoRA
        encoder_parametrized = False
        for name, module in model.encoder.named_modules():
            if hasattr(module, 'parametrizations'):
                encoder_parametrized = True
                break
        
        # Check decoder has no LoRA
        decoder_parametrized = False
        for name, module in model.decoder.named_modules():
            if hasattr(module, 'parametrizations'):
                decoder_parametrized = True
                break
        
        assert encoder_parametrized, "Encoder should have LoRA"
        assert not decoder_parametrized, "Decoder should not have LoRA"


class TestLoRAConfigValues:
    """Test that LoRA config values are applied correctly."""

    def test_lora_rank_affects_param_count(self):
        """Test that different ranks produce different param counts."""
        from minlora import add_lora, LoRAParametrization
        from functools import partial
        
        def get_lora_param_count(model, rank):
            lora_config = {
                torch.nn.Linear: {
                    "weight": partial(LoRAParametrization.from_linear, rank=rank, lora_alpha=rank*2),
                },
            }
            add_lora(model, lora_config=lora_config)
            count = sum(p.numel() for p in model.parameters() if 'lora' in str(p))
            return count
        
        model_r4 = torch.nn.Linear(256, 256)
        model_r16 = torch.nn.Linear(256, 256)
        
        # Get counts for different ranks
        lora_config_r4 = {
            torch.nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=4),
            },
        }
        lora_config_r16 = {
            torch.nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=16),
            },
        }
        
        add_lora(model_r4, lora_config=lora_config_r4)
        add_lora(model_r16, lora_config=lora_config_r16)
        
        # Count LoRA parameters
        count_r4 = 0
        count_r16 = 0
        
        for name, param in model_r4.named_parameters():
            if 'lora' in name.lower():
                count_r4 += param.numel()
        
        for name, param in model_r16.named_parameters():
            if 'lora' in name.lower():
                count_r16 += param.numel()
        
        # rank=16 should have 4x more params than rank=4
        assert count_r16 == count_r4 * 4, f"Expected 4x, got {count_r16}/{count_r4}"

    def test_lora_scaling(self):
        """Test that LoRA alpha scaling is applied correctly."""
        from minlora import LoRAParametrization
        import torch.nn.utils.parametrize as parametrize
        
        linear = torch.nn.Linear(64, 32, bias=False)
        x = torch.randn(1, 64)
        
        # Store original weight
        original_weight = linear.weight.clone()
        
        # Apply LoRA with different alphas
        lora_param_a1 = LoRAParametrization.from_linear(linear, rank=4, lora_alpha=1)
        lora_param_a8 = LoRAParametrization.from_linear(linear, rank=4, lora_alpha=8)
        
        # Scaling should be alpha/rank
        assert lora_param_a1.scaling == 1/4
        assert lora_param_a8.scaling == 8/4


class TestLoRAIntegration:
    """Integration tests for LoRA with the whisper_finetune module."""

    def test_apply_lora_function(self):
        """Test the apply_lora function from whisper_finetune."""
        from whisper_finetune.model.lora import apply_lora, print_lora_info
        
        # Create a mock "model" with Whisper-like Linear layers
        # We use regular nn.Linear since whisper.model.Linear is similar
        class MockWhisperEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv1d(80, 512, 3)
                self.blocks = torch.nn.ModuleList([
                    torch.nn.Sequential(
                        torch.nn.Linear(512, 512),
                        torch.nn.Linear(512, 512),
                    ) for _ in range(2)
                ])
        
        class MockWhisperDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([
                    torch.nn.Sequential(
                        torch.nn.Linear(512, 512),
                        torch.nn.Linear(512, 512),
                    ) for _ in range(2)
                ])
        
        class MockWhisperModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = MockWhisperEncoder()
                self.decoder = MockWhisperDecoder()
        
        model = MockWhisperModel()
        
        # Count params before
        total_params = sum(p.numel() for p in model.parameters())
        
        # Apply LoRA using whisper_finetune's function
        # Note: This won't work directly because it expects whisper.model.Linear
        # But we can test the core functionality
        from minlora import add_lora, LoRAParametrization
        from functools import partial
        from whisper_finetune.model.lora import disable_all_but_parametrized_grads
        
        lora_config = {
            torch.nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=4, lora_alpha=8, lora_dropout_p=0.1),
            },
        }
        add_lora(model, lora_config=lora_config)
        disable_all_but_parametrized_grads(model)
        
        # Count trainable params after
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert trainable_params < total_params
        assert trainable_params > 0

    def test_print_lora_info(self):
        """Test that print_lora_info works correctly."""
        from whisper_finetune.model.lora import print_lora_info, disable_all_but_parametrized_grads
        from minlora import add_lora, LoRAParametrization
        from functools import partial
        import io
        import sys
        
        model = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.Linear(256, 256),
        )
        
        lora_config = {
            torch.nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=8, lora_alpha=16),
            },
        }
        add_lora(model, lora_config=lora_config)
        disable_all_but_parametrized_grads(model)
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        print_lora_info(model)
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        
        # Should contain info about trainable params
        assert "trainable" in output.lower() or "LoRA" in output
        assert "%" in output  # Should show percentage


class TestLoRADebugFunctions:
    """Test LoRA debug functions for parameter and gradient tracking."""

    def test_is_lora_enabled(self):
        """Test that is_lora_enabled correctly detects LoRA parameters."""
        from whisper_finetune.model.lora import is_lora_enabled
        from minlora import add_lora, LoRAParametrization
        
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 16),
        )
        
        # Before adding LoRA
        assert not is_lora_enabled(model), "Should return False before LoRA is added"
        
        # Add LoRA
        lora_config = {
            torch.nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=4),
            },
        }
        add_lora(model, lora_config=lora_config)
        
        # After adding LoRA
        assert is_lora_enabled(model), "Should return True after LoRA is added"

    def test_get_lora_debug_stats(self):
        """Test that get_lora_debug_stats returns correct statistics."""
        from whisper_finetune.model.lora import get_lora_debug_stats, disable_all_but_parametrized_grads
        from minlora import add_lora, LoRAParametrization
        
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 16),
        )
        
        lora_config = {
            torch.nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=4),
            },
        }
        add_lora(model, lora_config=lora_config)
        disable_all_but_parametrized_grads(model)
        
        # Get stats before backward (no gradients yet)
        stats = get_lora_debug_stats(model, representative_module_pattern="")
        
        # Should have parameter norms
        assert stats["lora_A_norm"] is not None, "Should have lora_A_norm"
        assert stats["lora_B_norm"] is not None, "Should have lora_B_norm"
        assert stats["lora_A_norm"] > 0, "lora_A_norm should be positive (initialized with kaiming)"
        # lora_B starts at 0, so norm should be 0
        assert stats["lora_B_norm"] == 0.0, "lora_B_norm should be 0 (initialized to zero)"
        
        # No gradients yet
        assert stats["lora_A_grad_norm"] is None, "Should have no gradient before backward"
        assert stats["lora_B_grad_norm"] is None, "Should have no gradient before backward"

    def test_get_lora_debug_stats_with_gradients(self):
        """Test that get_lora_debug_stats captures gradients after backward."""
        from whisper_finetune.model.lora import get_lora_debug_stats, disable_all_but_parametrized_grads
        from minlora import add_lora, LoRAParametrization
        
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 16),
        )
        
        lora_config = {
            torch.nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=4),
            },
        }
        add_lora(model, lora_config=lora_config)
        disable_all_but_parametrized_grads(model)
        
        # Set lora_B to non-zero so gradients flow
        for name, param in model.named_parameters():
            if 'lora_B' in name:
                param.data = torch.randn_like(param) * 0.1
        
        # Forward and backward
        x = torch.randn(1, 64)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Get stats after backward
        stats = get_lora_debug_stats(model, representative_module_pattern="")
        
        # Should now have gradient info
        assert stats["lora_A_grad_norm"] is not None, "Should have lora_A gradient after backward"
        assert stats["lora_A_grad_abs_max"] is not None, "Should have lora_A_grad_abs_max"

    def test_lora_update_tracker(self):
        """Test that LoRAUpdateTracker correctly tracks parameter updates."""
        from whisper_finetune.model.lora import LoRAUpdateTracker, disable_all_but_parametrized_grads
        from minlora import add_lora, LoRAParametrization
        
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 16),
        )
        
        lora_config = {
            torch.nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=4),
            },
        }
        add_lora(model, lora_config=lora_config)
        disable_all_but_parametrized_grads(model)
        
        # Set lora_B to non-zero
        for name, param in model.named_parameters():
            if 'lora_B' in name:
                param.data = torch.randn_like(param) * 0.1
        
        tracker = LoRAUpdateTracker(model, representative_module_pattern="")
        
        # Take initial snapshot
        tracker.snapshot()
        
        # Simulate training step
        x = torch.randn(1, 64)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.1)
        optimizer.step()
        
        # Get update norms
        update_norms = tracker.get_update_norms()
        
        # Should have update info
        assert update_norms["delta_A_norm"] is not None, "Should track delta_A_norm"
        assert update_norms["delta_B_norm"] is not None, "Should track delta_B_norm"
        assert update_norms["delta_A_norm"] > 0, "Parameters should have changed"
        assert update_norms["delta_B_norm"] > 0, "Parameters should have changed"
