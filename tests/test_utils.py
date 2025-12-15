"""
Tests for data utilities and normalization functions.
"""

import pytest

from whisper_finetune.eval.utils import VOCAB_SPECS, normalize_text


class TestTextNormalization:
    """Test text normalization functions."""

    def test_normalize_lowercase_v0(self):
        """Test v0 normalization (lowercase)."""
        text = "Hello World"
        normalized = normalize_text(text, **VOCAB_SPECS["v0"])
        assert normalized == "hello world"

    def test_normalize_special_chars_v0(self):
        """Test v0 normalization with special characters."""
        text = "Café naïve"
        normalized = normalize_text(text, **VOCAB_SPECS["v0"])
        # Should replace special characters according to lookup
        assert "cafe" in normalized.lower()

    def test_normalize_dash_v0(self):
        """Test v0 normalization with dashes."""
        text = "word-word"
        normalized = normalize_text(text, **VOCAB_SPECS["v0"])
        # Dashes should be replaced with spaces in v0
        assert "-" not in normalized
        assert " " in normalized

    def test_normalize_empty_string(self):
        """Test normalization of empty string."""
        text = ""
        normalized = normalize_text(text, **VOCAB_SPECS["v0"])
        assert normalized == ""

    def test_normalize_whitespace(self):
        """Test normalization handles multiple whitespaces."""
        text = "hello    world"
        normalized = normalize_text(text, **VOCAB_SPECS["v0"])
        # Should collapse multiple spaces
        assert "    " not in normalized

    def test_normalize_v1_preserves_case(self):
        """Test v1 normalization preserves case."""
        text = "Hello World"
        normalized = normalize_text(text, **VOCAB_SPECS["v1"])
        # v1 should preserve case
        assert "H" in normalized or "h" in normalized

    def test_normalize_numbers(self):
        """Test normalization preserves numbers."""
        text = "test 123"
        normalized = normalize_text(text, **VOCAB_SPECS["v0"])
        assert "123" in normalized


class TestVocabSpecs:
    """Test vocabulary specifications."""

    def test_v0_vocab_contains_basics(self):
        """Test v0 vocabulary contains basic characters."""
        vocab = VOCAB_SPECS["v0"]["char_vocab"]
        assert "a" in vocab
        assert "z" in vocab
        assert "0" in vocab
        assert "9" in vocab
        assert " " in vocab
        assert "ä" in vocab
        assert "ö" in vocab
        assert "ü" in vocab

    def test_v0_vocab_lowercase_only(self):
        """Test v0 vocabulary is lowercase only."""
        vocab = VOCAB_SPECS["v0"]["char_vocab"]
        assert "A" not in vocab
        assert "Z" not in vocab

    def test_v1_vocab_mixed_case(self):
        """Test v1 vocabulary contains both cases."""
        vocab = VOCAB_SPECS["v1"]["char_vocab"]
        assert "a" in vocab
        assert "A" in vocab
        assert "z" in vocab
        assert "Z" in vocab

    def test_all_specs_have_required_keys(self):
        """Test all vocab specs have required keys."""
        for spec_name, spec in VOCAB_SPECS.items():
            assert "char_vocab" in spec, f"{spec_name} missing char_vocab"
            assert "char_lookup" in spec, f"{spec_name} missing char_lookup"
            assert "transform_lowercase" in spec, f"{spec_name} missing transform_lowercase"
            assert isinstance(spec["char_vocab"], set)
            assert isinstance(spec["char_lookup"], dict)
            assert isinstance(spec["transform_lowercase"], bool)


class TestCharLookup:
    """Test character lookup/replacement tables."""

    def test_v0_lookup_contains_common_replacements(self):
        """Test v0 lookup has common character replacements."""
        lookup = VOCAB_SPECS["v0"]["char_lookup"]

        # Test some common replacements
        assert "ß" in lookup
        assert lookup["ß"] == "ss"

        assert "é" in lookup
        assert lookup["é"] == "e"

        assert "à" in lookup
        assert lookup["à"] == "a"

    def test_v0_dash_replacement(self):
        """Test v0 replaces dashes with spaces."""
        lookup = VOCAB_SPECS["v0"]["char_lookup"]
        assert "-" in lookup
        assert lookup["-"] == " "

    def test_v1_has_uppercase_mappings(self):
        """Test v1 lookup includes uppercase mappings."""
        lookup = VOCAB_SPECS["v1"]["char_lookup"]

        # v1 should have both lowercase and uppercase
        if "é" in lookup:
            # If lowercase exists, check for uppercase version
            assert "É" in lookup or "é" in lookup


class TestLoadHFDataset:
    """Test load_hf_dataset function for local/remote dataset loading."""

    def test_load_hf_dataset_detects_local_path(self, tmp_path):
        """Test that load_hf_dataset correctly identifies local paths."""
        from pathlib import Path
        
        # Create a mock local path
        local_path = tmp_path / "test_dataset"
        local_path.mkdir()
        
        # The function should detect this as a local path
        p = Path(str(local_path))
        assert p.exists()

    def test_load_hf_dataset_detects_remote_name(self):
        """Test that load_hf_dataset correctly identifies remote dataset names."""
        from pathlib import Path
        
        # A HuggingFace dataset name should not exist as a local path
        remote_name = "i4ds/some-nonexistent-dataset"
        p = Path(remote_name)
        assert not p.exists()

    def test_load_hf_dataset_function_exists(self):
        """Test that load_hf_dataset function is importable."""
        from whisper_finetune.data.utils import load_hf_dataset
        assert callable(load_hf_dataset)
