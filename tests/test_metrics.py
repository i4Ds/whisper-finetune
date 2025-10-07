"""
Tests for evaluation metrics module.
"""

import numpy as np
import pytest
import torch

from whisper_finetune.eval.metrics import (
    DatasetMetrics,
    PerUtteranceMetrics,
    aggregate_dataset_metrics,
    compute_cer_batch,
    compute_ece,
    compute_macro_average,
    compute_token_metrics,
    compute_wer,
)


class TestWERComputation:
    """Test Word Error Rate computation."""

    def test_perfect_match(self):
        """Test WER for perfect predictions."""
        predictions = ["hello world"]
        references = ["hello world"]
        wers = compute_wer(predictions, references)
        assert wers[0] == 0.0

    def test_complete_mismatch(self):
        """Test WER for completely wrong predictions."""
        predictions = ["foo bar"]
        references = ["hello world"]
        wers = compute_wer(predictions, references)
        assert wers[0] > 0.0

    def test_empty_reference(self):
        """Test WER when reference is empty."""
        predictions = ["hello"]
        references = [""]
        wers = compute_wer(predictions, references)
        assert wers[0] == 1.0  # Should be 1.0 when prediction is not empty

        predictions = [""]
        references = [""]
        wers = compute_wer(predictions, references)
        assert wers[0] == 0.0  # Should be 0.0 when both are empty

    def test_multiple_utterances(self):
        """Test WER for multiple utterances."""
        predictions = ["hello world", "foo bar", "test case"]
        references = ["hello world", "hello world", "test case"]
        wers = compute_wer(predictions, references)
        assert len(wers) == 3
        assert wers[0] == 0.0  # perfect match
        assert wers[1] > 0.0  # mismatch
        assert wers[2] == 0.0  # perfect match


class TestCERComputation:
    """Test Character Error Rate computation."""

    def test_perfect_match(self):
        """Test CER for perfect predictions."""
        predictions = ["hello"]
        references = ["hello"]
        cers = compute_cer_batch(predictions, references)
        assert cers[0] == 0.0

    def test_single_char_error(self):
        """Test CER for single character error."""
        predictions = ["hallo"]
        references = ["hello"]
        cers = compute_cer_batch(predictions, references)
        assert cers[0] > 0.0
        assert cers[0] <= 1.0


class TestTokenMetrics:
    """Test token-level metrics computation."""

    def test_perfect_predictions(self):
        """Test metrics when all tokens are predicted correctly."""
        vocab_size = 10
        seq_len = 5

        # Create dummy data where predictions match targets
        logits = torch.randn(seq_len, vocab_size)
        target_ids = torch.randint(0, vocab_size, (seq_len,))

        # Make logits favor the correct tokens
        for i, target in enumerate(target_ids):
            logits[i, target] = 10.0  # Very high logit for correct token

        predicted_ids = torch.argmax(logits, dim=-1)

        mean_nll, avg_log_prob, mean_entropy, confidences, correct = compute_token_metrics(
            logits, target_ids, predicted_ids
        )

        # All tokens should be correct
        assert all(correct), "All tokens should be predicted correctly"

        # NLL should be low
        assert mean_nll < 1.0, "NLL should be low for correct predictions"

        # Confidences should be high
        assert all(c > 0.9 for c in confidences), "Confidences should be high"

    def test_with_padding(self):
        """Test metrics with padded sequences (containing -100)."""
        vocab_size = 10
        seq_len = 5

        logits = torch.randn(seq_len, vocab_size)
        target_ids = torch.tensor([1, 2, 3, -100, -100])  # Last two are padding
        predicted_ids = torch.tensor([1, 2, 3, 5, 5])

        mean_nll, avg_log_prob, mean_entropy, confidences, correct = compute_token_metrics(
            logits, target_ids, predicted_ids
        )

        # Should only compute metrics for first 3 tokens
        assert len(confidences) == 3
        assert len(correct) == 3

    def test_empty_sequence(self):
        """Test metrics with all tokens padded."""
        vocab_size = 10
        seq_len = 3

        logits = torch.randn(seq_len, vocab_size)
        target_ids = torch.tensor([-100, -100, -100])
        predicted_ids = torch.tensor([1, 2, 3])

        mean_nll, avg_log_prob, mean_entropy, confidences, correct = compute_token_metrics(
            logits, target_ids, predicted_ids
        )

        # All should be zero/empty for empty sequence
        assert mean_nll == 0.0
        assert avg_log_prob == 0.0
        assert mean_entropy == 0.0
        assert len(confidences) == 0
        assert len(correct) == 0


class TestECE:
    """Test Expected Calibration Error computation."""

    def test_perfect_calibration(self):
        """Test ECE for perfectly calibrated predictions."""
        # Create perfectly calibrated data
        n_samples = 1000
        np.random.seed(42)
        confidences = np.random.uniform(0, 1, n_samples)
        correct = (np.random.uniform(0, 1, n_samples) < confidences).tolist()
        confidences = confidences.tolist()

        ece = compute_ece(confidences, correct, n_bins=10)

        # Should be close to 0 for large random sample
        assert ece < 0.2, "ECE should be relatively low for random calibrated data"

    def test_overconfident(self):
        """Test ECE for overconfident predictions."""
        # All predictions have high confidence but are wrong
        confidences = [0.9] * 100
        correct = [False] * 100

        ece = compute_ece(confidences, correct, n_bins=10)

        # ECE should be high (close to 0.9)
        assert ece > 0.5, "ECE should be high for overconfident wrong predictions"

    def test_underconfident(self):
        """Test ECE for underconfident predictions."""
        # All predictions have low confidence but are correct
        confidences = [0.2] * 100
        correct = [True] * 100

        ece = compute_ece(confidences, correct, n_bins=10)

        # ECE should be high (close to 0.8)
        assert ece > 0.5, "ECE should be high for underconfident correct predictions"

    def test_empty_input(self):
        """Test ECE with empty input."""
        ece = compute_ece([], [], n_bins=10)
        assert ece == 0.0


class TestDatasetAggregation:
    """Test dataset metrics aggregation."""

    def test_aggregate_multiple_utterances(self):
        """Test aggregation of multiple utterances."""
        per_utterance = [
            PerUtteranceMetrics(
                prediction="hello",
                reference="hello",
                wer=0.0,
                cer=0.0,
                token_nll=0.1,
                avg_log_prob=-0.5,
                token_entropy=0.2,
                token_confidences=[0.9, 0.95],
                token_correct=[True, True],
            ),
            PerUtteranceMetrics(
                prediction="world",
                reference="word",
                wer=0.5,
                cer=0.4,
                token_nll=0.3,
                avg_log_prob=-1.0,
                token_entropy=0.4,
                token_confidences=[0.7, 0.8],
                token_correct=[True, False],
            ),
        ]

        dataset_metrics = aggregate_dataset_metrics(per_utterance, "test_dataset")

        assert dataset_metrics.dataset_name == "test_dataset"
        assert dataset_metrics.num_samples == 2
        assert dataset_metrics.wer == 0.25  # mean of [0.0, 0.5]
        assert dataset_metrics.cer == 0.2  # mean of [0.0, 0.4]
        assert dataset_metrics.mean_token_nll == 0.2  # mean of [0.1, 0.3]

    def test_aggregate_empty(self):
        """Test aggregation with no utterances."""
        dataset_metrics = aggregate_dataset_metrics([], "empty_dataset")

        assert dataset_metrics.dataset_name == "empty_dataset"
        assert dataset_metrics.num_samples == 0
        assert dataset_metrics.wer == 0.0
        assert dataset_metrics.ece == 0.0


class TestMacroAverage:
    """Test macro averaging across datasets."""

    def test_macro_average_equal_weight(self):
        """Test that macro average weights all datasets equally."""
        # Create datasets of different sizes
        dataset1 = DatasetMetrics(
            dataset_name="small",
            num_samples=10,  # Small dataset
            wer=0.1,
            cer=0.05,
            mean_token_nll=0.2,
            avg_log_prob=-0.5,
            mean_token_entropy=0.3,
            ece=0.1,
            per_utterance=[],
        )

        dataset2 = DatasetMetrics(
            dataset_name="large",
            num_samples=1000,  # Large dataset
            wer=0.9,
            cer=0.85,
            mean_token_nll=1.0,
            avg_log_prob=-2.0,
            mean_token_entropy=1.5,
            ece=0.5,
            per_utterance=[],
        )

        macro = compute_macro_average([dataset1, dataset2])

        # Macro average should be simple mean (not weighted by size)
        assert macro["macro_wer"] == 0.5  # (0.1 + 0.9) / 2
        assert macro["macro_cer"] == 0.45  # (0.05 + 0.85) / 2
        assert macro["macro_mean_token_nll"] == 0.6  # (0.2 + 1.0) / 2

    def test_macro_average_empty(self):
        """Test macro average with no datasets."""
        macro = compute_macro_average([])

        assert macro["macro_wer"] == 0.0
        assert macro["macro_cer"] == 0.0
