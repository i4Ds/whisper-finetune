"""
Advanced evaluation metrics for Whisper fine-tuning.
Includes WER, CER, NLL, log-prob, entropy, and calibration metrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from jiwer import cer


@dataclass
class PerUtteranceMetrics:
    """Metrics computed for a single utterance."""

    prediction: str
    reference: str
    wer: float
    cer: float
    token_nll: float  # Mean negative log-likelihood per token
    avg_log_prob: float  # Average log-probability of decoded sequence
    token_entropy: float  # Mean entropy per token
    token_confidences: List[float]  # Confidence (max prob) for each token
    token_correct: List[bool]  # Whether each token was predicted correctly


@dataclass
class DatasetMetrics:
    """Aggregated metrics for a dataset."""

    dataset_name: str
    num_samples: int
    wer: float
    cer: float
    mean_token_nll: float
    avg_log_prob: float
    mean_token_entropy: float
    ece: float  # Expected Calibration Error
    per_utterance: List[PerUtteranceMetrics]


def compute_wer(predictions: List[str], references: List[str]) -> List[float]:
    """
    Compute Word Error Rate for each prediction-reference pair.

    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions

    Returns:
        List of WER values (one per utterance)
    """
    from jiwer import compute_measures

    wers = []
    for pred, ref in zip(predictions, references):
        if ref.strip() == "":
            wers.append(0.0 if pred.strip() == "" else 1.0)
        else:
            measures = compute_measures(ref, pred)
            wers.append(measures["wer"])
    return wers


def compute_cer_batch(predictions: List[str], references: List[str]) -> List[float]:
    """
    Compute Character Error Rate for each prediction-reference pair.

    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions

    Returns:
        List of CER values (one per utterance)
    """
    cers = []
    for pred, ref in zip(predictions, references):
        if ref.strip() == "":
            cers.append(0.0 if pred.strip() == "" else 1.0)
        else:
            cers.append(cer(ref, pred))
    return cers


def compute_token_metrics(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    predicted_ids: torch.Tensor,
) -> Tuple[float, float, float, List[float], List[bool]]:
    """
    Compute token-level metrics from logits.

    Args:
        logits: Model logits (seq_len, vocab_size)
        target_ids: True token IDs (seq_len,)
        predicted_ids: Predicted token IDs (seq_len,)

    Returns:
        Tuple of:
            - mean_nll: Mean negative log-likelihood per token
            - avg_log_prob: Average log-probability of the sequence
            - mean_entropy: Mean entropy per token
            - confidences: List of max probabilities per token
            - correct: List of booleans indicating if each token was correct
    """
    # Filter out padding tokens (-100)
    valid_mask = target_ids != -100
    if valid_mask.sum() == 0:
        return 0.0, 0.0, 0.0, [], []

    valid_logits = logits[valid_mask]
    valid_targets = target_ids[valid_mask]
    valid_preds = predicted_ids[valid_mask]

    # Compute log probabilities
    log_probs = F.log_softmax(valid_logits, dim=-1)
    probs = F.softmax(valid_logits, dim=-1)

    # Negative log-likelihood per token
    nll_per_token = F.cross_entropy(valid_logits, valid_targets, reduction="none")
    mean_nll = nll_per_token.mean().item()

    # Average log-probability of predicted sequence
    pred_log_probs = log_probs.gather(1, valid_preds.unsqueeze(1)).squeeze(1)
    avg_log_prob = pred_log_probs.mean().item()

    # Entropy per token
    entropy_per_token = -(probs * log_probs).sum(dim=-1)
    mean_entropy = entropy_per_token.mean().item()

    # Token confidences (max probability)
    confidences = probs.max(dim=-1).values.cpu().tolist()

    # Token correctness
    correct = (valid_preds == valid_targets).cpu().tolist()

    return mean_nll, avg_log_prob, mean_entropy, confidences, correct


def compute_ece(all_confidences: List[float], all_correct: List[bool], n_bins: int = 20) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the calibration of a model's confidence scores.
    It bins predictions by confidence and computes the weighted average
    of the absolute difference between confidence and accuracy in each bin.

    Args:
        all_confidences: List of confidence scores (max probability per token)
        all_correct: List of booleans indicating correctness
        n_bins: Number of bins for confidence scores (default: 20)

    Returns:
        ECE value (lower is better, 0 is perfectly calibrated)
    """
    if len(all_confidences) == 0:
        return 0.0

    confidences = np.array(all_confidences)
    correct = np.array(all_correct, dtype=float)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find indices in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)

    return ece


def aggregate_dataset_metrics(per_utterance_metrics: List[PerUtteranceMetrics], dataset_name: str) -> DatasetMetrics:
    """
    Aggregate per-utterance metrics into dataset-level metrics.

    Args:
        per_utterance_metrics: List of metrics for each utterance
        dataset_name: Name of the dataset

    Returns:
        Aggregated dataset metrics
    """
    if len(per_utterance_metrics) == 0:
        return DatasetMetrics(
            dataset_name=dataset_name,
            num_samples=0,
            wer=0.0,
            cer=0.0,
            mean_token_nll=0.0,
            avg_log_prob=0.0,
            mean_token_entropy=0.0,
            ece=0.0,
            per_utterance=[],
        )

    # Aggregate scalar metrics
    wer = np.mean([m.wer for m in per_utterance_metrics])
    cer_val = np.mean([m.cer for m in per_utterance_metrics])
    mean_token_nll = np.mean([m.token_nll for m in per_utterance_metrics])
    avg_log_prob = np.mean([m.avg_log_prob for m in per_utterance_metrics])
    mean_token_entropy = np.mean([m.token_entropy for m in per_utterance_metrics])

    # Collect all token-level data for ECE
    all_confidences = []
    all_correct = []
    for m in per_utterance_metrics:
        all_confidences.extend(m.token_confidences)
        all_correct.extend(m.token_correct)

    ece = compute_ece(all_confidences, all_correct)

    return DatasetMetrics(
        dataset_name=dataset_name,
        num_samples=len(per_utterance_metrics),
        wer=wer,
        cer=cer_val,
        mean_token_nll=mean_token_nll,
        avg_log_prob=avg_log_prob,
        mean_token_entropy=mean_token_entropy,
        ece=ece,
        per_utterance=per_utterance_metrics,
    )


def compute_macro_average(dataset_metrics: List[DatasetMetrics]) -> Dict[str, float]:
    """
    Compute macro average (unweighted mean) across all datasets.

    This ensures that all datasets contribute equally to the overall metric,
    regardless of their size.

    Args:
        dataset_metrics: List of metrics for each dataset

    Returns:
        Dictionary of macro-averaged metrics
    """
    if len(dataset_metrics) == 0:
        return {
            "macro_wer": 0.0,
            "macro_cer": 0.0,
            "macro_mean_token_nll": 0.0,
            "macro_avg_log_prob": 0.0,
            "macro_mean_token_entropy": 0.0,
            "macro_ece": 0.0,
        }

    return {
        "macro_wer": np.mean([m.wer for m in dataset_metrics]),
        "macro_cer": np.mean([m.cer for m in dataset_metrics]),
        "macro_mean_token_nll": np.mean([m.mean_token_nll for m in dataset_metrics]),
        "macro_avg_log_prob": np.mean([m.avg_log_prob for m in dataset_metrics]),
        "macro_mean_token_entropy": np.mean([m.mean_token_entropy for m in dataset_metrics]),
        "macro_ece": np.mean([m.ece for m in dataset_metrics]),
    }
