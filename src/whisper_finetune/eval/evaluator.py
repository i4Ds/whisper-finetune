"""
Multi-dataset evaluator for Whisper fine-tuning.
Evaluates model on multiple validation datasets and computes comprehensive metrics.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from whisper import Whisper
from whisper.tokenizer import get_tokenizer

from whisper_finetune.eval.metrics import (
    DatasetMetrics,
    PerUtteranceMetrics,
    aggregate_dataset_metrics,
    compute_cer_batch,
    compute_macro_average,
    compute_token_metrics,
    compute_wer,
)
from whisper_finetune.eval.utils import VOCAB_SPECS, normalize_text


@torch.no_grad()
def evaluate_single_dataset(
    model: Whisper,
    dataloader: DataLoader,
    dataset_name: str,
    t_config: dict,
    tokenizer=None,
) -> DatasetMetrics:
    """
    Evaluate model on a single dataset with comprehensive metrics.

    Args:
        model: Whisper model to evaluate
        dataloader: DataLoader for the validation dataset
        dataset_name: Name of the dataset (for logging)
        t_config: Training configuration dict
        tokenizer: Tokenizer (if None, creates a new one)

    Returns:
        DatasetMetrics object with all computed metrics
    """
    model.eval()

    # Read variables from t_config
    mixed_precision_training = t_config.get("mixed_precision_training", True)
    mp_dtype = torch.float16 if t_config.get("mp_dtype", "fp16") == "fp16" else torch.bfloat16

    # Get tokenizer
    if tokenizer is None:
        tokenizer = get_tokenizer(multilingual=True, language="de", task="transcribe")

    per_utterance_metrics = []

    for x, y_in, y_out in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        x, y_in, y_out = x.to(model.device), y_in.to(model.device), y_out.to(model.device)

        with torch.autocast(device_type="cuda", enabled=mixed_precision_training, dtype=mp_dtype):
            logits = model(x, y_in)

            # Convert logits to token IDs
            pred_token_ids = torch.argmax(logits, dim=-1)

            # Process each sample in the batch
            for i in range(logits.size(0)):
                sample_logits = logits[i]
                sample_pred_ids = pred_token_ids[i]
                sample_true_ids = y_out[i]

                # Decode predictions and references
                pred_tokens = [
                    id
                    for id in sample_pred_ids.cpu().tolist()
                    if id not in tokenizer.special_tokens.values() and id != -100
                ]
                true_tokens = [
                    id
                    for id in sample_true_ids.cpu().tolist()
                    if id not in tokenizer.special_tokens.values() and id != -100
                ]

                pred_text = tokenizer.decode(pred_tokens)
                true_text = tokenizer.decode(true_tokens)

                # Skip empty references
                if true_text.strip() == "":
                    continue

                # Normalize texts
                pred_normalized = normalize_text(pred_text, **VOCAB_SPECS["v0"])
                true_normalized = normalize_text(true_text, **VOCAB_SPECS["v0"])

                # Compute WER and CER
                from jiwer import compute_measures

                wer_measures = compute_measures(true_normalized, pred_normalized)
                wer_val = wer_measures["wer"]

                from jiwer import cer

                cer_val = cer(true_normalized, pred_normalized)

                # Compute token-level metrics
                mean_nll, avg_log_prob, mean_entropy, confidences, correct = compute_token_metrics(
                    sample_logits, sample_true_ids, sample_pred_ids
                )

                # Store per-utterance metrics
                per_utterance_metrics.append(
                    PerUtteranceMetrics(
                        prediction=pred_normalized,
                        reference=true_normalized,
                        wer=wer_val,
                        cer=cer_val,
                        token_nll=mean_nll,
                        avg_log_prob=avg_log_prob,
                        token_entropy=mean_entropy,
                        token_confidences=confidences,
                        token_correct=correct,
                    )
                )

    # Aggregate metrics for the dataset
    dataset_metrics = aggregate_dataset_metrics(per_utterance_metrics, dataset_name)

    return dataset_metrics


def evaluate_multiple_datasets(
    model: Whisper,
    dataloaders: Dict[str, DataLoader],
    t_config: dict,
) -> Tuple[List[DatasetMetrics], Dict[str, float]]:
    """
    Evaluate model on multiple validation datasets.

    Args:
        model: Whisper model to evaluate
        dataloaders: Dictionary mapping dataset names to their DataLoaders
        t_config: Training configuration dict

    Returns:
        Tuple of:
            - List of DatasetMetrics (one per dataset)
            - Dictionary of macro-averaged metrics
    """
    tokenizer = get_tokenizer(multilingual=True, language="de", task="transcribe")

    all_dataset_metrics = []

    for dataset_name, dataloader in dataloaders.items():
        print(f"\n{'='*60}")
        print(f"Evaluating dataset: {dataset_name}")
        print(f"{'='*60}")

        dataset_metrics = evaluate_single_dataset(model, dataloader, dataset_name, t_config, tokenizer)
        all_dataset_metrics.append(dataset_metrics)

        # Print dataset-level metrics
        print(f"\nResults for {dataset_name}:")
        print(f"  Samples: {dataset_metrics.num_samples}")
        print(f"  WER: {dataset_metrics.wer:.4f}")
        print(f"  CER: {dataset_metrics.cer:.4f}")
        print(f"  Mean Token NLL: {dataset_metrics.mean_token_nll:.4f}")
        print(f"  Avg Log Prob: {dataset_metrics.avg_log_prob:.4f}")
        print(f"  Mean Token Entropy: {dataset_metrics.mean_token_entropy:.4f}")
        print(f"  ECE: {dataset_metrics.ece:.4f}")

    # Compute macro averages
    macro_metrics = compute_macro_average(all_dataset_metrics)

    print(f"\n{'='*60}")
    print("MACRO AVERAGES (unweighted across datasets)")
    print(f"{'='*60}")
    for metric_name, metric_value in macro_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    return all_dataset_metrics, macro_metrics


def log_metrics_to_wandb(
    dataset_metrics: List[DatasetMetrics],
    macro_metrics: Dict[str, float],
    step: int,
    prefix: str = "val",
):
    """
    Log all metrics to Weights & Biases.

    Args:
        dataset_metrics: List of metrics for each dataset
        macro_metrics: Dictionary of macro-averaged metrics
        step: Current training step
        prefix: Prefix for metric names (default: "val")
    """
    import wandb

    log_dict = {}

    # Log per-dataset metrics
    for dm in dataset_metrics:
        ds_name = dm.dataset_name
        log_dict[f"{prefix}/{ds_name}_wer"] = dm.wer
        log_dict[f"{prefix}/{ds_name}_cer"] = dm.cer
        log_dict[f"{prefix}/{ds_name}_loss"] = dm.mean_token_nll
        log_dict[f"{prefix}/{ds_name}_mean_token_nll"] = dm.mean_token_nll
        log_dict[f"{prefix}/{ds_name}_avg_log_prob"] = dm.avg_log_prob
        log_dict[f"{prefix}/{ds_name}_mean_token_entropy"] = dm.mean_token_entropy
        log_dict[f"{prefix}/{ds_name}_ece"] = dm.ece
        log_dict[f"{prefix}/{ds_name}_num_samples"] = dm.num_samples

    # Log macro averages
    for metric_name, metric_value in macro_metrics.items():
        log_dict[f"{prefix}/{metric_name}"] = metric_value

    # Add step
    log_dict["step"] = step

    wandb.log(log_dict)
