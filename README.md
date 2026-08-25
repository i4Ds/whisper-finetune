# Whisper-Finetune

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/i4ds/whisper-finetune.svg)](https://github.com/i4ds/whisper-finetune/issues)

This repository contains code for fine-tuning Whisper speech-to-text models. It supports:

- Multi-dataset validation with macro averages
- WER, CER, NLL, log-probability, entropy, and calibration metrics
- Single-GPU and PyTorch DDP training
- LoRA training and LoRA checkpoint merging
- Timestamp training
- Prompt training
- Stochastic depth, SpecAugment, gradient checkpointing, and mixed precision
- W&B logging and local checkpointing

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/i4ds/whisper-finetune.git
   cd whisper-finetune
   ```

2. Create and activate a virtual environment (strongly recommended) with Python 3.11 or higher.

3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```
   
   Or using UV (very strongly recommended):
   ```bash
   uv pip install -e .
   ```

## Data
Please have a look at https://github.com/i4Ds/whisper-prep. The data is passed as a [🤗 Datasets](https://huggingface.co/docs/datasets/en/index) to the script.

### Timestamp handling

Training records may include Whisper timestamp tokens in the transcript, for example:

```text
<|0.00|> Some text.<|3.64|><|3.66|> More text.<|7.78|>
```

The data loader behaves differently depending on whether no-timestamp training is enabled:

- With `no_timestamps=True`, timestamp tokens are removed from the decoder labels and the `<|notimestamps|>` special token is added.
- With `no_timestamps=False`, timestamp tokens are kept in the decoder labels and the model is trained to predict them.

The two config keys interact, and the boolean wins:

```python
no_timestamps = self.no_timestamp_training or torch.rand(1).item() < self.no_timestamps_rate
```

With `no_timestamp_training: True`, **every** sample is trained with `<|notimestamps|>` and
`no_timestamp_rate` has no effect. To train on a mixture, set `no_timestamp_training: False`
and use `no_timestamp_rate` alone — `0.5` means half the samples get `<|notimestamps|>`.

This matters for deployment. A model trained only without timestamps cannot predict them, and
Whisper's long-form decoding advances its 30 second window using exactly those predictions. Such
a model can score well on short clips and then emit full-width 30 second segments on long audio,
silently dropping speech. Several configs under `configs/` set `no_timestamp_training: True`
together with `no_timestamp_rate: 0.5`; that combination trains with no timestamps at all.

This mirrors the upstream behaviour in
[jumon/whisper-finetuning](https://github.com/jumon/whisper-finetuning), which this repository
started from. There the same short-circuit exists, but it is spelled out in the CLI help for
`--no-timestamps-rate`: *"How often to use the no-timestamps mode. Only used if
`--no-timestamps-training` is NOT set."* The note is worth keeping now that the two settings
live together in YAML, where both are easy to set at once.

Audio is normally padded or trimmed to Whisper's 30 second window. There is one special case in no-timestamp training: if the transcript ends with two consecutive timestamp tokens, such as:

```text
... final text.<|24.28|><|24.94|>
```

the loader treats the last timestamp as the start of a following partial segment. In that case, with `no_timestamps=True`, the mel spectrogram is cut at `24.94` seconds and then padded back to the 30 second training window so the remaining time is represented as silence.

If the transcript ends with only one timestamp token, such as:

```text
... final text.<|25.58|>
```

this special cut is not applied. With `no_timestamps=True`, the timestamp is still removed from the labels, but the audio follows the normal 30 second padding/trimming path.

## Usage

1. Create a configuration file (see `configs/example_config.yaml` for a fully documented example)

2. Run the fine-tuning script:
   ```bash
   python src/whisper_finetune/scripts/finetune.py --config configs/example_config.yaml
   ```

   On SLURM, use the provided batch script:
   ```bash
   sbatch sc_sbatch.sh configs/example_config.yaml
   ```

   For DDP, request multiple GPUs. `sc_sbatch.sh` detects the allocated GPUs through `CUDA_VISIBLE_DEVICES` and launches `torchrun` with one process per GPU:
   ```bash
   sbatch --gres=gpu:4 --cpus-per-task=32 sc_sbatch.sh configs/config_large_v3_best_muon_ddp4.yaml
   ```

   DDP uses `DistributedSampler`, rank-0-only logging/evaluation/checkpointing, and `model.no_sync()` during gradient accumulation. `training.accum_grad_steps` is the global accumulation window and must be divisible by `WORLD_SIZE`; with `accum_grad_steps: 8` and 4 GPUs each rank uses 2 local accumulation steps. The effective batch is:
   ```text
   batch_size * configured_accum_grad_steps
   ```
   so it stays comparable between single-GPU and DDP runs.

3. (Optional) Merge LoRA weights into a standard Whisper checkpoint (saved via `save_model`):
   ```bash
   python src/whisper_finetune/scripts/merge_lora_weights.py \
       --input /path/to/best_model.pt \
       --config configs/config_lora_only.yaml \
       --output /path/to/last_model_merged.pt
   ```

## Testing

Run the test suite:

```bash
pip install -e ".[dev]"
pytest
```

See [`tests/README.md`](tests/README.md) for more details.

## Deployment
We suggest using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). To convert your fine-tuned model, use `src/whisper_finetune/scripts/convert_c2t.py`.

Further quality improvements may be possible by serving requests with [whisperx](https://github.com/m-bain/whisperX).

## Configuration

Modify the YAML files in the `configs/` directory to customize your fine-tuning process. Refer to the existing configuration files for examples of available options.

## Thank you

The starting point of this repository was the excellent repository by [Jumon](https://github.com/jumon) at https://github.com/jumon/whisper-finetuning

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## Support

If you encounter any problems, please file an issue along with a detailed description.

## Maintainer

- Vincenzo Timmel (vincenzo.timmel@fhnw.ch)

## Developers

- Vincenzo Timmel (vincenzo.timmel@fhnw.ch)
- Claudio Paonessa (info@noxenum.io)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
