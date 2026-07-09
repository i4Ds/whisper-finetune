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
