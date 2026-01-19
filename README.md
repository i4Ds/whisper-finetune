# Whisper-Finetune

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/i4ds/whisper-finetune.svg)](https://github.com/i4ds/whisper-finetune/issues)

This repository contains code for fine-tuning the Whisper speech-to-text model. It utilizes Weights & Biases (wandb) for logging metrics and storing models. Key features include:

- **Multi-Dataset Validation** 🆕 - Evaluate on multiple validation sets simultaneously with macro averaging
- **Comprehensive Metrics** 🆕 - WER, CER, NLL, log-probability, entropy, and calibration (ECE)
- **Production-Ready Tests** 🆕 - Fast unit tests with pytest
- Timestamp training
- Prompt training
- Stochastic depth implementation for improved model generalization
- Correct implementation of SpecAugment for robust audio data augmentation
- Checkpointing functionality to save and resume training progress, crucial for handling long-running experiments and potential interruptions
- Integration with Weights & Biases (wandb) for experiment tracking and model versioning

## What's New

### Multi-Dataset Validation System
Evaluate your model on multiple validation datasets (e.g., clean speech, noisy environments, different microphones) with comprehensive metrics beyond WER:

- **6 metrics per dataset**: WER, CER, NLL, log-prob, entropy, ECE
- **Macro averaging**: Unweighted mean across datasets (each dataset contributes equally)
- **Per-utterance tracking**: Detailed metrics for in-depth analysis
- **Smart checkpointing**: All models saved locally, manual W&B upload to avoid clutter

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

3. (Optional) Merge LoRA weights into a standard Whisper checkpoint (saved via `save_model`):
   ```bash
   python src/whisper_finetune/scripts/merge_lora_weights.py \
       --input /path/to/best_model.pt \
       --config configs/config_lora_only.yaml \
     --output /path/to/last_model_merged.pt
   ```

## Testing

Run the test suite to ensure everything is working:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with verbose output and coverage
pytest -v --cov=whisper_finetune
```

See [`tests/README.md`](tests/README.md) for more details.

## Deployment
We suggest to use [faster-whisper](https://github.com/SYSTRAN/faster-whisper). To convert your fine-tuned model, you can use the script located at `src/whisper_finetune/scripts/convert_c2t.py`. 

Further improvement of quality can be archieved by serving the requests with [whisperx](https://github.com/m-bain/whisperX).

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
