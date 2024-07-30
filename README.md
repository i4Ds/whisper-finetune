# Whisper-Finetune

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/i4ds/whisper-finetune.svg)](https://github.com/i4ds/whisper-finetune/issues)

This repository contains code for fine-tuning the Whisper speech-to-text model. It utilizes Weights & Biases (wandb) for logging metrics and storing models. Key features include:

- Timestamp training
- Prompt training
- Stochastic depth implementation for improved model generalization
- Correct implementation of SpecAugment for robust audio data augmentation
- Checkpointing functionality to save and resume training progress, crucial for handling long-running experiments and potential interruptions
- Integration with Weights & Biases (wandb) for experiment tracking and model versioning

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/i4ds/whisper-finetune.git
   cd whisper-finetune
   ```

2. Create and activate a virtual environment (strongly recommended) with Python 3.9.* and a Rust compiler available.

3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Data
Please have a look at https://github.com/i4Ds/whisper-prep. The data is passed as a [🤗 Datasets](https://huggingface.co/docs/datasets/en/index) to the script.

## Usage

1. Create a configuration file (see examples in `configs/*.yaml`)

2. Run the fine-tuning script:
   ```bash
   python src/whisper_finetune/scripts/finetune.py --config configs/large-cv-srg-sg-corpus.yaml
   ```

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
- Claudio Paonessa (claudio.paonessa@fhnw.ch)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
