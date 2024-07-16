# Whisper-Finetune

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/i4ds/whisper-finetune.svg)](https://github.com/i4ds/whisper-finetune/issues)

This repository contains code for fine-tuning the Whisper speech-to-text model. It utilizes Weights & Biases (wandb) for logging metrics and storing models.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/i4ds/whisper-finetune.git
   cd whisper-finetune
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Usage

1. Create a configuration file (see examples in `configs/*.yaml`)

2. Run the fine-tuning script:
   ```bash
   python src/whisper_finetune/scripts/finetune.py --config configs/large-cv-srg-sg-corpus.yaml
   ```

## Configuration

Modify the YAML files in the `configs/` directory to customize your fine-tuning process. Refer to the existing configuration files for examples of available options.

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