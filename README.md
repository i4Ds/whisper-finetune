# Whisper-Finetune
This repository contains code for finetuning the Whisper speech-to-text model. It uses wandb to log metrics and store models.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/whisper-finetune.git
   cd whisper-finetune
    ```
2. Create a virtual enviroment.
3. Install the package in editable mode
 ```bash
 pip install -e . 
 ```

## Run
1. Create a config file (see `configs/*.yaml`)
2. Run script with    
```bash
python src/whisper_finetune/scripts/finetune.py --config configs/large-cv-srg-sg-corpus.yaml
```