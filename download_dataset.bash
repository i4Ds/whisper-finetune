#!/bin/bash

# Define the datasets to download
DATASET_NAMES=("i4ds/p5-synthetic-data-for-stt-noise-train-val" "i4ds/p5-synthetic-data-for-stt-real-1-train-val" "i4ds/p5-synthetic-data-for-stt-generated-1-train-val")

# Create a virtual environment in the current directory
python3 -m venv .hu_ds_download

# Get env variables
export $(cat .env | xargs)

# Overwrite some variables
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# Activate the virtual environment
source .hu_ds_download/bin/activate

# Update pip to its latest version
pip install --upgrade pip

# Install the datasets library by Hugging Face
pip install datasets

# Loop through the datasets and download each one
for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    python3 -c "from datasets import load_dataset; dataset = load_dataset('$DATASET_NAME', download_mode='force_redownload')"
done

# Deactivate the virtual environment when done
deactivate
