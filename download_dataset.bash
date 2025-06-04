#!/bin/bash

# Define the datasets to download
DATASET_NAMES=("i4ds/sg_corp_train_no_overlap_speaker_ret" "i4ds/srg-full-train-val-v2" "i4ds/mozilla-cv-13-long-text-de")

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
    python3 -c "from datasets import load_dataset; dataset = load_dataset('$DATASET_NAME')"
done

# Deactivate the virtual environment when done
deactivate
