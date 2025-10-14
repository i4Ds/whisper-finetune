#!/bin/bash

# Define the datasets to download
DATASET_NAMES=(
    "i4ds/srg_v3_pl_valid"
)


# Get env variables
export $(cat .env | xargs)

# Overwrite some variables
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# Loop through the datasets and download each one
for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    python3 -c "from datasets import load_dataset; dataset = load_dataset('$DATASET_NAME', download_mode='force_redownload')"
done
