#!/bin/bash

# Define the dataset to download and your Hugging Face token
DATASET_NAME="i4ds/SDS_STT_SPC_mixed"

# Create a virtual environment in the current directory
python3 -m venv .hu_ds_download

# Get env variables
export $(cat .env-dataset | xargs)

# Activate the virtual environment
source .hu_ds_download/bin/activate

# Update pip to its latest version
pip install --upgrade pip

# Install the datasets library by Hugging Face
pip install datasets

# Use the load_dataset function from the datasets library
python3 -c "from datasets import load_dataset; dataset = load_dataset('$DATASET_NAME')"

# Deactivate the virtual environment when done
deactivate
