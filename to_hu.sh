#!/bin/bash

# Set your Hugging Face token (recommended: export HF_TOKEN in your .env or shell)
# export HF_TOKEN=your_hf_token_here

# Variables
MODEL_DIR="58434435_/scicore/home/graber0001/GROUP/stt/data_nobackup/whisper/training_outputs/last_model.pt"   # Change to your model directory
REPO_NAME="worthy-sea-158"                   # Change to your desired repo name
ORG="i4ds"

# Create a new private repo under the organization
huggingface-cli repo create "$REPO_NAME" --organization "$ORG" --private

# Clone the repo locally
git clone https://huggingface.co/"$ORG"/"$REPO_NAME"
cd "$REPO_NAME"

# Copy model files into the repo
cp "$MODEL_DIR"/* .

# Add, commit, and push files
git add .
git commit -m "Add PyTorch model"
git push

echo "Model pushed to https://huggingface.co/$ORG/$REPO_NAME (private)"