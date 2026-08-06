#!/bin/bash

# Array of folders and corresponding repo names
declare -a FOLDERS=("63063750" "63063129" "63063128")
declare -a REPO_NAMES=("kenfus/effortless-butterfly-171" "kenfus/jumping-gorge-170" "kenfus/azure-grass-169")

# Loop through each folder/repo pair
for i in "${!FOLDERS[@]}"; do
    FOLDER="${FOLDERS[$i]}"
    REPO_NAME="${REPO_NAMES[$i]}"
    MODEL_DIR="/scicore/home/graber0001/GROUP/stt/data_nobackup/whisper/training_outputs/$FOLDER/last_model.pt"
    
    echo "Processing $REPO_NAME from $FOLDER..."
    
    # Create a new private repo under the organization
    # huggingface-cli repo create "$REPO_NAME" --organization "$ORG" --private
    
    # Upload the model file directly
    huggingface-cli upload "$REPO_NAME" "$MODEL_DIR"
    
    echo "✓ Model uploaded to https://huggingface.co/$ORG/$REPO_NAME (private)"
    echo ""
done

echo "All models uploaded successfully!"