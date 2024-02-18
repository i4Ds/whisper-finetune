#!/bin/bash

# List of configuration files
configs=("configs/tiny.yaml" "configs/tiny_mp.yaml" "configs/tiny_mp_gc.yaml" "configs/tiny_mp_bfloat.yaml" "configs/tiny_mp_bfloat_gc.yaml" "configs/tiny_gc.yaml")

# Loop over the configuration files
for config in ${configs[@]}; do
    # Run the command with the current configuration file
    python src/whisper_finetune/scripts/finetune.py --config $config

    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Command with configuration $config failed, aborting."
        exit 1
    fi
done

echo "All commands completed successfully."