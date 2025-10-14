#!/bin/bash
#
# Usage: ./submit_configs.sh
#

# A Bash array of config paths you care about:
configs=(
  configs/config_1.yaml
  configs/config_2.yaml
  configs/config_3.yaml
  configs/config_4.yaml
  configs/config_5.yaml
  configs/config_6.yaml
  configs/config_7.yaml
)

for cfg in "${configs[@]}"; do
  if [ ! -f "$cfg" ]; then
    echo "Warning: config file '$cfg' not found, skipping."
    continue
  fi

  base=$(basename "$cfg" .yaml)
  echo "Submitting SLURM job for config: $cfg → job-name=whisper_finetune_${base}"
  sbatch --job-name=whisper_finetune_${base} sc_sbatch.sh "$cfg"
done
