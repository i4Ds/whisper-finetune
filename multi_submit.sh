#!/bin/bash
#
# Usage: ./submit_configs.sh
#

# A Bash array of config paths you care about:
configs=(
  "configs/large-v3-sg-corpus-mc_all_specs.yaml"
  "configs/large-v3-sg-corpus-mc_deep_spec.yaml"
  "configs/large-v3-sg-corpus-mc_extreme_specs.yaml"
  "configs/large-v3-sg-corpus-mc.yaml"
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
