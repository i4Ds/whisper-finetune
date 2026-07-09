#!/bin/bash
#
# Usage: ./submit_configs.sh
#

# A Bash array of config paths you care about:
configs=(
  configs/experiments/config_large_v3_best_muon_multilingual_downsized_srg180k_pnv_10h.yaml
  configs/experiments/config_large_v3_best_muon_multilingual_downsized_srg180k_pnv_30h.yaml
  configs/experiments/config_large_v3_best_muon_multilingual_downsized_srg180k_pnv_50h.yaml
  configs/experiments/config_large_v3_best_muon_multilingual_downsized_srg180k_pnv_100h.yaml
  configs/experiments/config_large_v3_best_muon_multilingual_downsized_srg180k_pnv_200h.yaml
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
