#!/bin/bash

CONFIG_FILES=("configs/medium_gc.yaml" "configs/medium_mp_gc.yaml")

for config in "${CONFIG_FILES[@]}"; do
    sbatch sbatch_i4ds.sh $config
done
