#!/bin/bash
#SBATCH --job-name=debug_whisper_finetune     # create a short name for your job
#SBATCH --cpus-per-task=8                  #This is the number of cores reserved
#SBATCH --mem-per-cpu=8G              #This is the memory reserved per core.
#SBATCH --time=00:30:00         # total run time limit (HH:MM:SS)
#SBATCH --partition=h200     # or titanx
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --output=logs/debug_whisper_finetune_%j.out    # name of stdout output file
#SBATCH --error=logs/debug_whisper_finetune_%j.err     # name of stderr
#SBATCH --qos=h200-30min      # qos level

# Optional config argument (defaults to DEBUG baseline)
CONFIG_PATH="${1:-configs/DEBUG.yaml}"

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate whisper_finetune

nvidia-smi
export LD_LIBRARY_PATH=/scicore/soft/eessi/host_injections/default/x86_64/software/CUDA/12.4.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
source ./set_proxy.sh

# Get env variables
export $(cat .env | xargs)

echo "Running debug with config: ${CONFIG_PATH}"
NPROC=$(python -c "import torch; print(torch.cuda.device_count())")
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "torchrun nproc_per_node=${NPROC}"
torchrun --standalone --nproc_per_node="${NPROC}" src/whisper_finetune/scripts/finetune.py --config "${CONFIG_PATH}"
