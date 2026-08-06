#!/bin/bash
#SBATCH --job-name=whisper_finetune_${CONFIG_NAME:-job}   # job name (we'll override CONFIG_NAME below)
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=168:00:00
#SBATCH --partition=a100-80g # A100 80GB
#SBATCH --gres=gpu:1
#SBATCH --qos=a100-1week
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=logs/whisper_finetune_%j.out    # name of stdout output file
#SBATCH --error=logs/whisper_finetune_%j.err     # name of stderr
#SBATCH --mail-user=vincenzo293@gmail.com

# If you want each job to have a unique SLURM job name based on the config file name,
# uncomment the following two lines.  They’ll set CONFIG_NAME to the basename of $1 (no extension).
# CONFIG_NAME=$(basename "$1" .yaml)
# sbatch --job-name=whisper_finetune_${CONFIG_NAME} "$0" "$@"

# Activate your conda environment
eval "$(conda shell.bash hook)"
conda activate whisper_finetune


source ./set_proxy.sh

# Load any environment variables from .env
export $(cat .env | xargs)

nvidia-smi
export LD_LIBRARY_PATH=/scicore/soft/eessi/host_injections/default/x86_64/software/CUDA/12.4.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# TQDM spam reduction
TQDM_MININTERVAL=60
export PYTHONUNBUFFERED=1

# Run the Python finetuning script, passing in the config file as "$1"
NPROC=$(python -c "import torch; print(torch.cuda.device_count())")
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "torchrun nproc_per_node=${NPROC}"
torchrun --standalone --nproc_per_node="${NPROC}" src/whisper_finetune/scripts/finetune.py --config "$1"
