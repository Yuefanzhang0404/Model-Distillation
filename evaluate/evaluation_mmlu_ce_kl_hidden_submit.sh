#!/bin/bash
#SBATCH --partition=pt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH -w speed-42
#SBATCH --job-name=evaluation_mmlu_ce_kl_hidden
#SBATCH --output=/speed-scratch/%u/distill-project/original/evaluation_mmlu_ce_kl_hidden_%j.out
#SBATCH --error=/speed-scratch/%u/distill-project/original/evaluation_mmlu_ce_kl_hidden_%j.err

set -eo pipefail

MODEL_NAME="${1:-/nfs/speed-scratch/z_yuefan/distill-project/original/runs/qwen3b_smol360m_ce_kl_hidden/best}"

cd /speed-scratch/$USER/distill-project/original || exit 1

source /etc/profile
module load anaconda3/2023.03/default
source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate /speed-scratch/$USER/distill-env

echo "Job started"
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "Experiment: ce_kl_hidden"
echo "Model checkpoint root: $MODEL_NAME"
echo "Python: $(which python)"
python --version
nvidia-smi

CMD="python -u basemodel/evaluation_mmlu_ce_kl_hidden.py -c $MODEL_NAME -d /nfs/speed-scratch/z_yuefan/distill-project/original/data/mmlu/data"

echo "Running command:"
echo "$CMD"

eval "$CMD"

echo "Job finished"
