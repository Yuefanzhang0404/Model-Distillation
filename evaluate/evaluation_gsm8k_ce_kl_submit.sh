#!/bin/bash
#SBATCH --partition=pt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --job-name=evaluation_gsm8k_ce_kl
#SBATCH --output=/speed-scratch/%u/distill-project/original/evaluation_gsm8k_ce_kl_%j.out
#SBATCH --error=/speed-scratch/%u/distill-project/original/evaluation_gsm8k_ce_kl_%j.err

cd /speed-scratch/$USER/distill-project/original || exit 1
source /etc/profile
module load anaconda3/2023.03/default
source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate /speed-scratch/$USER/distill-env

python basemodel/evaluation_gsm8k_ce_kl.py
