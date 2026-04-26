#!/bin/bash
#SBATCH --partition=pt
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH -w speed-37
#SBATCH --job-name=evaluation_mmlu_teacher_submit
#SBATCH --output=/speed-scratch/%u/distill-project/original/evaluation_mmlu_teacher_submit_%j.out
#SBATCH --error=/speed-scratch/%u/distill-project/original/evaluation_mmlu_teacher_submit_%j.err

MODEL_NAME="$1"
OUT_JSON="$2"
LOAD4BIT="$3"
 
cd /speed-scratch/$USER/distill-project/original || exit 1
source /etc/profile
module load anaconda3/2023.03/default
source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate /speed-scratch/$USER/distill-env

CMD="python basemodel/evaluation_mmlu_teacher.py -c $MODEL_NAME -d /nfs/speed-scratch/z_yuefan/distill-project/original/data/mmlu/data"

echo "Running command:"
echo "$CMD"

eval "$CMD"