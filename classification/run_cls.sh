#!/bin/bash
#SBATCH --job-name=simpsons_cls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:1
#SBATCH --nodelist=turing
#SBATCH --partition=gpu_long
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#
# Train a classifier and write a Kaggle submission.
#   sbatch classification/run_cls.sh                                   # ConvNeXt-S, 10% val
#   FULL=1 NAME=convnext_s_full sbatch classification/run_cls.sh       # same on 100% data
#   MODEL=convnext_base.fb_in22k_ft_in1k_384 IMG=384 BS=32 NAME=convnext_b sbatch classification/run_cls.sh

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate simpsons
cd "$SLURM_SUBMIT_DIR"

export HF_HOME="$PWD/.hf_cache"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

DATA="${DATA:-data/journey/train}"
TEST="${TEST:-data/journey/testset}"
SAMPLE="${SAMPLE:-data/journey/sample_submission.csv}"
MODEL="${MODEL:-convnext_small.fb_in22k_ft_in1k_384}"
IMG="${IMG:-320}"
BS="${BS:-64}"
EPOCHS="${EPOCHS:-25}"
NAME="${NAME:-convnext_s}"

echo "job $SLURM_JOB_ID on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python classification/train_cls.py \
  --data "$DATA" --model "$MODEL" --img "$IMG" --bs "$BS" --epochs "$EPOCHS" \
  --name "$NAME" --workers 12 ${FULL:+--full}

python classification/predict_cls.py \
  --ckpt "runs_cls/$NAME/best.pt" --test-dir "$TEST" --sample-submission "$SAMPLE" \
  --out "runs_cls/$NAME/submission.csv" --workers 12

echo "done $(date). Submission: runs_cls/$NAME/submission.csv"
