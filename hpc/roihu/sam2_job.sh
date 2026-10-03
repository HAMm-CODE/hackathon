#!/bin/bash
# SAM 2 racket-axis run on one Roihu GH200 (gputest, 15 min max). Submit from the repo root:
#   sbatch --account=$PROJECT --export=ALL,FRAME_OFFSET=0 hpc/roihu/sam2_job.sh
# Needs hpc/roihu/setup_env.sh to have run once. Writes everything to out/sam2/:
#   slurm-<jobid>.out (this script's output), run_log.txt, sam_axes.json, sam_vs_clicks.csv, overlays/
# Slurm opens the log before the script runs, so out/sam2/ must already exist (mkdir -p out/sam2).
#SBATCH --job-name=sam2-axis
#SBATCH --partition=gputest
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=16
#SBATCH --output=out/sam2/slurm-%j.out
set -euo pipefail

PROJECT="${PROJECT:-${SLURM_JOB_ACCOUNT:-}}"
if [ -z "$PROJECT" ]; then
    echo "Set PROJECT or submit with --account=project_XXXXXXX" >&2; exit 1
fi
FRAME_OFFSET="${FRAME_OFFSET:-0}"
SAM2_DIR="/projappl/$PROJECT/sam2"
VENV="/projappl/$PROJECT/sam2-venv"

cd "${SLURM_SUBMIT_DIR:-.}"
if [ ! -f pipeline/sam2_axis.py ]; then
    echo "Submit from the repo root (no pipeline/sam2_axis.py in $(pwd))" >&2; exit 1
fi

set +u                              # module and activate scripts may read unset variables
module load python-pytorch          # includes CUDA/cuDNN; do not load cuda modules
source "$VENV/bin/activate"
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"   # newer Slurm: srun does not inherit it from sbatch

echo "job ${SLURM_JOB_ID:-local} on $(hostname) ($(uname -m)), project $PROJECT, frame offset $FRAME_OFFSET"
echo "repo $(pwd), sam2 $SAM2_DIR @ $(git -C "$SAM2_DIR" rev-parse --short HEAD)"
srun nvidia-smi
srun python3 -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

# --device cuda, not auto: fail fast instead of burning the 15 minutes on the Grace CPU
srun python3 pipeline/sam2_axis.py \
    --sam-dir "$SAM2_DIR" \
    --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_small.pt" \
    --model small \
    --frame-offset "$FRAME_OFFSET" \
    --device cuda

echo "done; copy out/sam2/ back and run: python pipeline/validate_sam.py"
