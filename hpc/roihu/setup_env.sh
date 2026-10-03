#!/bin/bash
# One-time SAM 2 setup on CSC Roihu. Everything lands in /projappl/$PROJECT so it persists:
#   /projappl/$PROJECT/sam2-venv                                  venv on top of the python-pytorch module
#   /projappl/$PROJECT/sam2                                       facebookresearch/sam2 clone (pip install -e)
#   /projappl/$PROJECT/sam2/checkpoints/sam2.1_hiera_small.pt     the only checkpoint we need
#
# Roihu GPU nodes are ARM (aarch64 Grace) and the login nodes are x86, so the venv must be
# built ON A GPU NODE; a venv built on the login node would not run in the job. Two ways, both
# from the repo root (/scratch/$PROJECT/$USER/hackathon):
#
#   a) as a short gputest batch job (log in hpc/roihu/setup-<jobid>.out):
#        sbatch --account=$PROJECT hpc/roihu/setup_env.sh
#
#   b) inside an interactive GPU session:
#        srun --account=$PROJECT --partition=gputest --gres=gpu:gh200:1 \
#             --cpus-per-task=16 --time=00:15:00 --pty bash
#        bash hpc/roihu/setup_env.sh
#      TODO(Roihu): confirm that interactive `srun --pty` is allowed on gputest, or whether CSC
#      provides a `sinteractive` variant for the GH200 nodes; option a) does not depend on this.
#
# TODO(Roihu): this needs outbound internet from the GPU node (git clone, PyPI, checkpoint
# download). If compute nodes are offline, the clone and the checkpoint are architecture
# independent and can be fetched on the login node first (rerun-safe: existing ones are kept);
# the pip install still has to run on the ARM node.
#
# Safe to rerun: an existing venv, clone or checkpoint is reused.
#SBATCH --job-name=sam2-setup
#SBATCH --partition=gputest
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=16
#SBATCH --output=hpc/roihu/setup-%j.out
set -euo pipefail

PROJECT="${PROJECT:-${SLURM_JOB_ACCOUNT:-}}"
if [ -z "$PROJECT" ]; then
    echo "Set PROJECT first, e.g. export PROJECT=project_XXXXXXX" >&2; exit 1
fi
if [ "$(uname -m)" != "aarch64" ]; then
    echo "This is $(uname -m), not aarch64: run inside a GH200 GPU session or job (see the header)." >&2; exit 1
fi

BASE="/projappl/$PROJECT"
VENV="$BASE/sam2-venv"
SAM2_DIR="$BASE/sam2"
CKPT="$SAM2_DIR/checkpoints/sam2.1_hiera_small.pt"
CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"

set +u                              # module and activate scripts may read unset variables
module load python-pytorch          # brings PyTorch with CUDA/cuDNN; do not load cuda modules
set -u
echo "host $(hostname) ($(uname -m)), python $(command -v python3)"
python3 -c "import torch; print('module torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

if [ ! -d "$VENV" ]; then
    python3 -m venv --system-site-packages "$VENV"
fi
set +u; source "$VENV/bin/activate"; set -u
export PIP_NO_CACHE_DIR=1           # keep pip's cache out of the small home quota

if [ ! -d "$SAM2_DIR/.git" ]; then
    git clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR"
fi
echo "sam2 commit $(git -C "$SAM2_DIR" rev-parse HEAD)"

# Pin torch/torchvision to the module's own builds, so pip errors out instead of silently
# replacing the module's CUDA torch with a PyPI wheel if sam2 ever asks for a newer version.
python3 - > "$VENV/torch-constraints.txt" <<'EOF'
import importlib
for name in ("torch", "torchvision"):
    try:
        print(f"{name}=={importlib.import_module(name).__version__}")
    except ImportError:
        pass
EOF
echo "constraints: $(tr '\n' ' ' < "$VENV/torch-constraints.txt")"

# SAM2_BUILD_CUDA=0 skips the optional CUDA extension (only used for mask post-processing),
# so nothing is compiled on ARM. --no-build-isolation makes pip build with the module's torch
# instead of downloading another torch into a temporary build env (sam2's pyproject.toml
# lists torch as a build requirement).
pip install -c "$VENV/torch-constraints.txt" --upgrade "setuptools>=64" wheel   # editable installs need setuptools >= 64
(cd "$SAM2_DIR" && SAM2_BUILD_CUDA=0 pip install --no-build-isolation -c "$VENV/torch-constraints.txt" -e .)
pip install -c "$VENV/torch-constraints.txt" opencv-python-headless

if [ ! -s "$CKPT" ]; then
    mkdir -p "$(dirname "$CKPT")"
    curl -fL --retry 3 -o "$CKPT.part" "$CKPT_URL"
    mv "$CKPT.part" "$CKPT"
fi
ls -lh "$CKPT"

# Check from a neutral directory: importing sam2 from the clone's parent picks up the wrong package.
(cd / && python3 - <<'EOF'
import torch, cv2, sam2
from sam2.build_sam import build_sam2_video_predictor
print("torch", torch.__version__, "from", torch.__file__)
print("cuda available:", torch.cuda.is_available(), "| opencv", cv2.__version__, "| sam2 from", sam2.__file__)
EOF
)
echo "Setup done. Venv: $VENV  SAM 2: $SAM2_DIR"
