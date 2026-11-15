#!/bin/bash
# HPC smoke job (P9a): imports torch and checks that it sees the GPU through MPS.
# Runs nothing else and needs no data. Submit from any directory on the login node:
#   sbatch --export=ALL,PYTHON=/path/to/python scripts/slurm/smoke.sbatch
# PYTHON: an interpreter with torch installed; default `uv run --frozen python` in a repo checkout.
# Header limits: see train.sbatch (QOS gpu_csis_course: per user cpu=4, gres/mps=50, mem=80G).
#
#SBATCH --job-name=ahg-smoke
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:02:00
#SBATCH --output=ahg-smoke-%j.out
#SBATCH --error=ahg-smoke-%j.out

set -euo pipefail
# Half a card, as in train.sbatch: MPS caps the SM share and the pinned device memory (96 GB card).
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=48G"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
${PYTHON:-uv run --frozen python} - <<'PY'
import sys
import torch
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda build", torch.version.cuda)
ok = torch.cuda.is_available()
print("cuda available", ok, "devices", torch.cuda.device_count())
if not ok:
    sys.exit("torch does not see a GPU")
print("device 0", torch.cuda.get_device_name(0))
x = torch.ones(1024, 1024, device="cuda")
print("matmul sum", float((x @ x).sum()))  # 1024 * 1024 * 1024: a kernel ran on the GPU
PY
