#!/bin/bash
# HPC smoke job (P9a, P9a2): imports the repo (ahg) and torch from the locked environment and checks that
# torch sees the GPU through MPS. Runs nothing else and needs no data. Submit from the repo root on the
# login node, after `uv sync --locked`:
#   sbatch scripts/slurm/smoke.sbatch
# PYTHON overrides the interpreter; the default is the repo's locked environment.
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
cd "${SLURM_SUBMIT_DIR:?}"
# Half a card, as in train.sbatch: MPS caps the SM share and the pinned device memory (96 GB card).
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=48G"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
# uv is on the login node only (P9a2: exit 127 on gpunode8), so jobs run the env `uv sync --locked` built there.
${PYTHON:-.venv/bin/python} - <<'PY'
import subprocess
import sys

import torch

import ahg
from ahg.learning.models import GNN  # the repo's model code imports and runs on the GPU below

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda build", torch.version.cuda)
print("ahg from", ahg.__file__, "commit", commit)
ok = torch.cuda.is_available()
print("cuda available", ok, "devices", torch.cuda.device_count())
if not ok:
    sys.exit("torch does not see a GPU")
print("device 0", torch.cuda.get_device_name(0))
x = torch.ones(1024, 1024, device="cuda")
print("matmul sum", float((x @ x).sum()))  # 1024 * 1024 * 1024: a kernel ran on the GPU
gnn = GNN(8, 2, label_scale=1.0, cost_scale=1.0).cuda()
print("ahg GNN on", next(gnn.parameters()).device)
PY
