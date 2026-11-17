#!/bin/bash
# D28 exact labels (P7) as a SLURM job array: each task labels one strided shard of one cohort, both cost
# regimes, resumable (content-addressed records under data/labels/d28/; a rerun skips what exists).
# Labelling is single-threaded Python and never touches the GPU; the cluster has no CPU partition, so it runs
# on the GPU partition with the smallest MPS share. Limits as in labels.sbatch (QOS gpu_csis_course: per user
# cpu=4, gres/mps=50, mem=80G, no whole GPUs; partition MaxTime 2 days).
#
# Submit from the repo root on the login node, at a clean commit, after the cohort record exists:
#   sbatch --array=0-23 --export=ALL,COHORT=d28_train,N_SHARDS=24,STATE_BUDGET=<frozen> scripts/slurm/d28_labels.sbatch
# A test COHORT logs a "label generation" read in results/test_access_log.jsonl (Round 10).
#
#SBATCH --job-name=ahg-d28-labels
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/d28-labels-%A_%a.out
#SBATCH --error=logs/d28-labels-%A_%a.err

set -euo pipefail
: "${COHORT:?set COHORT via --export}" "${N_SHARDS:?set N_SHARDS via --export}" "${STATE_BUDGET:?set the frozen STATE_BUDGET via --export}"
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" && "$SLURM_ARRAY_TASK_COUNT" != "$N_SHARDS" ]]; then
    echo "array has $SLURM_ARRAY_TASK_COUNT tasks but N_SHARDS=$N_SHARDS" >&2
    exit 1
fi
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs
if [[ -n "$(git status --porcelain --untracked-files=no)$(git status --porcelain --untracked-files=all -- . ':(exclude)results')" ]]; then
    echo "working tree is dirty; label only at a clean commit" >&2
    exit 1
fi
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""  # CPU only
# uv is on the login node only (P9a2), so jobs run the env `uv sync --locked` built there.
.venv/bin/python scripts/d28_pipe.py label "$COHORT" --state-budget "$STATE_BUDGET" \
    --shard "${SLURM_ARRAY_TASK_ID:?}/$N_SHARDS"
