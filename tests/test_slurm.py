"""The sbatch headers match the BITS HPC limits read in P9a (15 Nov 2026): the only partition, the account's
only QOS (gpu_csis_course: per user cpu=4, gres/mps=50, mem=80G, no whole GPUs), at most half a card."""

import re
from pathlib import Path

import pytest

SLURM = Path(__file__).resolve().parents[1] / "scripts" / "slurm"
FILES = sorted(SLURM.glob("*.sbatch"))


def header(path: Path) -> dict[str, str]:
    return dict(re.findall(r"^#SBATCH --([\w-]+)=(\S+)", path.read_text(), re.M))


def test_every_job_script_is_checked():
    assert {f.name for f in FILES} == {"d28_labels.sbatch", "labels.sbatch", "smoke.sbatch", "train.sbatch"}


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_header_within_limits(path):
    h, text = header(path), path.read_text()
    assert "SET_ME" not in text
    assert h["partition"] == "gpu_rtx_pro_6000_6_csis_hyd" and h["qos"] == "gpu_csis_course"
    assert "gpu:" not in h.get("gres", "")  # gres/gpu=0 under the QOS: MPS shares only
    assert h["gres"].startswith("mps:") and 1 <= int(h["gres"].split(":")[1]) <= 50  # at most half a card
    assert int(h["cpus-per-task"]) <= 4 and int(h["mem"].rstrip("G")) <= 80
    hh, mm, ss = map(int, h["time"].split(":"))
    assert hh * 3600 + mm * 60 + ss <= 48 * 3600  # partition MaxTime 2 days


def test_smoke_job_imports_the_repo_from_its_locked_environment():
    text = (SLURM / "smoke.sbatch").read_text()
    assert "import ahg" in text and "${PYTHON:-.venv/bin/python}" in text and "nlp/.venv" not in text


def test_half_card_jobs_cap_mps_threads_and_memory():
    for name in ("train.sbatch", "smoke.sbatch"):
        text = (SLURM / name).read_text()
        assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50" in text and 'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=48G"' in text


def test_jobs_never_call_uv_on_the_compute_node():
    for path in FILES:
        assert "uv run" not in path.read_text(), path.name
