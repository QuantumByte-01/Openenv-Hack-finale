#!/usr/bin/env python3
# /// script
# dependencies = [
#   "nbformat>=5.10.4",
#   "nbclient>=0.10.0",
#   "ipykernel>=6.29.5",
#   "huggingface_hub>=0.22",
#   "matplotlib>=3.8",
#   "datasets>=2.18",
#   "transformers>=4.40",
#   "trl>=0.14.0",
#   "torch>=2.3",
#   "wandb>=0.16",
# ]
# ///

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import nbformat
from huggingface_hub import HfApi, snapshot_download
from ipykernel.kernelspec import install as install_kernel_spec
from nbclient import NotebookClient


REPO_ID = os.environ.get("REPO_ID", "Swastikr/polyglot_optima")
MODEL_NAME = os.environ.get("MODEL_NAME", "sshleifer/tiny-gpt2")
EPISODES_BASELINE = os.environ.get("EPISODES_BASELINE", "3")
EPISODES_EVAL = os.environ.get("EPISODES_EVAL", "3")
MAX_STEPS = os.environ.get("MAX_STEPS", "6")
TRAINING_MODE = os.environ.get("TRAINING_MODE", "sft_demo")
USE_WANDB = os.environ.get("USE_WANDB", "0")
USE_CPU = os.environ.get("USE_CPU", "0")


def _locate_workdir() -> Path:
    # Prefer local checkout path to avoid duplicate network fetches in jobs.
    script_repo_root = Path(__file__).resolve().parents[1]
    local_notebook = script_repo_root / "training" / "openenv_hackathon_training.ipynb"
    if local_notebook.exists() and os.environ.get("FORCE_SNAPSHOT_DOWNLOAD", "0") != "1":
        return script_repo_root

    token = os.environ.get("HF_TOKEN")
    workdir = Path("/tmp/polyglot-optima-notebook")
    print(f"[runner] Downloading space snapshot to {workdir} ...", flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="space",
        local_dir=str(workdir),
        token=token,
    )
    return workdir


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    workdir = _locate_workdir()

    notebook_path = workdir / "training" / "openenv_hackathon_training.ipynb"
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    # Force UTF-8 and partial-run settings inside job runtime.
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["MODEL_NAME"] = MODEL_NAME
    os.environ["EPISODES_BASELINE"] = EPISODES_BASELINE
    os.environ["EPISODES_EVAL"] = EPISODES_EVAL
    os.environ["MAX_STEPS"] = MAX_STEPS
    os.environ["TRAINING_MODE"] = TRAINING_MODE
    os.environ["USE_WANDB"] = USE_WANDB
    os.environ["USE_CPU"] = USE_CPU

    print("[runner] Runtime configuration:", flush=True)
    print(
        {
            "repo_id": REPO_ID,
            "model_name": MODEL_NAME,
            "episodes_baseline": EPISODES_BASELINE,
            "episodes_eval": EPISODES_EVAL,
            "max_steps": MAX_STEPS,
            "training_mode": TRAINING_MODE,
            "use_wandb": USE_WANDB,
            "use_cpu": USE_CPU,
            "workdir": str(workdir),
        },
        flush=True,
    )

    # Ensure a kernel spec exists in fresh HF Jobs runtime.
    print("[runner] Ensuring python3 kernel spec exists...", flush=True)
    install_kernel_spec(
        kernel_name="python3",
        display_name="Python 3",
        prefix=sys.prefix,
    )

    # Execute from notebook folder so relative imports/path logic behave as expected.
    os.chdir(workdir / "training")
    print(f"[runner] CWD set to {Path.cwd()}", flush=True)

    nb = nbformat.read(notebook_path, as_version=4)
    timeout_seconds = int(os.environ.get("NOTEBOOK_EXEC_TIMEOUT", "14400"))
    client = NotebookClient(nb, timeout=timeout_seconds, kernel_name="python3")

    done = threading.Event()

    def _heartbeat() -> None:
        start = time.time()
        while not done.wait(30):
            elapsed = int(time.time() - start)
            print(f"[runner] Notebook still running... elapsed={elapsed}s", flush=True)

    heartbeat = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat.start()
    print("[runner] Starting notebook execution...", flush=True)
    try:
        client.execute()
    finally:
        done.set()
    print("[runner] Notebook execution finished.", flush=True)

    out_name = f"openenv_hackathon_training.partial.{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.ipynb"
    out_path = workdir / "training" / out_name
    nbformat.write(nb, out_path)
    print(f"[runner] Executed notebook saved: {out_path}", flush=True)

    if not token:
        print("[runner] HF_TOKEN not found; skipping upload", flush=True)
        return

    api = HfApi(token=token)
    print("[runner] Uploading executed notebook artifact...", flush=True)
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=f"training_runs/notebooks/{out_name}",
        repo_id=REPO_ID,
        repo_type="space",
    )
    print(f"[runner] Uploaded executed notebook to spaces/{REPO_ID}/training_runs/notebooks/{out_name}", flush=True)


if __name__ == "__main__":
    main()
