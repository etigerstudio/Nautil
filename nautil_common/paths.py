"""Where the scripts read and write data, and how many API calls run in parallel.

Everything lives under one data root, set with the NAUTIL_DATA environment
variable (default: ./data next to this repository):

    <data root>/
        raw/          source documents
        cases/        evidence packages built from reports
        host_cases/   evidence packages built from host incidents
        dataset/      teacher trajectories, one folder per split and case
        runs/         intermediate and final outputs of every pipeline step
        experiments/  evaluation and RLVR replay outputs

API keys are read from NAUTIL_ENV_FILE (default: ./.env), never from code.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NAUTIL_DATA", REPO / "data")).resolve()

RAW = DATA / "raw"
CASES = DATA / "cases"
HOST_CASES = DATA / "host_cases"
DATASET = DATA / "dataset"
RUN = DATA / "runs"
EXPERIMENTS = DATA / "experiments"

CONFIGS = REPO / "investigation" / "configs"
ENV_FILE = Path(os.environ.get("NAUTIL_ENV_FILE", REPO / ".env")).resolve()

# Parallel requests to the hosted model API. Pick what your provider allows.
CONCURRENCY = int(os.environ.get("NAUTIL_CONCURRENCY", "4"))
