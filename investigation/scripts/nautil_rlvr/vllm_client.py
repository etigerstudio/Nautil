from __future__ import annotations

import json
import subprocess
import time
import sys
from pathlib import Path

import requests

from . import SCRIPTS_DIR

PREFIX_SCRIPT = SCRIPTS_DIR / "make_vllm_lora_prefix_copy_v3.py"


def list_models(endpoint: str) -> list[dict]:
    reply = requests.get(f"{endpoint}/v1/models", timeout=30)
    reply.raise_for_status()
    return reply.json()["data"]


def load_lora(endpoint: str, name: str, path: Path) -> None:
    reply = requests.post(f"{endpoint}/v1/load_lora_adapter",
                          json={"lora_name": name, "lora_path": str(path)}, timeout=600)
    if reply.status_code != 200:
        raise RuntimeError(f"load_lora_adapter {name} failed: HTTP {reply.status_code} {reply.text[:300]}")


def unload_lora(endpoint: str, name: str) -> None:
    reply = requests.post(f"{endpoint}/v1/unload_lora_adapter", json={"lora_name": name}, timeout=120)
    if reply.status_code not in (200, 404):
        raise RuntimeError(f"unload_lora_adapter {name} failed: HTTP {reply.status_code} {reply.text[:300]}")


def prefix_copy(source: Path, output: Path, python: str | None = None) -> dict:
    proc = subprocess.run([python or sys.executable, str(PREFIX_SCRIPT), "--source", str(source),
                           "--output", str(output)], capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"prefix copy failed: {proc.stderr[-800:]}")
    manifest = json.loads((Path(output) / "prefix_copy_manifest.json").read_text())
    if manifest["bit_exact_mismatches"]:
        raise RuntimeError("prefix copy not bit exact")
    return manifest


def served_nll(endpoint: str, model: str, input_ids: list[int], loss_mask: list[int]) -> dict:
    body = {"model": model, "prompt": input_ids, "max_tokens": 1, "temperature": 0.0,
            "prompt_logprobs": 1, "return_token_ids": True}
    reply = requests.post(f"{endpoint}/v1/completions", json=body, timeout=900)
    reply.raise_for_status()
    entries = reply.json()["choices"][0]["prompt_logprobs"]
    total, count, values = 0.0, 0, []
    for pos in range(1, len(input_ids)):
        if not loss_mask[pos]:
            continue
        v = entries[pos][str(input_ids[pos])]["logprob"]
        total -= v
        values.append(v)
        count += 1
    return {"loss": total / max(count, 1), "tokens": count, "logprobs": values}


def verify(endpoint: str, model: str, probe: dict, trainer_probe: dict, tolerance: float,
           previous: dict | None = None) -> dict:
    served = served_nll(endpoint, model, probe["input_ids"], probe["loss_mask"])
    diffs = [abs(a - b) for a, b in zip(served["logprobs"], trainer_probe["logprobs"])]
    out = {"model": model, "vllm_loss": served["loss"], "trainer_loss": trainer_probe["loss"],
           "abs_loss_diff": abs(served["loss"] - trainer_probe["loss"]),
           "mean_abs_token_logprob_diff": sum(diffs) / max(len(diffs), 1),
           "tokens": served["tokens"], "tolerance": tolerance}
    if previous is not None:
        out["previous_trainer_loss"] = previous.get("trainer_loss")
        out["previous_vllm_loss"] = previous.get("vllm_loss")
    out["passed"] = served["tokens"] == trainer_probe["tokens"] and out["abs_loss_diff"] <= tolerance
    return out


def sleep(endpoint: str, level: int = 1) -> float:
    began = time.perf_counter()
    reply = requests.post(f"{endpoint}/sleep", params={"level": level}, timeout=600)
    if reply.status_code != 200:
        raise RuntimeError(f"sleep failed on {endpoint}: HTTP {reply.status_code} {reply.text[:300]}")
    if not is_sleeping(endpoint):
        raise RuntimeError(f"{endpoint} did not report sleeping")
    return time.perf_counter() - began


def wake(endpoint: str) -> float:
    began = time.perf_counter()
    reply = requests.post(f"{endpoint}/wake_up", timeout=600)
    if reply.status_code != 200:
        raise RuntimeError(f"wake_up failed on {endpoint}: HTTP {reply.status_code} {reply.text[:300]}")
    if is_sleeping(endpoint):
        raise RuntimeError(f"{endpoint} still sleeping after wake_up")
    return time.perf_counter() - began


def is_sleeping(endpoint: str) -> bool:
    reply = requests.get(f"{endpoint}/is_sleeping", timeout=60)
    reply.raise_for_status()
    return bool(reply.json().get("is_sleeping"))


def scrape_generation_tokens(endpoint: str) -> float | None:
    try:
        text = requests.get(f"{endpoint}/metrics", timeout=10).text
    except Exception:
        return None
    total = 0.0
    for line in text.splitlines():
        if line.startswith("vllm:generation_tokens_total"):
            total += float(line.rsplit(" ", 1)[1])
    return total
