from __future__ import annotations

import gzip
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repository root
from nautil_common import paths

SCRIPTS = Path(__file__).resolve().parents[2]
RUN = paths.RUN
OUT = RUN / "results/rlvr_v1/mock_e2e"
PY = sys.executable
sys.path.insert(0, str(SCRIPTS))

from nautil_rlvr.common import write_jsonl
from nautil_rlvr.testing import synthetic


def build_assets(root: Path) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
                            linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                            max_position_embeddings=32768, eos_token_id=2, pad_token_id=3, bos_token_id=1)
    model = Qwen3_5ForCausalLM(cfg)
    model.save_pretrained(root / "model")
    peft = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                                            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    with torch.no_grad():
        for n, p in peft.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    peft.save_pretrained(root / "e2_init")
    cases, refs, teachers = synthetic.make_dataset(6)
    write_jsonl(root / "train_bundle.jsonl", cases)
    write_jsonl(root / "train_refs.jsonl", refs)
    write_jsonl(root / "train_teacher.jsonl", teachers)
    vcases, vrefs, _ = [], [], []
    for n in (101, 102, 103, 104):
        c, r, _t = synthetic.make_case(n, n % 2 == 1, "host" if n % 2 else "ntsb")
        c["case_id"] = c["case_id"].replace("MOCK", "MVAL")
        c["user_message"] = c["user_message"].replace("MOCK", "MVAL")
        r["case_id"] = c["case_id"]
        vcases.append(c)
        vrefs.append(r)
    write_jsonl(root / "val_bundle.jsonl", vcases)
    write_jsonl(root / "val_judge_refs.jsonl", vrefs)
    write_jsonl(root / "val_refs_expected.jsonl",
                [{"case_id": r["case_id"], "expected_closure": ("determined: bearing" if r["closure"] == "closed"
                                                                 else "undetermined: open")} for r in vrefs])
    (root / "system.txt").write_text(synthetic.SYSTEM + "\n")
    return {"cases": [c["case_id"] for c in cases]}


def config(root: Path, name: str, **over) -> Path:
    cfg = {
        "run_name": name, "total_steps": 3,
        "paths": {"base_model": str(root / "model"), "tokenizer": "mock:bytes", "init_adapter": str(root / "e2_init"),
                  "system_prompt": str(root / "system.txt"), "train_bundle": str(root / "train_bundle.jsonl"),
                  "train_refs": str(root / "train_refs.jsonl"), "train_teacher": str(root / "train_teacher.jsonl"),
                  "val_bundle": str(root / "val_bundle.jsonl"), "val_refs": str(root / "val_refs_expected.jsonl"),
                  "val_judge_refs": str(root / "val_judge_refs.jsonl"),
                  "rollout_protocol": str(SCRIPTS / "nautil_rlvr/configs/rl_rollout_protocol_v1.json"),
                  "eval_protocol": str(paths.CONFIGS / "scheduler_protocol_v1.json"),
                  "output_root": str(root / "runs")},
        "pipeline": {"mode": "sync", "colocate": True, "dist_backend": "gloo", "sleep_level": 1},
        "policy": {"device": "cpu", "dtype": "float32", "packing": "auto", "logit_chunk": 512,
                   "lora_rank": 16, "lora_alpha": 32, "gradient_checkpointing": True},
        "rollout": {"endpoints": ["http://127.0.0.1:18300", "http://127.0.0.1:18301"], "concurrency": 16,
                    "lora_prefix": "rlvr", "base_model_name": "mock-base", "max_attempts": 2},
        "grpo": {"group_size": 4, "cases_per_step": 4, "min_groups": 2, "max_refill_rounds": 1,
                 "straggler_done_fraction": 0.7, "minibatches_per_step": 2, "lr": 5e-3, "clip_low": 0.2,
                 "clip_high": 0.28, "kl_coef": 0.0, "adv_std_normalize": True, "drop_identical_groups": True,
                 "max_staleness": 1},
        "sampler": {"alpha": 0.0, "seed": 1},
        "reward": {"phase": 2, "judge_failure_policy": "exclude_dimension"},
        "judges": {"train": {"model": "mock-judge", "routes": [
                        {"name": "k1", "base_url": "http://127.0.0.1:18310/v1", "key_env": "MOCK_KEY_1", "wire_model": "m", "max_concurrent": 4},
                        {"name": "k2", "base_url": "http://127.0.0.1:18310/v1", "key_env": "MOCK_KEY_2", "wire_model": "m", "max_concurrent": 4}],
                        "retry_backoff": [0.1, 0.2], "timeout_seconds": 30, "group_deadline_seconds": 120},
                   "eval": {"model": "mock-eval-judge", "routes": [
                        {"name": "e1", "base_url": "http://127.0.0.1:18310/v1", "key_env": "MOCK_KEY_1", "wire_model": "e", "max_concurrent": 2}],
                        "response_format_json": False, "temperature": None, "retry_backoff": [0.1]}},
        "lora_check": {"tolerance": 1e-4},
        "checkpoint": {"every_steps": 1, "keep_recent_adapters": 2},
        "eval": {"every_steps": 2, "samples": 1, "concurrency": 4, "tokenizer": "mock:bytes3", "selftest_mock": True,
                 "scheduler_scripts_dir": str(SCRIPTS), "python": PY, "llm_closure": True},
        "monitor": {"kl_sequences": 2, "rank_sync_check_every": 1, "max_judge_failure_rate": 0.05,
                    "max_judge_parse_failure_rate": 0.02, "first_ratio_tolerance": 0.05},
        "logging": {"tensorboard_dir": str(root / "tb" / name), "dump_every": 1, "dump_count": 2},
    }
    for k, v in over.items():
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = v
    path = root / f"config_{name}.json"
    path.write_text(json.dumps(cfg, indent=1))
    return path


def start(cmd, log):
    return subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, cwd=str(SCRIPTS),
                            start_new_session=True)


def wait_up(url):
    for _ in range(200):
        try:
            requests.get(url, timeout=1)
            return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"{url} did not come up")


def train(cfg_path: Path, log: Path, world: int, *extra) -> int:
    env = {**os.environ, "MOCK_KEY_1": "mock-key-one", "MOCK_KEY_2": "mock-key-two", "OMP_NUM_THREADS": "2"}
    if world > 1:
        cmd = [PY, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}", "-m",
               "nautil_rlvr.train", "--config", str(cfg_path), *extra]
    else:
        cmd = [PY, "-m", "nautil_rlvr.train", "--config", str(cfg_path), *extra]
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(SCRIPTS), env=env, timeout=3600).returncode


def events(log: Path) -> list[dict]:
    out = []
    for line in log.read_text().splitlines():
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def metrics(run_dir: Path) -> list[dict]:
    p = run_dir / "metrics.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    build_assets(OUT)
    procs = [start([PY, "-m", "nautil_rlvr.testing.mock_vllm", "--port", str(p), "--model-dir", str(OUT / "model")],
                   OUT / f"mock_vllm_{p}.log") for p in (18300, 18301)]
    procs.append(start([PY, "-m", "nautil_rlvr.testing.mock_judge", "--port", "18310", "--fail-rate", "0.03"],
                       OUT / "mock_judge.log"))
    report = {}
    try:
        for p in (18300, 18301):
            wait_up(f"http://127.0.0.1:{p}/v1/models")
        wait_up("http://127.0.0.1:18310/")
        cfg = config(OUT, "A_sync")
        t0 = time.time()
        code1 = train(cfg, OUT / "A_part1.log", 2, "--max-steps", "2")
        code2 = train(cfg, OUT / "A_part2.log", 2, "--resume", "--max-steps", "3")
        run_dir = OUT / "runs" / "A_sync"
        m = metrics(run_dir)
        ev1, ev2 = events(OUT / "A_part1.log"), events(OUT / "A_part2.log")
        start2 = next(e for e in ev2 if e.get("event") == "start")
        checks = json.loads((run_dir / "lora_checks" / "v00003_a1.json").read_text())
        evals = sorted((run_dir / "eval").glob("step_*/summary.json"))
        llm = sorted((run_dir / "eval").glob("step_*/llm_closure.json"))
        recs = [json.loads(x) for x in gzip.open(run_dir / "rollouts" / "batch_00000.jsonl.gz", "rt")]
        report["A_sync_colocate_2ranks"] = {
            "exit_codes": [code1, code2], "seconds": round(time.time() - t0, 1),
            "steps_logged": [r["step"] for r in m],
            "resume_started_at_step": start2["start_step"], "attempts": len((run_dir / "attempts.jsonl").read_text().splitlines()),
            "checkpoints": sorted(p.name for p in (run_dir / "checkpoints").glob("step_*")),
            "adapter_fingerprints": [r["lora"]["adapter_fingerprint"] for r in m],
            "lora_check_abs_diff_by_step": [r["lora"]["vllm_trainer_abs_loss_diff"] for r in m],
            "vllm_base_vs_lora_probe_loss": [next(e for e in ev1 if e.get("event") == "start")["vllm_base_probe_loss"],
                                             next(e for e in ev1 if e.get("event") == "start")["initial_lora_check"]["vllm_loss"]],
            "last_check": {k: checks[0][k] for k in ("vllm_loss", "trainer_loss", "abs_loss_diff", "passed",
                                                     "previous_trainer_loss")},
            "first_minibatch_ratio": [r["train"].get("first_minibatch_ratio_mean") for r in m],
            "ratio_max": [r["train"].get("ratio_max") for r in m],
            "reward_mean": [r["reward"]["total_mean"] for r in m],
            "kept_groups": [r["groups"]["kept"] for r in m], "dropped": [
                {k: v for k, v in r["groups"].items() if k.startswith("dropped_")} for r in m],
            "straggler_cutoffs": [r["groups"]["straggler_cutoffs"] for r in m],
            "aborted_trajectories": [r["groups"]["trajectories_aborted"] for r in m],
            "turn_token_mismatch_rate": [r["behaviour"]["turn_token_mismatch_rate"] for r in m],
            "judge_calls": [r["judge"].get("calls") for r in m], "judge_retries": [r["judge"].get("retries") for r in m],
            "judge_errors": [{k: v for k, v in r["judge"].items() if k.startswith("error_")} for r in m],
            "judge_per_key": [{k: v for k, v in r["judge"].items() if k.startswith("usage_")} for r in m],
            "time_split_step1": m[0]["time"] if m else None,
            "packing": json.loads((run_dir / "packing_check_attempt0.json").read_text()),
            "eval_summaries": [json.loads(p.read_text()) for p in evals],
            "eval_llm_closure": [{k: v for k, v in json.loads(p.read_text()).items() if not isinstance(v, (list, dict))} for p in llm],
            "rollout_records_batch0": len(recs), "record_has_full_turns": "turns" in recs[0]["trajectory"],
            "dumps": sorted(str(p.relative_to(run_dir)) for p in (run_dir / "dumps").rglob("*.md"))[:4],
            "tensorboard_files": len(list((OUT / "tb" / "A_sync").glob("events*"))),
            "rank_sync_checks_passed": code1 == 0 and code2 == 0,
        }
        cfg = config(OUT, "B_async", **{"pipeline.mode": "async", "pipeline.colocate": False,
                                        "rollout.endpoints": ["http://127.0.0.1:18300"], "eval.every_steps": 0})
        code = train(cfg, OUT / "B.log", 1, "--max-steps", "3")
        m = metrics(OUT / "runs" / "B_async")
        report["B_async_1rank"] = {"exit_code": code, "steps": [r["step"] for r in m],
                                   "staleness": [r["train"]["staleness"] for r in m],
                                   "first_minibatch_ratio": [r["train"].get("first_minibatch_ratio_mean") for r in m],
                                   "lora_diff": [r["lora"]["vllm_trainer_abs_loss_diff"] for r in m]}
        procs.append(start([PY, "-m", "nautil_rlvr.testing.mock_judge", "--port", "18311", "--garbage-rate", "0.5"],
                           OUT / "mock_judge_garbage.log"))
        wait_up("http://127.0.0.1:18311/")
        cfg = config(OUT, "C_pause", **{"pipeline.colocate": False, "rollout.endpoints": ["http://127.0.0.1:18300"],
                                        "eval.every_steps": 0})
        raw = json.loads(cfg.read_text())
        for r in raw["judges"]["train"]["routes"]:
            r["base_url"] = "http://127.0.0.1:18311/v1"
        cfg.write_text(json.dumps(raw))
        code = train(cfg, OUT / "C.log", 1, "--max-steps", "2")
        alert = OUT / "runs" / "C_pause" / "PAUSED_ALERT.json"
        report["C_judge_garbage_pause"] = {"exit_code": code, "alert_written": alert.exists(),
                                           "reasons": json.loads(alert.read_text())["reasons"] if alert.exists() else None,
                                           "checkpoint_for_resume": sorted(p.name for p in (OUT / "runs" / "C_pause" / "checkpoints").glob("step_*"))}
        from nautil_rlvr import vllm_client
        ep = "http://127.0.0.1:18300"
        vllm_client.load_lora(ep, "not_renamed", OUT / "e2_init")
        probe = {"input_ids": [1, 20, 30, 40, 50, 60, 70, 2], "loss_mask": [0, 0, 0, 1, 1, 1, 1, 1]}
        served_bad = vllm_client.served_nll(ep, "not_renamed", probe["input_ids"], probe["loss_mask"])["loss"]
        served_base = vllm_client.served_nll(ep, "mock-base", probe["input_ids"], probe["loss_mask"])["loss"]
        vllm_client.prefix_copy(OUT / "e2_init", OUT / "e2_init_lmprefix")
        vllm_client.load_lora(ep, "renamed", OUT / "e2_init_lmprefix")
        served_ok = vllm_client.served_nll(ep, "renamed", probe["input_ids"], probe["loss_mask"])["loss"]
        from nautil_rlvr.policy import Policy
        pol = Policy(OUT / "model", OUT / "e2_init", device="cpu", dtype="float32")
        trainer = pol.probe(probe["input_ids"], probe["loss_mask"])["loss"]
        report["D_lora_pitfall"] = {"not_renamed_equals_base": abs(served_bad - served_base) < 1e-6,
                                    "renamed_matches_trainer": abs(served_ok - trainer) < 1e-5,
                                    "not_renamed_vs_trainer_diff": abs(served_bad - trainer),
                                    "skipped_keys": requests.get(f"{ep}/mock/skipped").json()}
    finally:
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
