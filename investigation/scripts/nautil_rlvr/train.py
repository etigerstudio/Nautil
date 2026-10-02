from __future__ import annotations

import argparse
import collections
import datetime
import gzip
import json
import math
import os
import queue
import random
import shutil
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import behaviour, evaluation, evaluation_v22, reward_v2, reward_v22, vllm_client
from .common import atomic_json, read_jsonl, refuse_test_path, sha256_file
from .config import load as load_config
from .data import load_extra_sources, make_sampler
from .grpo import center_by_decision, group_advantages, group_is_degenerate, group_std
from .reward import RewardConfig, compute, judge_requests, model_closed
from .rollout import render_prompt_ids, run_trajectory
from .sequences import build_sequence
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root
from nautil_common import paths

PAUSE_EXIT = 4


def load_tokenizer(spec: str):
    if spec == "mock:bytes":
        from .testing.byte_tokenizer import ByteTokenizer
        return ByteTokenizer()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(spec, local_files_only=True)


def log(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False, default=str), flush=True)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


class TB:
    def __init__(self, logdir):
        self.w = None
        if logdir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.w = SummaryWriter(str(logdir))
            except Exception as exc:
                log({"event": "tensorboard_unavailable", "error": str(exc)})

    def scalars(self, prefix: str, values: dict, step: int) -> None:
        if not self.w:
            return
        for k, v in values.items():
            if isinstance(v, bool):
                v = float(v)
            if isinstance(v, (int, float)) and math.isfinite(v):
                self.w.add_scalar(f"{prefix}/{k}", v, step)
            elif isinstance(v, dict):
                self.scalars(f"{prefix}/{k}", v, step)

    def hist(self, tag: str, values: list, step: int) -> None:
        if self.w and values:
            import numpy as np
            self.w.add_histogram(tag, np.asarray(values, dtype=float), step, bins=np.arange(-0.5, 11.0, 1.0))

    def hist_auto(self, tag: str, values: list, step: int) -> None:
        if self.w and values:
            import numpy as np
            self.w.add_histogram(tag, np.asarray(values, dtype=float), step, bins=40)

    def text(self, tag: str, text: str, step: int) -> None:
        if self.w:
            self.w.add_text(tag, text, step)

    def flush(self):
        if self.w:
            self.w.flush()


class Dist:
    def __init__(self, backend: str):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.ctrl = None
        if self.world > 1:
            import torch
            import torch.distributed as dist
            timeout = datetime.timedelta(hours=12)
            if backend == "nccl":
                torch.cuda.set_device(self.local_rank)
                dist.init_process_group("nccl", timeout=timeout, device_id=torch.device("cuda", self.local_rank))
            else:
                dist.init_process_group("gloo", timeout=timeout)
            self.ctrl = dist.new_group(backend="gloo", timeout=timeout)
            self.dist = dist

    def bcast(self, obj=None):
        if self.world == 1:
            return obj
        box = [obj]
        self.dist.broadcast_object_list(box, src=0, group=self.ctrl)
        return box[0]

    def gather(self, obj):
        if self.world == 1:
            return [obj]
        out = [None] * self.world
        self.dist.all_gather_object(out, obj, group=self.ctrl)
        return out


def make_policy(cfg: dict, d: Dist, init_adapter: Path):
    from .policy import Policy
    pc, g, paths = cfg["policy"], cfg["grpo"], cfg["paths"]
    device = pc.get("device", "cuda")
    if device == "cuda":
        device = f"cuda:{d.local_rank if d.world > 1 else 0}"
    need_ref = float(g.get("kl_coef", 0)) > 0 or int(cfg.get("monitor", {}).get("kl_sequences", 0)) > 0
    return Policy(Path(paths["base_model"]), init_adapter, device=device, dtype=pc.get("dtype", "bfloat16"),
                  lr=float(g["lr"]), weight_decay=float(g.get("weight_decay", 0.0)),
                  betas=g.get("betas", [0.9, 0.999]), max_grad_norm=float(g.get("max_grad_norm", 1.0)),
                  ref_adapter=Path(paths["init_adapter"]) if need_ref else None,
                  chunk_size=int(pc.get("logit_chunk", 2048)), expect_rank=pc.get("lora_rank", 16),
                  expect_alpha=pc.get("lora_alpha", 32),
                  gradient_checkpointing=pc.get("gradient_checkpointing", True),
                  attn_implementation=pc.get("attn_implementation", "sdpa"),
                  pack_tokens=int(pc.get("pack_tokens", 32768)))


def worker_loop(cfg: dict, d: Dist) -> None:
    import torch
    init = d.bcast(None)
    policy = make_policy(cfg, d, Path(init["adapter"]))
    if init.get("state"):
        state = torch.load(init["state"], map_location="cpu", weights_only=False)
        policy.load_trainer_state(state["trainer"])
    while True:
        cmd = d.bcast(None)
        op = cmd["op"]
        if op == "stop":
            return
        if op == "packing":
            policy.packing = bool(cmd["value"])
        elif op == "update":
            shard = torch.load(cmd["paths"][d.rank], map_location="cpu", weights_only=False)
            policy.update(shard, cmd["clip_low"], cmd["clip_high"], cmd["kl_coef"], cmd["lr"])
        elif op == "fingerprint":
            d.gather(policy.adapter_fingerprint())


class Run:
    def __init__(self, cfg: dict, d: Dist, resume: bool, max_steps: int | None = None):
        self.cfg, self.d = cfg, d
        self.paths, self.g, self.pipe = cfg["paths"], cfg["grpo"], cfg.get("pipeline", {})
        self.mode = self.pipe.get("mode", "sync")
        self.colocate = bool(self.pipe.get("colocate", False))
        self.rcfg = RewardConfig.from_dict(cfg["reward"])
        self.rmod = {"2": reward_v2, "2.2": reward_v22}.get(str(cfg["reward"].get("version", 1)))
        self.v2 = self.rmod.RewardV2Config.from_dict(cfg["reward"]) if self.rmod else None
        self.protocol = json.loads(Path(self.paths["rollout_protocol"]).read_text())
        self.run_dir = Path(self.paths["output_root"]) / cfg["run_name"]
        self.endpoints = [e.rstrip("/") for e in cfg["rollout"]["endpoints"]]
        self.total_steps = int(max_steps if max_steps is not None else cfg["total_steps"])
        self.max_staleness = int(self.g.get("max_staleness", 1)) if self.mode == "async" else 0
        self.stop = threading.Event()
        self.producer_error = None
        self.cv = threading.Condition()
        self.published = None
        self.version_names: dict[int, str] = {}
        self.version_dirs: dict[int, Path] = {}
        self.in_use = collections.Counter()
        self.pinned: set[str] = set()
        self.loaded: list[tuple[int, str]] = []
        self.batches: queue.Queue = queue.Queue(maxsize=1)
        self.resume = resume
        self.tb = TB(cfg["logging"].get("tensorboard_dir"))
        self.eval_threads: list = []
        self.score_threads: list[threading.Thread] = []
        self.alerts: list[dict] = []
        self.dispatch = 0
        self.last_check = None
        self.stop_files = [self.run_dir / "STOP"] + [Path(x) for x in cfg.get("control", {}).get("stop_files", [])]
        self.stop_reason = None

    def setup(self) -> None:
        cfg, paths = self.cfg, self.paths
        if self.run_dir.exists() and any(self.run_dir.iterdir()) and not self.resume:
            raise SystemExit(f"{self.run_dir} exists; use --resume or a new run_name")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for f in self.stop_files:
            if f.exists():
                f.rename(f.with_name(f"{f.name}.stale_at_start_{int(time.time())}"))
                log({"event": "stale_stop_file_moved", "path": str(f)})
        attempts = self.run_dir / "attempts.jsonl"
        self.attempt = sum(1 for _ in attempts.open()) if attempts.exists() else 0
        with attempts.open("a") as f:
            f.write(json.dumps({"attempt": self.attempt, "started": time.time(), "argv": sys.argv,
                                "config_sha256": cfg["_config_sha256"], "resume": self.resume}) + "\n")
        atomic_json(self.run_dir / f"config_attempt{self.attempt}.json", cfg)
        self.tokenizer = load_tokenizer(paths["tokenizer"])
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.system = Path(paths["system_prompt"]).read_text().strip()
        bundle = read_jsonl(refuse_test_path(paths["train_bundle"]))
        refs = read_jsonl(refuse_test_path(paths["train_refs"]))
        self.cases = {c["case_id"]: c for c in bundle}
        self.refs = {r["case_id"]: r for r in refs}
        if set(self.cases) != set(self.refs):
            raise ValueError("train bundle and refs disagree")
        load_extra_sources(cfg.get("sampler", {}).get("extra_sources", {}))
        room = self.protocol["context_limit"] - self.protocol["max_tokens_per_turn"]
        too_long = {}
        for cid, case in self.cases.items():
            n = len(render_prompt_ids(self.tokenizer, [{"role": "system", "content": self.system},
                                                       {"role": "user", "content": case["user_message"]}],
                                      case["tools"]))
            if n >= room:
                too_long[cid] = n
        s = cfg.get("sampler", {})
        self.sampler = make_sampler(list(self.refs.values()), s, set(too_long))
        atomic_json(self.run_dir / "data_report.json",
                    {"cases": len(self.cases), "excluded_opening_too_long": too_long,
                     "sampler": s, "strata": self.sampler.describe(),
                     "subset_case_ids": self.sampler.case_ids() if hasattr(self.sampler, "case_ids") else None,
                     "train_bundle_sha256": sha256_file(paths["train_bundle"]),
                     "train_refs_sha256": sha256_file(paths["train_refs"])})
        self.probe = self._build_probe()
        start_step, state_path, init_adapter = 0, None, Path(paths["init_adapter"])
        state = None
        if self.resume:
            ck = self.latest_checkpoint()
            if ck is not None:
                import torch
                state_path = ck / "trainer_state.pt"
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                start_step = int(state["step"])
                init_adapter = ck / "adapter"
                self.sampler.restore(state["sampler_state"])
        self.start_step = start_step
        self.d.bcast({"adapter": str(init_adapter), "state": str(state_path) if state_path else None})
        self.policy = make_policy(cfg, self.d, init_adapter)
        if state is not None:
            self.policy.load_trainer_state(state["trainer"])
        self.judge = self.eval_judge = None
        self.eval_judges = {}
        if self.rcfg.phase >= 2:
            from .judge import JudgeClient, JudgeConfig
            self.judge = JudgeClient(JudgeConfig.from_dict(cfg["judges"]["train"]))
        if cfg["eval"].get("design") == "v22":
            if cfg["eval"].get("llm_closure"):
                from .judge import JudgeClient, JudgeConfig
                for label, key in (("sol", "eval"), ("luna", "eval_luna")):
                    if cfg.get("judges", {}).get(key):
                        self.eval_judges[label] = JudgeClient(JudgeConfig.from_dict(cfg["judges"][key]))
        elif cfg["eval"].get("llm_closure") and cfg.get("judges", {}).get("eval"):
            from .judge import JudgeClient, JudgeConfig
            self.eval_judge = JudgeClient(JudgeConfig.from_dict(cfg["judges"]["eval"]))
        self.judge_cost_prior = float(state.get("judge_cost_usd", 0.0)) if state else 0.0
        self.rollout_pool = ThreadPoolExecutor(max_workers=int(cfg["rollout"]["concurrency"]))
        pmode = cfg["policy"].get("packing", "auto")
        pack = {"enabled": False, "mode": pmode}
        if pmode in ("auto", True, "on"):
            half = len(self.probe["input_ids"]) // 2
            seqs = [{"input_ids": self.probe["input_ids"], "loss_mask": self.probe["loss_mask"]},
                    {"input_ids": self.probe["input_ids"][:half], "loss_mask": self.probe["loss_mask"][:half]}]
            check = self.policy.packing_equivalence(seqs)
            pack.update(check)
            pack["enabled"] = bool(check["passed"]) or pmode == "on"
        self.policy.packing = pack["enabled"]
        self.d.bcast({"op": "packing", "value": pack["enabled"]})
        atomic_json(self.run_dir / f"packing_check_attempt{self.attempt}.json", pack)
        base_name = cfg["rollout"].get("base_model_name")
        base_check = None
        if base_name:
            base_check = vllm_client.served_nll(self.endpoints[0], base_name, self.probe["input_ids"],
                                                self.probe["loss_mask"])["loss"]
        check = self.publish(start_step, init_adapter)
        log({"event": "start", "run_dir": str(self.run_dir), "start_step": start_step,
             "total_steps": self.total_steps, "attempt": self.attempt, "mode": self.mode,
             "colocate": self.colocate, "world": self.d.world, "phase": self.rcfg.phase,
             "reward_version": str(self.cfg["reward"].get("version", 1)),
             "reward_variant": self.v2.variant if self.v2 else None,
             "judge_prompts": ({d: [self.judge.prompt_paths[d], self.judge.prompt_sha[d][:12]]
                                for d in self.judge.prompts} if self.judge else None),
             "stop_files": [str(f) for f in self.stop_files],
             "probe_case": self.probe["case_id"], "vllm_base_probe_loss": base_check,
             "initial_lora_check": check, "packing": pack, "excluded_too_long": len(too_long)})
        self.tb.scalars("lora", {"vllm_base_probe_loss": base_check}, start_step)

    def _build_probe(self) -> dict:
        from verify_qwen35_wire_cpu import analyze
        pc = self.cfg.get("lora_check", {})
        rows = {r["case_id"]: r for r in read_jsonl(refuse_test_path(self.paths["train_teacher"]))}
        cid = pc.get("probe_case") or min(rows, key=lambda c: len(json.dumps(rows[c])))
        _, record = analyze(rows[cid], self.tokenizer, Path(self.paths["system_prompt"]).read_text(),
                            return_tokens=True)
        mask = [int(x != -100) for x in record["labels"]]
        atomic_json(self.run_dir / "probe.json", {"case_id": cid, "tokens": len(mask), "targets": sum(mask)})
        return {"case_id": cid, "input_ids": record["input_ids"], "loss_mask": mask}

    def latest_checkpoint(self) -> Path | None:
        root = self.run_dir / "checkpoints"
        done = sorted(p for p in root.glob("step_*") if (p / "COMPLETE").exists()) if root.exists() else []
        return done[-1] if done else None

    def publish(self, version: int, adapter_dir: Path, trainer_probe: dict | None = None) -> dict:
        began = time.perf_counter()
        name = f"{self.cfg['rollout']['lora_prefix']}_{self.cfg['run_name']}_v{version:05d}_a{self.attempt}"
        prefixed = self.run_dir / "adapters" / f"{name}_lmprefix"
        manifest = vllm_client.prefix_copy(adapter_dir, prefixed, self.cfg["rollout"].get("prefix_copy_python"))
        for ep in self.endpoints:
            vllm_client.load_lora(ep, name, prefixed)
        if trainer_probe is None:
            trainer_probe = self.policy.probe(self.probe["input_ids"], self.probe["loss_mask"])
        checks = [vllm_client.verify(ep, name, self.probe, trainer_probe,
                                     float(self.cfg["lora_check"].get("tolerance", 0.01)), self.last_check)
                  for ep in self.endpoints]
        for c in checks:
            c["tensors"] = manifest["tensors"]
        atomic_json(self.run_dir / "lora_checks" / f"v{version:05d}_a{self.attempt}.json", checks)
        if not all(c["passed"] for c in checks):
            raise RuntimeError(f"vLLM does not serve adapter v{version} as trained: {checks}")
        self.last_check = checks[0]
        with self.cv:
            self.version_names[version] = name
            self.version_dirs[version] = prefixed
            self.loaded.append((version, name))
            self.published = version
            self.cv.notify_all()
        self.unload_old(version)
        return {"version": version, "name": name, "seconds": time.perf_counter() - began,
                "vllm_loss": checks[0]["vllm_loss"], "trainer_loss": checks[0]["trainer_loss"],
                "abs_loss_diff": max(c["abs_loss_diff"] for c in checks),
                "mean_abs_token_logprob_diff": checks[0]["mean_abs_token_logprob_diff"]}

    def unload_old(self, newest: int) -> None:
        keep_from = newest - self.max_staleness
        with self.cv:
            victims = [(v, n) for v, n in self.loaded if v < keep_from and self.in_use[v] == 0]
            self.loaded = [x for x in self.loaded if x not in victims]
        for _, n in victims:
            for ep in self.endpoints:
                try:
                    vllm_client.unload_lora(ep, n)
                except Exception as exc:
                    log({"event": "unload_failed", "name": n, "error": str(exc)})

    def one_trajectory(self, b, rnd, cid, k, name, ep, abort) -> dict:
        case = self.cases[cid]
        seed_base = int(self.protocol["seed"]) + 1_000_003 * b + 10_007 * rnd + k
        traj = None
        for attempt in range(int(self.cfg["rollout"].get("max_attempts", 3))):
            traj = run_trajectory(ep, name, self.tokenizer, self.end_id, self.system, case,
                                  self.protocol, seed_base, want_logprobs=True, abort=abort)
            if not traj["runtime_error"] or traj["aborted"]:
                break
            time.sleep(5 * (attempt + 1))
        traj.update({"sample_index": k, "round": rnd, "attempts": attempt + 1})
        if traj["runtime_error"] or traj["aborted"]:
            return {"traj": traj, "failed": traj["runtime_error"], "aborted": traj["aborted"], "futures": {}}
        seq = build_sequence(traj, self.tokenizer, self.system, case)
        futures = {}
        if self.judge is not None:
            reqs = (self.rmod.judge_requests(traj["result"], case, self.refs[cid], self.v2) if self.v2
                    else judge_requests(traj["result"], case, self.refs[cid], self.rcfg))
            for dim, req in reqs.items():
                futures[dim] = self.judge.submit(dim, req[0], req[1], model_closed(traj["result"]),
                                                 *(req[2:3] if len(req) > 2 and req[2] is not None else ()))
        return {"traj": traj, "seq": seq, "futures": futures, "failed": False, "aborted": False}

    def evaluate_group(self, cid: str, group: list[dict]) -> tuple[str, list[dict]]:
        if any(o["aborted"] for o in group):
            return "straggler", group
        if any(o["failed"] for o in group):
            return "infra", group
        if any(o.get("seq") is None or "error" in o["seq"] for o in group):
            return "sequence", group
        for o in group:
            o["judged"] = {dm: f.result() for dm, f in o["futures"].items()}
        failed = {dm for o in group for dm, r in o["judged"].items() if "error" in r}
        if "closure" in failed or (failed and self.cfg["reward"].get("judge_failure_policy") == "drop_group"):
            return "judge_failure", group
        for o in group:
            if self.v2:
                o["reward"] = self.rmod.compute(o["traj"]["result"], self.cases[cid], self.refs[cid], self.v2,
                                                o["judged"], excluded=failed)
            else:
                o["reward"] = compute(o["traj"]["result"], self.cases[cid], self.refs[cid], self.rcfg,
                                      o["judged"], excluded=failed)
            o["behaviour"] = behaviour.stats(o["traj"]["result"], self.cases[cid], self.refs[cid],
                                             o["traj"]["turn_rl"])
        rewards = [o["reward"]["total"] for o in group]
        if self.cfg["reward"].get("center_by_decision"):
            centred = center_by_decision(rewards, [o["reward"].get("model_closed") for o in group],
                                         [o["reward"]["components"]["D"] for o in group])
            for o, c in zip(group, centred):
                o["reward"]["total_raw"] = o["reward"]["total"]
                o["reward"]["components"]["decision_centering_shift"] = c - o["reward"]["total"]
                o["reward"]["total"] = c
            rewards = centred
        if self.g.get("drop_identical_groups", True) and group_is_degenerate(rewards):
            return "identical_rewards", group
        for o, a in zip(group, group_advantages(rewards, bool(self.g.get("adv_std_normalize", True)))):
            o["advantage"] = a
        return "kept", group

    def make_batch(self, b: int, v: int, name: str) -> dict:
        began = time.perf_counter()
        G = int(self.g["group_size"])
        n_first, min_groups = int(self.g["cases_per_step"]), int(self.g.get("min_groups", 32))
        cut_frac = float(self.g.get("straggler_done_fraction", 0.9))
        if self.judge:
            self.judge.reset_window()
        sampler_before = self.sampler.state()
        gen0 = [vllm_client.scrape_generation_tokens(ep) for ep in self.endpoints]
        kept, dropped, records, used, stds = [], collections.Counter(), [], set(), []
        keep_rate, rounds, straggler_events, gen_done_at = 1.0, 0, 0, None
        for rnd in range(int(self.g.get("max_refill_rounds", 2)) + 1):
            if rnd > 0 and len(kept) >= min_groups:
                break
            n = n_first if rnd == 0 else math.ceil((min_groups - len(kept)) / max(keep_rate, 0.25))
            cids = self.sampler.draw(n, avoid=used)
            used.update(cids)
            abort = threading.Event()
            futs = {}
            for i, cid in enumerate(cids):
                ep = self.endpoints[(self.dispatch + i) % len(self.endpoints)]
                for k in range(G):
                    futs[(cid, k)] = self.rollout_pool.submit(self.one_trajectory, b, rnd, cid, k, name, ep, abort)
            self.dispatch += len(cids)
            evaluated: dict[str, str] = {}
            rollouts_done_at: dict[str, float] = {}
            deadline = float(self.cfg["judges"]["train"].get("group_deadline_seconds", 600)) if self.judge else 0
            valid_round = 0
            while True:
                done = sum(f.done() for f in futs.values())
                for cid in cids:
                    if cid in evaluated or not all(futs[(cid, k)].done() for k in range(G)):
                        continue
                    rollouts_done_at.setdefault(cid, time.perf_counter())
                    group = [futs[(cid, k)].result() for k in range(G)]
                    if not all(f.done() for o in group for f in o["futures"].values()):
                        if time.perf_counter() - rollouts_done_at[cid] < deadline:
                            continue
                        status = "judge_deadline"
                        for o in group:
                            o["futures"] = {}
                        evaluated[cid] = status
                        dropped[status] += 1
                        for o in group:
                            records.append(self.record(b, v, o, status))
                        continue
                    status, group = self.evaluate_group(cid, group)
                    evaluated[cid] = status
                    for o in group:
                        records.append(self.record(b, v, o, status))
                    if "reward" in group[0]:
                        stds.append(group_std([o["reward"]["total"] for o in group]))
                    if status == "kept":
                        kept.append({"case_id": cid, "items": group})
                        valid_round += 1
                    else:
                        dropped[status] += 1
                if not abort.is_set() and len(kept) >= min_groups and done >= cut_frac * len(futs) \
                        and done < len(futs):
                    abort.set()
                    straggler_events += 1
                    log({"event": "straggler_cutoff", "batch": b, "done": done, "total": len(futs),
                         "kept_groups": len(kept)})
                if len(evaluated) == len(cids):
                    break
                if time.perf_counter() - getattr(self, "_last_progress", 0) > 60:
                    self._last_progress = time.perf_counter()
                    log({"event": "progress", "batch": b, "round": rnd, "trajectories_done": done,
                         "trajectories": len(futs), "groups_evaluated": len(evaluated), "kept": len(kept),
                         "judge_backlog": (self.judge.submitted - self.judge.finished) if self.judge else None,
                         "seconds": round(time.perf_counter() - began, 1)})
                if done == len(futs) and gen_done_at is None:
                    gen_done_at = time.perf_counter()
                time.sleep(0.2)
            keep_rate = valid_round / max(len(cids), 1)
            rounds += 1
        gen1 = [vllm_client.scrape_generation_tokens(ep) for ep in self.endpoints]
        gen_tokens = sum((b1 or 0) - (a or 0) for a, b1 in zip(gen0, gen1))
        total_s = time.perf_counter() - began
        judge_window = self.judge.reset_window() if self.judge else None
        batch = {"index": b, "version": v, "lora_name": name, "groups": kept, "records": records,
                 "dropped": dict(dropped), "refill_rounds": rounds - 1, "group_reward_std": stds,
                 "straggler_cutoffs": straggler_events,
                 "sampler_before": sampler_before, "sampler_after": self.sampler.state(),
                 "rollout_seconds": total_s,
                 "judge_tail_seconds": (time.perf_counter() - gen_done_at) if gen_done_at else 0.0,
                 "gen_tokens": gen_tokens, "judge": judge_window}
        self.save_records(b, records)
        return batch

    def record(self, b: int, v: int, o: dict, status: str) -> dict:
        t = o["traj"]
        r = t["result"]
        rec = {"batch": b, "version": v, "case_id": t["case_id"], "source": t["source"],
               "sample": t.get("sample_index"), "round": t.get("round"), "group_status": status,
               "status": r["status"], "failed": o["failed"], "aborted": o["aborted"],
               "turns": r.get("assistant_turns"), "fetch_rounds": r.get("fetch_rounds"),
               "unique_fetched": len(r.get("unique_fetched_evidence_ids") or []),
               "generated_tokens": t.get("generated_tokens"), "conclusion_marker": r.get("conclusion_marker"),
               "reference_closure": self.refs[t["case_id"]]["closure"], "advantage": o.get("advantage"),
               "final_answer": r.get("final_answer"), "errors": r.get("errors")}
        if "reward" in o:
            rec["reward"] = o["reward"]
            rec["behaviour"] = o["behaviour"]
            rec["judge"] = {dm: {k: x for k, x in j.items()} for dm, j in o.get("judged", {}).items()}
        seq = o.get("seq") or {}
        rec["sequence"] = {k: seq.get(k) for k in ("target_tokens", "turns_ok", "turns_mismatch",
                                                   "turn_status", "error")}
        rec["sequence"]["tokens"] = len(seq.get("input_ids") or [])
        rec["trajectory"] = r
        if self.cfg["logging"].get("save_token_ids"):
            rec["turn_rl"] = t.get("turn_rl")
        return rec

    def save_records(self, b: int, records: list[dict]) -> None:
        path = self.run_dir / "rollouts" / f"batch_{b:05d}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".partial")
        with gzip.open(tmp, "wt") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        tmp.replace(path)

    def plan_minibatches(self, batch: dict, step: int) -> list[list[list[dict]]]:
        items = [{"input_ids": o["seq"]["input_ids"], "loss_mask": o["seq"]["loss_mask"],
                  "behaviour_logprobs": o["seq"]["behaviour_logprobs"], "advantage": o["advantage"]}
                 for g in batch["groups"] for o in g["items"] if o["seq"]["target_tokens"] > 0]
        random.Random(1_000_003 * step + 17).shuffle(items)
        M = int(self.g.get("minibatches_per_step", 4))
        plan = []
        for m in range(M):
            part = items[m::M]
            shards = [[] for _ in range(self.d.world)]
            load = [0] * self.d.world
            for it in sorted(part, key=lambda x: -len(x["input_ids"])):
                r = load.index(min(load))
                shards[r].append(it)
                load[r] += len(it["input_ids"])
            plan.append(shards)
        return plan

    def train_step(self, step: int, batch: dict) -> dict:
        import torch
        times = {}
        plan = self.plan_minibatches(batch, step)
        lr = float(self.g["lr"])
        warm = int(self.g.get("warmup_steps", 0))
        if warm and step < warm:
            lr = lr * (step + 1) / warm
        for _, done in self.eval_threads:
            done.wait()
        self.eval_threads = [(t, e) for t, e in self.eval_threads if not e.is_set()]
        if self.colocate:
            t0 = time.perf_counter()
            for ep in self.endpoints:
                vllm_client.sleep(ep, int(self.pipe.get("sleep_level", 1)))
            times["sleep"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        tmp = self.run_dir / "tmp"
        tmp.mkdir(exist_ok=True)
        shard_paths = []
        for r in range(self.d.world):
            p = tmp / f"step_{step:05d}_rank{r}.pt"
            torch.save([mb[r] for mb in plan], p)
            shard_paths.append(str(p))
        clip_low, clip_high = float(self.g["clip_low"]), float(self.g["clip_high"])
        kl_coef = float(self.g.get("kl_coef", 0.0))
        self.d.bcast({"op": "update", "paths": shard_paths, "clip_low": clip_low, "clip_high": clip_high,
                      "kl_coef": kl_coef, "lr": lr})
        upd = self.policy.update([mb[0] for mb in plan], clip_low, clip_high, kl_coef, lr)
        times["train"] = time.perf_counter() - t0
        for p in shard_paths:
            Path(p).unlink(missing_ok=True)
        t0 = time.perf_counter()
        ksub = int(self.cfg.get("monitor", {}).get("kl_sequences", 0))
        kl = None
        if ksub:
            pool = [it for mb in plan for it in mb[0]]
            kl = self.policy.kl_to_ref(random.Random(step).sample(pool, min(ksub, len(pool))))
        new_v = step + 1
        adapter_dir = self.run_dir / "adapters" / f"v{new_v:05d}_a{self.attempt}"
        self.policy.save_adapter(adapter_dir)
        probe = self.policy.probe(self.probe["input_ids"], self.probe["loss_mask"])
        self.policy.release_memory()
        times["kl_save_probe"] = time.perf_counter() - t0
        if self.colocate:
            import torch
            if torch.cuda.is_available():
                mem = {"allocated_gb": torch.cuda.memory_allocated() / 1e9,
                       "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                       "device_used_gb": (lambda f, t: (t - f) / 1e9)(*torch.cuda.mem_get_info())}
                times["_trainer_mem_before_wake"] = mem
                log({"event": "trainer_memory_before_wake", "step": step, **mem})
            t0 = time.perf_counter()
            for ep in self.endpoints:
                vllm_client.wake(ep)
            times["wake"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        check = self.publish(new_v, adapter_dir, probe)
        times["swap"] = time.perf_counter() - t0
        return {"update": upd, "check": check, "kl": kl, "lr": lr, "times": times, "adapter_dir": adapter_dir}

    def train(self) -> int:
        if self.mode == "async":
            return self.train_async()
        step = self.start_step
        code = 0
        if self.cfg["eval"].get("at_start") and step == 0 and not self.resume:
            self.start_eval(0)
        while step < self.total_steps and not self.stop.is_set():
            t_step = time.perf_counter()
            batch = self.make_batch(step, self.published, self.version_names[self.published])
            if self.judge_alert(step, batch):
                self.checkpoint(step, self.current_adapter(step), batch["sampler_before"], "paused")
                code = PAUSE_EXIT
                break
            res = self.train_step(step, batch)
            self.poll_stop(step + 1)
            step = self.after_step(step, batch, res, 0, time.perf_counter() - t_step)
        self.shutdown()
        return code

    def poll_stop(self, new_step: int) -> bool:
        for f in self.stop_files:
            if f.exists():
                honoured = f.with_name(f"{f.name}.honoured_step{new_step:05d}_{int(time.time())}")
                try:
                    f.rename(honoured)
                except OSError:
                    honoured = f
                self.stop_reason = str(f)
                log({"event": "stop_file", "path": str(f), "moved_to": str(honoured), "step": new_step,
                     "action": "checkpoint this step, stop all ranks, exit 0"})
                self.stop.set()
        return self.stop.is_set()

    def current_adapter(self, version: int) -> Path:
        if version == self.start_step and self.latest_checkpoint() is None:
            return Path(self.paths["init_adapter"])
        cand = self.run_dir / "adapters" / f"v{version:05d}_a{self.attempt}"
        if cand.exists():
            return cand
        ck = self.run_dir / "checkpoints" / f"step_{version:05d}" / "adapter"
        return ck if ck.exists() else Path(self.paths["init_adapter"])

    def after_step(self, step, batch, res, staleness, step_seconds) -> int:
        new_v = step + 1
        ck_every = int(self.cfg["checkpoint"]["every_steps"])
        if new_v % ck_every == 0 or new_v == self.total_steps or self.stop.is_set():
            kind = ("stop_file" if self.stop_reason else "stop_signal" if self.stop.is_set()
                    else "final" if new_v == self.total_steps else "periodic")
            self.checkpoint(new_v, res["adapter_dir"], batch["sampler_after"], kind)
        self.prune_adapters(new_v)
        ev = int(self.cfg["eval"].get("every_steps") or 0)
        if ev and (new_v % ev == 0 or (new_v == self.total_steps and self.cfg["eval"].get("at_end"))) \
                and not self.stop_reason:
            self.start_eval(new_v)
        metrics = self.step_metrics(batch, res, staleness, step_seconds)
        self.write_metrics(new_v, batch, metrics)
        sc = int(self.cfg.get("monitor", {}).get("rank_sync_check_every", 5))
        if self.d.world > 1 and sc and new_v % sc == 0:
            self.d.bcast({"op": "fingerprint"})
            prints = self.d.gather(self.policy.adapter_fingerprint())
            if max(prints) - min(prints) > 1e-6 * max(1.0, abs(prints[0])):
                raise RuntimeError(f"ranks diverged: {prints}")
        return new_v

    def train_async(self) -> int:
        if self.d.world != 1 or self.colocate:
            raise ValueError("async mode runs one trainer process on its own GPU (no colocation)")
        prod = threading.Thread(target=self.producer, name="producer", daemon=True)
        prod.start()
        step, last = self.start_step, time.perf_counter()
        while step < self.total_steps:
            batch = None
            while batch is None and not (self.stop.is_set() and self.batches.empty()):
                if self.producer_error:
                    raise RuntimeError("producer failed:\n" + self.producer_error)
                try:
                    batch = self.batches.get(timeout=1.0)
                except queue.Empty:
                    pass
            if batch is None:
                break
            staleness = step - batch["version"]
            if batch["index"] != step or not 0 <= staleness <= self.max_staleness:
                raise RuntimeError("staleness bound broken")
            if self.judge_alert(step, batch):
                self.stop.set()
                self.checkpoint(step, self.current_adapter(step), batch["sampler_before"], "paused")
                self.shutdown()
                return PAUSE_EXIT
            res = self.train_step(step, batch)
            if self.poll_stop(step + 1):
                self.stop.set()
            step = self.after_step(step, batch, res, staleness, time.perf_counter() - last)
            if self.stop_reason:
                break
            last = time.perf_counter()
        self.stop.set()
        prod.join(timeout=600)
        self.shutdown()
        return 0

    def producer(self) -> None:
        try:
            b = self.start_step
            while b < self.total_steps and not self.stop.is_set():
                with self.cv:
                    self.cv.wait_for(lambda: self.published >= b - self.max_staleness or self.stop.is_set())
                    if self.stop.is_set():
                        break
                    v = self.published
                    name = self.version_names[v]
                    self.in_use[v] += 1
                try:
                    batch = self.make_batch(b, v, name)
                finally:
                    with self.cv:
                        self.in_use[v] -= 1
                while not self.stop.is_set():
                    try:
                        self.batches.put(batch, timeout=1.0)
                        break
                    except queue.Full:
                        continue
                b += 1
        except Exception as exc:
            self.producer_error = traceback.format_exc()
            log({"event": "producer_error", "error": f"{type(exc).__name__}: {exc}"})
            self.stop.set()

    def shutdown(self) -> None:
        self.d.bcast({"op": "stop"})
        for t, _ in self.eval_threads:
            t.join()
        for t in self.score_threads:
            t.join(timeout=float(self.cfg["eval"].get("llm_join_timeout_seconds", 1800)))
        for j in (self.judge, self.eval_judge, *self.eval_judges.values()):
            if j:
                j.close()
        with self.cv:
            names = [n for _, n in self.loaded]
            self.loaded = []
        for n in names:
            for ep in self.endpoints:
                try:
                    vllm_client.unload_lora(ep, n)
                except Exception:
                    pass
        self.rollout_pool.shutdown(wait=False, cancel_futures=True)
        self.tb.flush()
        log({"event": "finished", "published_version": self.published, "alerts": self.alerts,
             "stopped_by": self.stop_reason})

    def judge_alert(self, step: int, batch: dict) -> bool:
        w = batch.get("judge")
        if not w or not w["calls"]:
            return False
        mon = self.cfg.get("monitor", {})
        reasons = []
        if (w["failure_rate"] or 0) > float(mon.get("max_judge_failure_rate", 0.05)):
            reasons.append(f"judge failure rate {w['failure_rate']:.3f}")
        if (w["parse_failure_rate"] or 0) > float(mon.get("max_judge_parse_failure_rate", 0.02)):
            reasons.append(f"judge JSON/parse failure rate {w['parse_failure_rate']:.3f}")
        if (w.get("inconsistent_rate") or 0) > float(mon.get("max_judge_inconsistent_rate", 0.10)):
            reasons.append(f"judge n/a-answer (inconsistent) rate {w['inconsistent_rate']:.3f}")
        if not reasons:
            return False
        alert = {"step": step, "reasons": reasons, "judge": w, "time": time.time(),
                 "action": "training paused before using this batch; fix and restart with --resume"}
        atomic_json(self.run_dir / "PAUSED_ALERT.json", alert)
        self.alerts.append(alert)
        log({"event": "ALERT_PAUSE", **alert})
        return True

    def prune_adapters(self, newest: int) -> None:
        keep = int(self.cfg["checkpoint"].get("keep_recent_adapters", 3))
        root = self.run_dir / "adapters"
        with self.cv:
            live = {n for _, n in self.loaded} | self.pinned
        for p in root.glob("*"):
            name = p.name
            try:
                v = int(name.rsplit("_v", 1)[1][:5]) if "_v" in name else int(name[1:6])
            except (IndexError, ValueError):
                continue
            base = name[:-len("_lmprefix")] if name.endswith("_lmprefix") else None
            if v <= newest - keep and (base is None or base not in live):
                shutil.rmtree(p, ignore_errors=True)

    def checkpoint(self, step: int, adapter_dir: Path, sampler_state: dict, kind: str) -> None:
        import torch
        d = self.run_dir / "checkpoints" / f"step_{step:05d}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        shutil.copytree(adapter_dir, d / "adapter")
        cost = self.judge_cost_prior + (self.judge.cumulative()["cost_usd"] if self.judge else 0.0)
        torch.save({"step": step, "trainer": self.policy.trainer_state(), "sampler_state": sampler_state,
                    "judge_cost_usd": cost, "config_sha256": self.cfg["_config_sha256"], "kind": kind},
                   d / "trainer_state.pt")
        atomic_json(d / "manifest.json", {"step": step, "kind": kind,
                                          "adapter_sha256": sha256_file(d / "adapter" / "adapter_model.safetensors"),
                                          "sampler_state": sampler_state, "judge_cost_usd": cost, "saved": time.time()})
        (d / "COMPLETE").write_text("ok\n")
        log({"event": "checkpoint", "step": step, "kind": kind, "dir": str(d)})

    def start_eval(self, step: int) -> None:
        if self.cfg["eval"].get("design") == "v22":
            return self.start_eval_v22(step)
        self.start_eval_v1(step)

    def start_eval_v22(self, step: int) -> None:
        ecfg = self.cfg["eval"]
        name = f"eval_{self.cfg['rollout']['lora_prefix']}_{self.cfg['run_name']}_v{step:05d}_a{self.attempt}"
        src, src_name = self.version_dirs[step], self.version_names[step]
        samples = int(ecfg.get("final_samples", ecfg.get("samples", 1))) if step == self.total_steps \
            else int(ecfg.get("samples", 1))
        with self.cv:
            self.pinned.add(src_name)
        done = threading.Event()
        out_dir = self.run_dir / "eval" / f"step_{step:05d}"

        def score(records):
            try:
                cases = {c["case_id"]: c for c in read_jsonl(refuse_test_path(self.paths["val_bundle"]))}
                refs = {r["case_id"]: r for r in read_jsonl(refuse_test_path(self.paths["val_judge_refs"]))}
                cfg22 = reward_v22.RewardV2Config.from_dict({**self.cfg["reward"], "variant": "outcome",
                                                              "quality_only_if_correct": False})
                with ThreadPoolExecutor(max_workers=max(1, len(self.eval_judges))) as ex:
                    futs = {lab: ex.submit(evaluation_v22.llm_scores, records, j, cases, refs, cfg22)
                            for lab, j in self.eval_judges.items()}
                    res = {lab: f.result() for lab, f in futs.items()}
                for lab, r in res.items():
                    atomic_json(out_dir / f"llm_{lab}.json", r)
                    self.tb.scalars(f"eval_llm_{lab}", r["summary"], step)
                cmp = evaluation_v22.compare_judges(res["luna"], res["sol"]) if {"luna", "sol"} <= set(res) else {}
                atomic_json(out_dir / "llm_compare.json", cmp)
                self.tb.scalars("eval_llm_compare", cmp, step)
                self.tb.flush()
                log({"event": "eval_llm", "step": step,
                     **{f"{lab}_q_conc": r["summary"]["q_conc_mean"] for lab, r in res.items()},
                     **{f"{lab}_scored": r["summary"]["scored"] for lab, r in res.items()},
                     **{f"{lab}_failed": r["summary"]["failed"] for lab, r in res.items()},
                     "luna_minus_sol_q_conc": cmp.get("luna_minus_sol_q_conc")})
            except Exception as exc:
                log({"event": "eval_llm_failed", "step": step, "error": f"{type(exc).__name__}: {exc}",
                     "trace": traceback.format_exc()[-1500:]})

        def work():
            records = []
            try:
                for ep in self.endpoints:
                    vllm_client.load_lora(ep, name, src)
                summary, records = evaluation_v22.run_eval(step, name, src, ecfg, self.paths,
                                                           Path(self.paths["eval_protocol"]), self.endpoints,
                                                           self.run_dir, samples)
                self.tb.scalars("eval", summary["program"], step)
                self.tb.scalars("eval_cf", summary["counterfactual"], step)
                if "program_sample0" in summary:
                    self.tb.scalars("eval_final_s0", summary["program_sample0"], step)
                    self.tb.scalars("eval_final_s0_cf", summary["counterfactual_sample0"], step)
                self.tb.flush()
                p, c = summary["program"], summary["counterfactual"]
                log({"event": "eval", "step": step, "samples": samples, "records": p.get("records"),
                     "closure_balanced_acc": p.get("closure_balanced_acc"),
                     "should_close": p.get("closure_acc_should_close"),
                     "should_not_close": p.get("closure_acc_should_not_close"),
                     "host_balanced": (p.get("host_teacher_label") or {}).get("balanced"),
                     "report_balanced": (p.get("report_label") or {}).get("balanced"),
                     "valid_citation_rate": p.get("valid_citation_rate"), "invalid_fetch_rate": p.get("invalid_fetch_rate"),
                     "missing_marker_rate": p.get("missing_marker_rate"), "turns_per_case": p.get("turns_per_case"),
                     "cf_closed_full": c.get("closed_rate_full"), "cf_closed_removed": c.get("closed_rate_removed"),
                     "cf_closed_control": c.get("closed_rate_control"), "cf_removed_minus_control": c.get("removed_minus_control"),
                     "scheduler_exit_code": summary.get("scheduler_exit_code")})
            except Exception as exc:
                log({"event": "eval_failed", "step": step, "error": f"{type(exc).__name__}: {exc}",
                     "trace": traceback.format_exc()[-1500:]})
            finally:
                with self.cv:
                    self.pinned.discard(src_name)
                for ep in self.endpoints:
                    try:
                        vllm_client.unload_lora(ep, name)
                    except Exception:
                        pass
                done.set()
            if self.eval_judges and records:
                t2 = threading.Thread(target=score, args=(records,), name=f"evalllm{step}", daemon=True)
                t2.start()
                self.score_threads.append(t2)

        t = threading.Thread(target=work, name=f"eval{step}", daemon=True)
        t.start()
        self.eval_threads.append((t, done))

    def start_eval_v1(self, step: int) -> None:
        ecfg = self.cfg["eval"]
        name = f"eval_{self.cfg['rollout']['lora_prefix']}_{self.cfg['run_name']}_v{step:05d}_a{self.attempt}"
        src, src_name = self.version_dirs[step], self.version_names[step]
        with self.cv:
            self.pinned.add(src_name)
        rcfg2 = RewardConfig.from_dict({**self.cfg["reward"], "phase": 2})
        done = threading.Event()

        def score(records):
            try:
                out = evaluation.llm_closure_scores(records, self.paths, self.eval_judge, rcfg2,
                                                    ecfg.get("llm_closure_subset"))
                atomic_json(self.run_dir / "eval" / f"step_{step:05d}" / "llm_closure.json", out)
                self.tb.scalars("eval_llm", {k: v for k, v in out.items() if not isinstance(v, (list, dict))}, step)
                self.tb.hist("eval_llm/raw_conclusion", out["raw_conclusion_scores"], step)
                self.tb.flush()
                log({"event": "eval_llm_closure", "step": step, "scored": out["scored"], "failed": out["failed"],
                     "conclusion_raw_mean": out["conclusion_raw_mean"]})
            except Exception as exc:
                log({"event": "eval_llm_failed", "step": step, "error": f"{type(exc).__name__}: {exc}"})

        def work():
            records = []
            try:
                for ep in self.endpoints:
                    vllm_client.load_lora(ep, name, src)
                summary, records = evaluation.run_eval(step, name, src, ecfg, self.paths,
                                                       Path(self.paths["eval_protocol"]), self.endpoints, self.run_dir)
                self.tb.scalars("eval", summary, step)
                self.tb.flush()
                log({"event": "eval", "step": step, **{k: summary.get(k) for k in
                     ("records", "closure_balanced_acc", "valid_citation_rate", "final_rate", "turns_per_case",
                      "scheduler_exit_code")}})
            except Exception as exc:
                log({"event": "eval_failed", "step": step, "error": f"{type(exc).__name__}: {exc}"})
            finally:
                with self.cv:
                    self.pinned.discard(src_name)
                for ep in self.endpoints:
                    try:
                        vllm_client.unload_lora(ep, name)
                    except Exception:
                        pass
                done.set()
            if self.eval_judge is not None and records:
                t2 = threading.Thread(target=score, args=(records,), name=f"evalllm{step}", daemon=True)
                t2.start()
                self.score_threads.append(t2)

        t = threading.Thread(target=work, name=f"eval{step}", daemon=True)
        t.start()
        self.eval_threads.append((t, done))

    def step_metrics(self, batch: dict, res: dict, staleness: int, step_seconds: float) -> dict:
        recs = [r for r in batch["records"] if "reward" in r]
        n_all = len(batch["records"]) or 1
        seen = len(batch["groups"]) + sum(batch["dropped"].values())
        by = lambda key: collections.defaultdict(list)
        comp, dims = by(0), by(0)
        rew_src, rew_lab = by(0), by(0)
        for r in recs:
            for k, v in r["reward"]["components"].items():
                comp[k].append(v)
            for k, v in r["reward"]["dimensions"].items():
                dims[k].append(v)
            rew_src[r["source"]].append(r["reward"]["total"])
            rew_lab[r["reference_closure"]].append(r["reward"]["total"])
        totals = [r["reward"]["total"] for r in recs]
        m_tot = mean(totals)
        closed_src, closed_lab, acc_src = by(0), by(0), by(0)
        for r in batch["records"]:
            if r["failed"] or r["aborted"]:
                continue
            closed = float(r["conclusion_marker"] == "CASE CLOSED")
            closed_src[r["source"]].append(closed)
            closed_lab[r["reference_closure"]].append(closed)
            dec = "closed" if closed else "not_closed" if r["conclusion_marker"] == "CASE NOT CLOSED" else None
            acc_src[r["source"]].append(float(dec == r["reference_closure"]))
        finished = [r for r in batch["records"] if not r["failed"] and not r["aborted"]]
        n_fin = len(finished) or 1
        beh = by(0)
        for r in recs:
            for k, v in r["behaviour"].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    beh[k].append(v)
                elif isinstance(v, bool):
                    beh[k].append(float(v))
        seqs = [r["sequence"] for r in recs]
        turns_total = sum(len(s.get("turn_status") or []) for s in seqs)
        upd = res["update"]
        first_ratio = upd.get("first_minibatch_ratio_mean")
        ratio_tol = float(self.cfg.get("monitor", {}).get("first_ratio_tolerance", 0.05))
        if first_ratio is not None and abs(first_ratio - 1.0) > ratio_tol:
            self.alerts.append({"step": batch["index"], "warning": f"first-minibatch ratio {first_ratio:.4f}"})
            log({"event": "WARNING_importance_ratio", "step": batch["index"], "first_minibatch_ratio_mean": first_ratio})
        if self.v2:
            judge_prog = self.v2_split(recs)
        else:
            judge_prog = {"judge_dims_mean": mean([r["reward"]["components"]["fetch_judge"] +
                                                  r["reward"]["components"]["hypothesis_judge"] +
                                                  r["reward"]["components"]["closure_conclusion"] +
                                                  r["reward"]["components"]["closure_gap"] for r in recs]),
                          "program_dims_mean": mean([r["reward"]["components"]["key_hit"] +
                                                    r["reward"]["components"]["closure_status"] +
                                                    r["reward"]["dimensions"]["structure"] for r in recs])}
        closed_all = mean([float(r["conclusion_marker"] == "CASE CLOSED") for r in finished])
        t = {**{k: v for k, v in res["times"].items() if not k.startswith("_")}, "generate_and_judge": batch["rollout_seconds"],
             "judge_tail_after_generation": batch["judge_tail_seconds"], "step": step_seconds}
        out = {
            "reward": {"total_mean": m_tot,
                       "total_std": (sum((x - m_tot) ** 2 for x in totals) / len(totals)) ** 0.5 if totals else None,
                       **{f"dim_{k}": mean(v) for k, v in dims.items()},
                       **{f"comp_{k}": mean(v) for k, v in comp.items()},
                       **{f"by_source_{k}": mean(v) for k, v in rew_src.items()},
                       **{f"by_label_{k}": mean(v) for k, v in rew_lab.items()},
                       "judge_excluded_dims": sum(len(r["reward"]["judge_excluded"]) for r in recs) / max(len(recs), 1)},
            "groups": {"kept": len(batch["groups"]), "seen": seen,
                       "dropped_share": sum(batch["dropped"].values()) / seen if seen else None,
                       **{f"dropped_{k}": v for k, v in batch["dropped"].items()},
                       "within_group_reward_std_mean": mean(batch["group_reward_std"]),
                       "refill_rounds": batch["refill_rounds"], "straggler_cutoffs": batch["straggler_cutoffs"],
                       "trajectories_aborted": sum(r["aborted"] for r in batch["records"])},
            "train": {**{k: v for k, v in upd.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
                      "grad_norm_mean": mean(upd.get("grad_norm") or []), "lr": res["lr"], "staleness": staleness,
                      "kl_to_e2_monitor": (res["kl"] or {}).get("kl_to_e2")},
            "behaviour": {"closed_rate": closed_all,
                          **{f"closed_rate_source_{k}": mean(v) for k, v in closed_src.items()},
                          **{f"closed_rate_label_{k}": mean(v) for k, v in closed_lab.items()},
                          **{f"closure_accuracy_source_{k}": mean(v) for k, v in acc_src.items()},
                          "closure_accuracy": mean([x for v in acc_src.values() for x in v]),
                          "no_marker_rate": sum(r["conclusion_marker"] is None for r in finished) / n_fin,
                          "turn_limit_rate": sum(r["status"] == "turn_limit" for r in finished) / n_fin,
                          "wall_budget_rate": sum(r["status"] == "wall_budget" for r in finished) / n_fin,
                          "context_limit_rate": sum(r["status"] == "context_limit" for r in finished) / n_fin,
                          "turns": mean([r["turns"] for r in finished]),
                          "unique_fetched_items": mean([r["unique_fetched"] for r in finished]),
                          "generated_tokens": mean([r["generated_tokens"] for r in finished]),
                          **{k: mean(v) for k, v in beh.items()},
                          "sequence_tokens": mean([s.get("tokens") for s in seqs]),
                          "target_tokens": mean([s.get("target_tokens") for s in seqs]),
                          "turn_token_mismatch_rate": (sum(s.get("turns_mismatch") or 0 for s in seqs) / turns_total) if turns_total else None},
            "hacking_signals": {**judge_prog, "final_answer_chars": mean(beh["final_answer_chars"]),
                                "tokens_per_turn": mean(beh["tokens_per_turn"]),
                                "closed_rate_extreme": float(closed_all is not None and (closed_all < 0.05 or closed_all > 0.95))},
            "judge": {**{k: v for k, v in (batch["judge"] or {}).items() if not isinstance(v, dict)},
                      **{f"error_{k}": v for k, v in ((batch["judge"] or {}).get("errors") or {}).items()},
                      **{f"parse_{k}": v for k, v in ((batch["judge"] or {}).get("parse_failures") or {}).items()},
                      **{f"usage_{k}": v for k, v in ((batch["judge"] or {}).get("per_key") or {}).items()},
                      "cost_usd_cumulative": self.judge_cost_prior + (self.judge.cumulative()["cost_usd"] if self.judge else 0.0)},
            "judge_providers": dict((batch["judge"] or {}).get("providers") or {}),
            "lora": {"vllm_probe_loss": res["check"]["vllm_loss"], "trainer_probe_loss": res["check"]["trainer_loss"],
                     "vllm_trainer_abs_loss_diff": res["check"]["abs_loss_diff"],
                     "mean_abs_token_logprob_diff": res["check"]["mean_abs_token_logprob_diff"],
                     "adapter_fingerprint": self.policy.adapter_fingerprint()},
            "time": t,
            "system": {"gen_tokens": batch["gen_tokens"],
                       "gen_tokens_per_second": batch["gen_tokens"] / batch["rollout_seconds"] if batch["rollout_seconds"] else None,
                       "trainer_peak_memory_gb_rank0": upd.get("peak_memory_gb")},
        }
        if self.v2:
            out["reward_v2"] = self.v2_metrics(recs)
        return out

    def v2_split(self, recs: list[dict]) -> dict:
        judge, prog = [], []
        for r in recs:
            c = r["reward"]["components"]
            j = c["judge_part"]
            judge.append(j)
            prog.append(r["reward"]["total"] - j)
        return {"judge_dims_mean": mean(judge), "program_dims_mean": mean(prog)}

    def v2_metrics(self, recs: list[dict]) -> dict:
        out = {}
        tot = collections.defaultdict(lambda: [0, 0])
        by_lab = collections.defaultdict(lambda: [0, 0])
        for r in recs:
            for q, (y, n) in (r["reward"].get("answers") or {}).items():
                tot[q][0] += y
                tot[q][1] += n
                k = f"{q}_ref_{r['reference_closure']}"
                by_lab[k][0] += y
                by_lab[k][1] += n
        out.update({f"yes_rate_{q}": y / n for q, (y, n) in tot.items() if n})
        out.update({f"yes_rate_{q}": y / n for q, (y, n) in by_lab.items() if n})
        out.update({f"answered_{q}": n for q, (y, n) in tot.items()})
        cells = collections.Counter()
        for r in recs:
            mc = r["reward"]["model_closed"]
            dec = "none" if mc is None else "closed" if mc else "not_closed"
            cells[f"cell_model_{dec}__ref_{r['reference_closure']}"] += 1
        out.update({k: v / max(len(recs), 1) for k, v in cells.items()})
        out["decision_right_rate"] = mean([float(r["reward"]["components"]["D"] > 0) for r in recs])
        wrong_pos = [r for r in recs if r["reward"]["components"]["D"] < 0 and r.get("advantage") is not None]
        out["wrong_decision_positive_advantage_share"] = (
            sum(r["advantage"] > 0 for r in wrong_pos) / len(wrong_pos) if wrong_pos else None)
        return out

    def write_metrics(self, step: int, batch: dict, metrics: dict) -> None:
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **metrics}, default=str) + "\n")
        for prefix, values in metrics.items():
            if prefix != "judge_providers":
                self.tb.scalars(prefix, values, step)
        self.tb.scalars("judge/providers", metrics["judge_providers"], step)
        raw = collections.defaultdict(list)
        for r in batch["records"]:
            for dim, vals in (r.get("reward") or {}).get("raw_judge_scores", {}).items():
                if dim == "closure":
                    raw["closure_conclusion"].append(vals["conclusion"])
                    if vals.get("gap") is not None:
                        raw["closure_gap"].append(vals["gap"])
                else:
                    raw[dim].extend(vals)
        for dim, vals in raw.items():
            self.tb.hist(f"judge_raw/{dim}", vals, step)
        if self.v2:
            vals = collections.defaultdict(list)
            for r in batch["records"]:
                rw = r.get("reward")
                if not rw:
                    continue
                vals["total"].append(rw["total"])
                for k, v in rw["dimensions"].items():
                    vals[k].append(v)
                for k in ("q_conc", "q_gap", "conclusion_part", "Kcov", "c3_supported_share"):
                    if rw["components"].get(k) is not None:
                        vals[k].append(rw["components"][k])
                vals["fetch_per_request"].extend(rw.get("fetch_items") or [])
                vals["hypothesis_per_update"].extend(rw.get("hyp_items") or [])
            for k, v in vals.items():
                self.tb.hist_auto(f"reward_v2_hist/{k}", v, step)
        dump_every = int(self.cfg["logging"].get("dump_every", 10))
        if dump_every and step % dump_every == 0:
            self.dump(step, batch)
        self.tb.flush()
        log({"event": "step", "step": step, "reward_mean": metrics["reward"]["total_mean"],
             "kept_groups": metrics["groups"]["kept"], "dropped_share": metrics["groups"]["dropped_share"],
             "closure_accuracy": metrics["behaviour"]["closure_accuracy"],
             "first_ratio": metrics["train"].get("first_minibatch_ratio_mean"),
             "step_seconds": round(metrics["time"]["step"], 1), "time": {k: round(v, 1) for k, v in metrics["time"].items()},
             "judge_cost_usd": metrics["judge"].get("cost_usd"), "lora_diff": metrics["lora"]["vllm_trainer_abs_loss_diff"]})

    def dump(self, step: int, batch: dict) -> None:
        from nautil_harness_v2_2 import render_markdown
        recs = [r for r in batch["records"] if "reward" in r]
        pick = random.Random(step).sample(recs, min(int(self.cfg["logging"].get("dump_count", 5)), len(recs)))
        out = self.run_dir / "dumps" / f"step_{step:05d}"
        out.mkdir(parents=True, exist_ok=True)
        for r in pick:
            md = render_markdown(r["trajectory"])
            md += "\n## Reward\n\n```json\n" + json.dumps({k: r["reward"][k] for k in
                  ("total", "dimensions", "components", "raw_judge_scores", "judge_excluded")}, indent=1) + "\n```\n"
            md += "\n## Judge justifications\n\n```json\n" + json.dumps(r.get("judge"), indent=1, ensure_ascii=False)[:20000] + "\n```\n"
            (out / f"{r['case_id']}__s{r['sample']}.md").write_text(md)
        self.tb.text("dumps/cases", ", ".join(f"{r['case_id']}__s{r['sample']}" for r in pick), step)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, help="stop after this global step (smoke tests)")
    ap.add_argument("--judge-keys-stdin", action="store_true",
                    help="read NAME=value key lines from stdin (single-process runs only)")
    ap.add_argument("--set", action="append", default=[], help="override dotted.key=json_value")
    args = ap.parse_args(argv)
    overrides = {}
    for item in args.set:
        k, v = item.split("=", 1)
        try:
            overrides[k] = json.loads(v)
        except json.JSONDecodeError:
            overrides[k] = v
    cfg = load_config(args.config, overrides)
    d = Dist(cfg.get("pipeline", {}).get("dist_backend", "nccl"))
    if d.rank != 0:
        worker_loop(cfg, d)
        return 0
    if args.judge_keys_stdin:
        from .judge import read_keys_from_stdin
        read_keys_from_stdin(cfg["judges"]["train"]["key_env"])
    run = Run(cfg, d, args.resume, args.max_steps)

    def on_term(signum, frame):
        log({"event": "signal", "signal": signum, "action": "finish current step, checkpoint, exit"})
        run.stop.set()
    signal.signal(signal.SIGTERM, on_term)
    run.setup()
    return run.train()


if __name__ == "__main__":
    sys.exit(main())
