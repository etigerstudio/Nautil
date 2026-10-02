#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import transformers
import peft
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig, Qwen3_5ForCausalLM

from canary_qwen35_lora_ddp import evaluate, read_needed, sparse_sum_loss, tensors


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def lr_for_step(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def save_adapter(adapter, output_dir: Path, persistent_dir: Path, *, step: int,
                 eval_loss: float, plan_sha256: str, data_sha256: str) -> dict:
    checkpoint_dir = output_dir / f"step_{step:04d}"
    adapter_dir = checkpoint_dir / "adapter"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    adapter.save_pretrained(str(adapter_dir), safe_serialization=True)
    manifest = {
        "schema_version": "nautil.sft.adapter.v1", "model": "Qwen/Qwen3.5-9B",
        "step": step, "evaluation_cross_entropy": eval_loss,
        "plan_sha256": plan_sha256, "data_sha256": data_sha256,
        "format": "PEFT LoRA adapter; load on Qwen3.5-9B text backbone",
    }
    (checkpoint_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = output_dir / f"checkpoint_{step:04d}.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(adapter_dir, arcname="adapter")
        tar.add(checkpoint_dir / "manifest.json", arcname="manifest.json")
    archive_sha256 = digest(archive)
    persistent_dir.mkdir(parents=True, exist_ok=True)
    target = persistent_dir / f"qwen35_9b_lora_sft_step_{step:04d}.tar"
    pending = persistent_dir / f".{target.name}.partial"
    shutil.copyfile(archive, pending)
    os.replace(pending, target)
    archive.unlink()
    if digest(target) != archive_sha256:
        raise AssertionError("persistent adapter archive failed verification")
    return {"step": step, "eval_loss": eval_loss,
            "adapter_dir": str(adapter_dir), "persistent_archive": str(target),
            "archive_sha256": archive_sha256}


def save_training_state(adapter, optimizer, trainable_names: list[str], output_dir: Path,
                        persistent_dir: Path, *, step: int, rng_states: list[dict],
                        step_records: list[dict], evaluation_history: list[dict],
                        initial_eval: float, best_loss: float, best_checkpoint: dict | None,
                        plan_sha256: str, data_sha256: str, frozen_sha256: str,
                        scheduler_config: dict, gpu_memory: list[dict],
                        prior_resume_checkpoints: list[dict]) -> dict:
    checkpoint_dir = output_dir / f"resume_step_{step:04d}"
    adapter_dir = checkpoint_dir / "adapter"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    adapter.save_pretrained(str(adapter_dir), safe_serialization=True)
    state = {
        "schema_version": "nautil.sft.resume.v1", "completed_step": step,
        "next_pair_index": step, "optimizer": optimizer.state_dict(),
        "trainable_names": trainable_names, "rng_states": rng_states,
        "step_records": step_records, "evaluation_history": evaluation_history,
        "initial_eval": initial_eval, "best_loss": best_loss,
        "best_checkpoint": best_checkpoint, "plan_sha256": plan_sha256,
        "data_sha256": data_sha256, "frozen_sha256": frozen_sha256,
        "scheduler": scheduler_config, "gpu_memory": gpu_memory,
        "prior_resume_checkpoints": prior_resume_checkpoints,
    }
    torch.save(state, checkpoint_dir / "trainer_state.pt")
    manifest = {
        "schema_version": "nautil.sft.resume_manifest.v1", "step": step,
        "evaluation_cross_entropy": evaluation_history[-1]["loss"],
        "useful_tokens_seen": sum(row["useful_tokens"] for row in step_records),
        "supervised_tokens_seen": sum(row["supervised_tokens"] for row in step_records),
        "useful_tokens_per_second": round(
            sum(row["useful_tokens"] for row in step_records) /
            sum(row["seconds"] for row in step_records), 1),
        "gpu_memory": gpu_memory,
        "plan_sha256": plan_sha256, "data_sha256": data_sha256,
        "frozen_sha256": frozen_sha256,
    }
    (checkpoint_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = output_dir / f"resume_step_{step:04d}.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(adapter_dir, arcname="adapter")
        tar.add(checkpoint_dir / "trainer_state.pt", arcname="trainer_state.pt")
        tar.add(checkpoint_dir / "manifest.json", arcname="manifest.json")
    archive_sha256 = digest(archive)
    persistent_dir.mkdir(parents=True, exist_ok=True)
    target = persistent_dir / f"qwen35_9b_lora_sft_resume_step_{step:04d}.tar"
    pending = persistent_dir / f".{target.name}.partial"
    shutil.copyfile(archive, pending)
    os.replace(pending, target)
    archive.unlink()
    if digest(target) != archive_sha256:
        raise AssertionError("persistent resume archive failed verification")
    return {"step": step, "archive": str(target), "sha256": archive_sha256,
            "manifest": manifest}


def extract_resume(archive: Path, output_dir: Path) -> tuple[Path, dict]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    extracted = output_dir / "resume_input"
    extracted.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as tar:
        names = set(tar.getnames())
        if "trainer_state.pt" not in names or "adapter/adapter_config.json" not in names:
            raise ValueError("incomplete resume archive")
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError("unsafe resume archive path")
        tar.extractall(extracted)
    state = torch.load(extracted / "trainer_state.pt", map_location="cpu", weights_only=False)
    return extracted / "adapter", state


def resolve_best_adapter(checkpoint: dict, output_dir: Path) -> Path:
    adapter_dir = Path(checkpoint["adapter_dir"])
    if adapter_dir.is_dir():
        return adapter_dir
    archive = Path(checkpoint["persistent_archive"])
    if digest(archive) != checkpoint["archive_sha256"]:
        raise ValueError("best adapter archive hash changed")
    extracted = output_dir / "restored_best"
    extracted.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError("unsafe best adapter archive path")
        tar.extractall(extracted)
    return extracted / "adapter"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--persistent-dir", type=Path, required=True)
    ap.add_argument("--frozen-manifest", type=Path, required=True)
    ap.add_argument("--resume-from", type=Path)
    ap.add_argument("--init-from", type=Path)
    args = ap.parse_args()
    if (args.resume_from is None) == (args.init_from is None):
        raise ValueError("specify exactly one of --init-from or --resume-from")
    began = time.perf_counter()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise ValueError("expected exactly two H100 GPUs")
    torch.manual_seed(20260925)
    torch.cuda.manual_seed_all(20260925)
    torch.backends.cuda.matmul.allow_tf32 = True
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    frozen_bytes = args.frozen_manifest.read_bytes()
    frozen = json.loads(frozen_bytes)
    frozen_sha256 = hashlib.sha256(frozen_bytes).hexdigest()
    if frozen["plan_sha256"] != plan_sha256 or frozen["data_sha256"] != plan["data_sha256"]:
        raise ValueError("frozen run manifest disagrees with plan or data")
    if frozen["source_export_sha256"] != plan["source_export_sha256"] or \
       frozen["split_audit_sha256"] != plan["split_audit_sha256"] or \
       frozen["test_case_ids_sha256"] != plan["test_case_ids_sha256"]:
        raise ValueError("frozen source or split differs from plan")
    if frozen["test_tokenized_sha256"] != plan["test_tokenized_sha256"]:
        raise ValueError("frozen test record differs from plan")
    if plan["training_case_count"] != 545 or plan["evaluation_case_count"] != 74 or \
       plan["test_case_count"] != 112 or plan["max_sequence_length"] != 32768:
        raise ValueError("unexpected v2.2 training policy")
    if frozen["epoch_number"] not in (2, 3) or \
       plan["epoch_number"] != frozen["epoch_number"]:
        raise ValueError("unexpected continuation epoch")
    if frozen["training_script_sha256"] != digest(Path(__file__)):
        raise ValueError("running script differs from frozen version")
    if frozen["software"]["torch"] != torch.__version__ or \
       frozen["software"]["transformers"] != transformers.__version__ or \
       frozen["software"]["peft"] != peft.__version__:
        raise ValueError("software version differs from frozen run manifest")
    pairs, eval_ids = plan["train_pairs"], plan["eval_case_ids"]
    trained = [cid for pair in pairs for cid in pair if cid is not None]
    if len(pairs) != plan["optimizer_steps"] or \
       len(trained) != plan["training_case_count"] or \
       len(set(trained)) != plan["training_case_count"] or \
       len(eval_ids) != plan["evaluation_case_count"]:
        raise ValueError("unexpected training/evaluation plan size")
    if set(trained) & set(eval_ids) or set(trained) | set(eval_ids) != set(plan["case_token_lengths"]):
        raise ValueError("train/evaluation overlap or incomplete coverage")
    if pairs[-1] != [plan["shadow_case_id"], None]:
        raise ValueError("invalid final singleton step")
    if rank == 0 and digest(args.data) != plan["data_sha256"]:
        raise ValueError("input data SHA-256 differs from validated tokenizer output")
    if rank == 0:
        for name, expected in frozen["base_files_sha256"].items():
            if digest(args.model / name) != expected:
                raise ValueError(f"base model file changed: {name}")
    dist.barrier(device_ids=[local_rank])
    data = read_needed(args.data, set(trained) | set(eval_ids))
    for cid, row in data.items():
        if len(row["input_ids"]) != plan["case_token_lengths"][cid]:
            raise ValueError(f"changed token count for {cid}")
        if len(row["input_ids"]) > plan["max_sequence_length"]:
            raise ValueError(f"case exceeds context: {cid}")
    config = AutoConfig.from_pretrained(str(args.model), local_files_only=True).text_config
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier(device_ids=[local_rank])
    resume_adapter = None
    resume_state = None
    init_state = None
    if args.resume_from is not None:
        resume_adapter, resume_state = extract_resume(args.resume_from,
                                                      args.output_dir / f"rank_{rank}")
        if resume_state["plan_sha256"] != plan_sha256 or \
           resume_state["data_sha256"] != plan["data_sha256"] or \
           resume_state["frozen_sha256"] != frozen_sha256:
            raise ValueError("resume state does not match this frozen run")
    elif args.init_from is not None:
        if digest(args.init_from) != frozen["previous_epoch"]["resume_sha256"]:
            raise ValueError("previous epoch resume archive changed")
        resume_adapter, init_state = extract_resume(args.init_from,
                                                   args.output_dir / f"rank_{rank}")
        if init_state["data_sha256"] != plan["data_sha256"] or \
           init_state["frozen_sha256"] != frozen["previous_epoch"]["frozen_sha256"] or \
           init_state["plan_sha256"] != frozen["previous_epoch"]["plan_sha256"] or \
           init_state["completed_step"] != 273 or \
           init_state["next_pair_index"] != 273 or \
           len(init_state["step_records"]) != 273 or \
           init_state["evaluation_history"][-1]["step"] != 273:
            raise ValueError("previous epoch is not a complete compatible run")
    elif any(args.output_dir.iterdir()):
        raise ValueError("fresh run output directory is not empty")
    base, info = Qwen3_5ForCausalLM.from_pretrained(str(args.model), config=config,
        dtype=torch.bfloat16, device_map={"": local_rank}, low_cpu_mem_usage=True,
        local_files_only=True, output_loading_info=True, attn_implementation="sdpa")
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise ValueError("base checkpoint did not map cleanly to text backbone")
    base.config.use_cache = False
    if resume_adapter is None:
        adapter = get_peft_model(base, LoraConfig(r=plan["lora_rank"],
            lora_alpha=plan["lora_alpha"], target_modules="all-linear", lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM"))
    else:
        adapter = PeftModel.from_pretrained(base, str(resume_adapter), is_trainable=True)
    adapter.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    adapter.enable_input_require_grads()
    adapter.train()
    wrapped = DistributedDataParallel(adapter, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=False, find_unused_parameters=False, gradient_as_bucket_view=True)
    trainable = [p for p in wrapped.parameters() if p.requires_grad]
    trainable_names = [name for name, p in wrapped.named_parameters() if p.requires_grad]
    if sum(p.numel() for p in trainable) != 43_278_336:
        raise ValueError("unexpected LoRA parameter count")
    optimizer = torch.optim.AdamW(trainable, lr=plan["base_learning_rate"],
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    scheduler_config = {"kind": "linear_warmup_cosine_decay", "total_steps": len(pairs),
                        "warmup_steps": plan["warmup_steps"],
                        "peak_learning_rate": plan["base_learning_rate"],
                        "minimum_learning_rate": plan["base_learning_rate"] * 0.1}
    if resume_state is None:
        if init_state is not None:
            if init_state["trainable_names"] != trainable_names:
                raise ValueError("optimizer parameter order changed across epochs")
            optimizer.load_state_dict(init_state["optimizer"])
            local_rng = init_state["rng_states"][rank]
            torch.set_rng_state(local_rng["torch_cpu"])
            torch.cuda.set_rng_state(local_rng["torch_cuda"], device)
            random.setstate(local_rng["python"])
            np.random.set_state(local_rng["numpy"])
        initial_eval = evaluate(wrapped.module, data, eval_ids, rank, device)
        if init_state is not None and abs(initial_eval - init_state["evaluation_history"][-1]["loss"]) > 0.01:
            raise ValueError("loaded adapter validation disagrees with previous epoch end")
        evaluation_history = []
        best_loss = float("inf")
        best_checkpoint = None
        step_records = []
        start_index = 0
        resume_checkpoints = ([{"source_epoch": frozen["epoch_number"] - 1,
            "archive": str(args.init_from),
            "sha256": frozen["previous_epoch"]["resume_sha256"]}]
            if init_state is not None else [])
    else:
        if resume_state["trainable_names"] != trainable_names or \
           resume_state["scheduler"] != scheduler_config:
            raise ValueError("resume optimizer parameter order or schedule changed")
        optimizer.load_state_dict(resume_state["optimizer"])
        start_index = resume_state["next_pair_index"]
        initial_eval = resume_state["initial_eval"]
        evaluation_history = resume_state["evaluation_history"]
        best_loss = resume_state["best_loss"]
        best_checkpoint = resume_state["best_checkpoint"]
        step_records = resume_state["step_records"]
        resume_checkpoints = list(resume_state["prior_resume_checkpoints"])
        resume_checkpoints.append({"step": resume_state["completed_step"],
            "archive": str(args.resume_from), "sha256": digest(args.resume_from),
            "manifest": json.loads((resume_adapter.parent / "manifest.json").read_text())})
        if start_index != len(step_records) or not 0 < start_index < len(pairs):
            raise ValueError("resume data position is invalid")
        local_rng = resume_state["rng_states"][rank]
        torch.set_rng_state(local_rng["torch_cpu"])
        torch.cuda.set_rng_state(local_rng["torch_cuda"], device)
        random.setstate(local_rng["python"])
        np.random.set_state(local_rng["numpy"])
    args.persistent_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0 and resume_state is None:
        shutil.copyfile(args.frozen_manifest, args.persistent_dir /
                        f"sft_v2_2_epoch{frozen['epoch_number']}_frozen_manifest.json")
    log_file = args.output_dir / "steps.jsonl"
    torch.cuda.reset_peak_memory_stats()
    milestones = {math.ceil(len(pairs) * 0.25), math.ceil(len(pairs) * 0.5), len(pairs)}
    for index in range(start_index, len(pairs)):
        pair = pairs[index]
        step = index + 1
        active = pair[rank] is not None
        cid = pair[rank] if active else plan["shadow_case_id"]
        row = data[cid]
        batch = tensors(row, device, 0, base.config.eos_token_id)
        local_count = len(batch[3]) if active else 0
        global_count = torch.tensor(float(local_count), device=device)
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        lr = lr_for_step(index, len(pairs), plan["warmup_steps"], plan["base_learning_rate"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        dist.barrier(device_ids=[local_rank])
        torch.cuda.synchronize()
        step_began = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        local_sum, _ = sparse_sum_loss(wrapped, batch)
        objective_sum = local_sum if active else local_sum * 0.0
        (objective_sum * world / global_count).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, plan["max_grad_norm"])
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        if step == 1 and float(gradient_norm) <= 0:
            raise FloatingPointError("no LoRA gradient on first optimizer step")
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.perf_counter() - step_began, dtype=torch.float64, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        loss_sum = objective_sum.detach().double()
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        useful_tokens = torch.tensor(float(len(row["input_ids"]) if active else 0),
                                     dtype=torch.float64, device=device)
        dist.all_reduce(useful_tokens, op=dist.ReduceOp.SUM)
        if rank == 0:
            record = {"step": step, "pair": pair, "loss": round(float(loss_sum / global_count), 6),
                      "lr": lr, "supervised_tokens": int(global_count.item()),
                      "useful_tokens": int(useful_tokens.item()),
                      "seconds": round(float(elapsed.item()), 3)}
            step_records.append(record)
            with log_file.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if step == 1 or step % 10 == 0 or step == len(pairs):
                print(json.dumps(record), flush=True)
        if step % plan["eval_every_steps"] == 0 or step in milestones:
            eval_loss = evaluate(wrapped.module, data, eval_ids, rank, device)
            if rank == 0:
                print(json.dumps({"evaluation_step": step, "eval_loss": eval_loss,
                                  "initial_eval_loss": initial_eval}), flush=True)
                evaluation_history.append({"step": step, "loss": eval_loss})
                if eval_loss < best_loss:
                    checkpoint = save_adapter(wrapped.module, args.output_dir,
                        args.persistent_dir, step=step, eval_loss=eval_loss,
                        plan_sha256=plan_sha256,
                        data_sha256=plan["data_sha256"])
                    best_checkpoint = checkpoint
                    best_loss = eval_loss
            dist.barrier(device_ids=[local_rank])
            if step in milestones:
                local_rng = {"torch_cpu": torch.get_rng_state(),
                             "torch_cuda": torch.cuda.get_rng_state(device),
                             "python": random.getstate(), "numpy": np.random.get_state()}
                rng_states = [None] * world
                dist.all_gather_object(rng_states, local_rng)
                local_memory = {"rank": rank,
                                "peak_allocated_gb": round(torch.cuda.max_memory_allocated()/1e9, 2),
                                "peak_reserved_gb": round(torch.cuda.max_memory_reserved()/1e9, 2)}
                memory = [None] * world
                dist.all_gather_object(memory, local_memory)
                if rank == 0:
                    saved = save_training_state(wrapped.module, optimizer, trainable_names,
                        args.output_dir, args.persistent_dir, step=step,
                        rng_states=rng_states, step_records=step_records,
                        evaluation_history=evaluation_history, initial_eval=initial_eval,
                        best_loss=best_loss, best_checkpoint=best_checkpoint,
                        plan_sha256=plan_sha256, data_sha256=plan["data_sha256"],
                        frozen_sha256=frozen_sha256, scheduler_config=scheduler_config,
                        gpu_memory=memory, prior_resume_checkpoints=resume_checkpoints)
                    resume_checkpoints.append(saved)
                    print(json.dumps({"resume_checkpoint_step": step,
                                      "archive": saved["archive"],
                                      "sha256": saved["sha256"]}), flush=True)
                dist.barrier(device_ids=[local_rank])
    if rank == 0 and best_checkpoint is None:
        raise RuntimeError("no persistent checkpoint")
    sample = next(p for p in trainable if p.numel() > 1024).detach().flatten().float()[:4096]
    gathered = [torch.empty_like(sample) for _ in range(world)]
    dist.all_gather(gathered, sample)
    max_rank_diff = float((gathered[0] - gathered[1]).abs().max().item())
    if max_rank_diff != 0.0:
        raise ValueError("DDP ranks diverged")
    stats = {"rank": rank, "peak_allocated_gb": round(torch.cuda.max_memory_allocated()/1e9, 2),
             "peak_reserved_gb": round(torch.cuda.max_memory_reserved()/1e9, 2)}
    all_stats = [None] * world
    dist.all_gather_object(all_stats, stats)
    dist.destroy_process_group()
    if rank != 0:
        return
    del wrapped, adapter, base, optimizer, trainable
    gc.collect()
    torch.cuda.empty_cache()
    base_reload = Qwen3_5ForCausalLM.from_pretrained(str(args.model), config=config,
        dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True,
        attn_implementation="sdpa")
    base_reload.config.use_cache = False
    restored = PeftModel.from_pretrained(base_reload,
                                        str(resolve_best_adapter(best_checkpoint, args.output_dir)),
                                        is_trainable=False)
    restored.eval()
    loss_total = 0.0
    target_total = 0
    with torch.no_grad():
        for cid in eval_ids:
            subtotal, count = sparse_sum_loss(restored, tensors(data[cid], device))
            loss_total += float(subtotal.item())
            target_total += count
    reloaded_eval = loss_total / target_total
    if abs(reloaded_eval - best_loss) > 0.01:
        raise ValueError(f"reloaded adapter validation loss changed: {best_loss} -> {reloaded_eval}")
    summary = {
        "schema_version": "nautil.sft.training_summary.v1",
        "status": "completed", "model": "Qwen/Qwen3.5-9B", "epochs": 1,
        "epoch_number": frozen["epoch_number"],
        "continued_from": frozen["previous_epoch"],
        "training_cases": plan["training_case_count"],
        "evaluation_cases": plan["evaluation_case_count"],
        "optimizer_steps": len(pairs),
        "useful_tokens_seen": sum(item["useful_tokens"] for item in step_records),
        "supervised_tokens_seen": sum(item["supervised_tokens"] for item in step_records),
        "initial_eval_loss": initial_eval, "best_eval_loss": best_loss,
        "best_step": best_checkpoint["step"], "best_adapter": best_checkpoint,
        "reloaded_eval_loss": reloaded_eval,
        "adapter_max_rank_difference": max_rank_diff, "gpu_memory": all_stats,
        "training_step_seconds": round(sum(item["seconds"] for item in step_records), 2),
        "elapsed_seconds": round(time.perf_counter() - began, 2),
        "data_sha256": plan["data_sha256"],
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "evaluation_history": evaluation_history,
        "persistent_resume_checkpoints": resume_checkpoints,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    shutil.copyfile(summary_path, args.persistent_dir /
                    f"sft_v2_2_epoch{frozen['epoch_number']}_summary.json")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
