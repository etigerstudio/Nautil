#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig, Qwen3_5ForCausalLM


def read_needed(path: Path, needed: set[str]) -> dict[str, dict]:
    result = {}
    with path.open() as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                if row["case_id"] in needed:
                    result[row["case_id"]] = row
    if set(result) != needed:
        raise ValueError(f"missing {len(needed-set(result))} planned cases")
    return result


def tensors(row: dict, device: torch.device, bucket_size: int = 0, pad_id: int = 248044):
    extra = (-len(row["input_ids"])) % bucket_size if bucket_size else 0
    ids = torch.tensor([row["input_ids"] + [pad_id] * extra], dtype=torch.long, device=device)
    labels = torch.tensor([row["labels"] + [-100] * extra], dtype=torch.long, device=device)
    mask = torch.tensor([row["attention_mask"] + [0] * extra], dtype=torch.long, device=device)
    positions = torch.nonzero(labels[0, 1:] != -100).flatten()
    if not len(positions):
        raise ValueError(f"no assistant targets for {row['case_id']}")
    return ids, labels, mask, positions


def sparse_sum_loss(model, batch) -> tuple[torch.Tensor, int]:
    ids, labels, mask, positions = batch
    logits = model(input_ids=ids, attention_mask=mask,
                   logits_to_keep=positions, use_cache=False).logits
    targets = labels[:, positions + 1]
    loss_sum = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                               targets.reshape(-1), reduction="sum")
    return loss_sum, len(positions)


def evaluate(model, data: dict[str, dict], eval_ids: list[str], rank: int,
             device: torch.device) -> float:
    model.eval()
    total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_targets = torch.tensor(0.0, dtype=torch.float64, device=device)
    with torch.no_grad():
        for cid in eval_ids[rank::dist.get_world_size()]:
            loss_sum, count = sparse_sum_loss(model, tensors(data[cid], device))
            total_loss += loss_sum.detach().double()
            total_targets += count
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_targets, op=dist.ReduceOp.SUM)
    model.train()
    return float((total_loss / total_targets).item())


def single_eval(model, row: dict, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        loss_sum, count = sparse_sum_loss(model, tensors(row, device))
    return float((loss_sum / count).item())


def set_lr(optimizer, step: int, total: int, base: float) -> float:
    if step < 2:
        lr = base * (step+1)/2
    else:
        progress = (step-2) / max(1,total-3)
        lr = base * (0.1 + 0.9*0.5*(1+math.cos(math.pi*progress)))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bucket-size", type=int, default=0)
    args = ap.parse_args()
    canary_started = time.perf_counter()
    if args.bucket_size not in {0, 1024, 2048, 4096}:
        raise ValueError("bucket-size must be 0, 1024, 2048, or 4096")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2:
        raise ValueError("canary requires exactly two GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(20260925)
    torch.cuda.manual_seed_all(20260925)
    torch.backends.cuda.matmul.allow_tf32 = True
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    plan = json.loads(args.plan.read_text())
    if rank == 0:
        digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
        if digest != plan["data_sha256"]:
            raise ValueError("training data differs from tokenizer-validated file")
    dist.barrier()
    eval_ids = plan["eval_case_ids"]
    pairs = plan["train_pairs"]
    if len(pairs) != 20 or set(eval_ids) & {cid for pair in pairs for cid in pair}:
        raise ValueError("canary plan overlap or wrong step count")
    needed = set(eval_ids) | {cid for pair in pairs for cid in pair}
    data = read_needed(args.data, needed)
    for cid, row in data.items():
        if len(row["input_ids"]) != plan["case_token_lengths"][cid]:
            raise ValueError(f"token length changed for {cid}")
    config = AutoConfig.from_pretrained(str(args.model), local_files_only=True).text_config
    base, info = Qwen3_5ForCausalLM.from_pretrained(str(args.model), config=config,
        dtype=torch.bfloat16, device_map={"": local_rank}, low_cpu_mem_usage=True,
        local_files_only=True, output_loading_info=True, attn_implementation="sdpa")
    if any(info.get(key) for key in ["missing_keys", "unexpected_keys", "mismatched_keys"]):
        raise ValueError("checkpoint mapping incomplete")
    base.config.use_cache = False
    adapter = get_peft_model(base, LoraConfig(r=16,lora_alpha=32,
        target_modules="all-linear",lora_dropout=0.0,bias="none",task_type="CAUSAL_LM"))
    adapter.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    adapter.enable_input_require_grads()
    adapter.train()
    wrapped = DistributedDataParallel(adapter,device_ids=[local_rank],output_device=local_rank,
        broadcast_buffers=False,find_unused_parameters=False,gradient_as_bucket_view=True)
    trainable = [p for p in wrapped.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable,lr=plan["learning_rate"])
    initial_eval = evaluate(wrapped.module,data,eval_ids,rank,device)
    torch.cuda.reset_peak_memory_stats()
    step_logs = []
    total_input_tokens = 0
    total_useful_tokens = 0
    total_supervised_tokens = 0
    grad_nonzero = False
    for step,pair in enumerate(pairs):
        row = data[pair[rank]]
        batch = tensors(row,device,args.bucket_size,base.config.eos_token_id)
        local_count = len(batch[3])
        global_count = torch.tensor(float(local_count),dtype=torch.float32,device=device)
        dist.all_reduce(global_count,op=dist.ReduceOp.SUM)
        lr = set_lr(optimizer,step,len(pairs),plan["learning_rate"])
        dist.barrier()
        torch.cuda.synchronize()
        began = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        local_sum,_ = sparse_sum_loss(wrapped,batch)
        scaled = local_sum * world / global_count
        scaled.backward()
        if step == 0:
            grad_nonzero = any(p.grad is not None and bool(torch.count_nonzero(p.grad).item())
                               for p in trainable[:6])
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.perf_counter()-began,dtype=torch.float64,device=device)
        dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
        global_loss_sum = local_sum.detach().double()
        dist.all_reduce(global_loss_sum,op=dist.ReduceOp.SUM)
        pair_tokens = torch.tensor(float(batch[0].shape[1]),dtype=torch.float64,device=device)
        dist.all_reduce(pair_tokens,op=dist.ReduceOp.SUM)
        useful_tokens = torch.tensor(float(len(row["input_ids"])),dtype=torch.float64,device=device)
        dist.all_reduce(useful_tokens,op=dist.ReduceOp.SUM)
        if rank == 0:
            total_input_tokens += int(pair_tokens.item())
            total_useful_tokens += int(useful_tokens.item())
            total_supervised_tokens += int(global_count.item())
            log={"step":step+1,"pair":pair,"processed_tokens":int(pair_tokens.item()),
                 "useful_tokens":int(useful_tokens.item()),
                 "supervised_tokens":int(global_count.item()),"loss":round(float(global_loss_sum/global_count),6),
                 "lr":lr,"step_seconds":round(float(elapsed.item()),3),
                 "processed_tokens_per_second":round(float(pair_tokens/elapsed),1),
                 "useful_tokens_per_second":round(float(useful_tokens/elapsed),1)}
            step_logs.append(log)
            print(json.dumps(log,ensure_ascii=False),flush=True)
    final_eval = evaluate(wrapped.module,data,eval_ids,rank,device)
    sample=next(p for p in trainable if p.numel() > 1024).detach().flatten().float()[:4096]
    gathered=[torch.empty_like(sample) for _ in range(world)]
    dist.all_gather(gathered,sample)
    adapter_max_rank_diff=float((gathered[0]-gathered[1]).abs().max().item())
    local_stats={"rank":rank,"peak_allocated_gb":round(torch.cuda.max_memory_allocated()/1e9,2),
                 "peak_reserved_gb":round(torch.cuda.max_memory_reserved()/1e9,2),
                 "gradient_nonzero":grad_nonzero}
    stats=[None]*world
    dist.all_gather_object(stats,local_stats)
    probe_case=eval_ids[0]
    before_reload=single_eval(wrapped.module,data[probe_case],device) if rank==0 else None
    if rank==0:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        wrapped.module.save_pretrained(str(args.output_dir/'adapter'),safe_serialization=True)
    dist.barrier()
    dist.destroy_process_group()
    if rank!=0:
        return
    del wrapped,adapter,base,optimizer,trainable
    gc.collect()
    torch.cuda.empty_cache()
    base_reload=Qwen3_5ForCausalLM.from_pretrained(str(args.model),config=config,
        dtype=torch.bfloat16,device_map={"":0},local_files_only=True,attn_implementation="sdpa")
    base_reload.config.use_cache=False
    restored=PeftModel.from_pretrained(base_reload,str(args.output_dir/'adapter'),is_trainable=False)
    after_reload=single_eval(restored,data[probe_case],device)
    result={"steps":len(pairs),"train_cases_seen":40,"eval_cases":len(eval_ids),
            "seed":20260925,
            "bucket_size":args.bucket_size,
            "initial_eval_loss":initial_eval,"final_eval_loss":final_eval,
            "eval_loss_change":final_eval-initial_eval,
            "processed_tokens_seen":total_input_tokens,"useful_tokens_seen":total_useful_tokens,
            "supervised_tokens_seen":total_supervised_tokens,
            "train_processed_tokens_per_second":round(total_input_tokens/sum(x["step_seconds"] for x in step_logs),1),
            "train_useful_tokens_per_second":round(total_useful_tokens/sum(x["step_seconds"] for x in step_logs),1),
            "adapter_max_rank_diff":adapter_max_rank_diff,
            "adapter_reload_probe_loss_before":before_reload,
            "adapter_reload_probe_loss_after":after_reload,
            "adapter_reload_loss_difference":abs(before_reload-after_reload),
            "training_step_wall_seconds":round(sum(x["step_seconds"] for x in step_logs),2),
            "full_canary_wall_seconds":round(time.perf_counter()-canary_started,2),
            "per_rank":stats,"steps_log":step_logs,
            "adapter_path":str(args.output_dir/'adapter'),
            "training_complete":False,"purpose":"bounded canary only"}
    (args.output_dir/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ["steps","bucket_size","seed","initial_eval_loss","final_eval_loss",
        "processed_tokens_seen","useful_tokens_seen","supervised_tokens_seen",
        "train_processed_tokens_per_second","train_useful_tokens_per_second",
        "training_step_wall_seconds","full_canary_wall_seconds",
        "adapter_max_rank_diff","adapter_reload_loss_difference","per_rank"]},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
