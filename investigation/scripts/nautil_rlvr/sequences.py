from __future__ import annotations

from .common import sha256_ids
from .rollout import render_prompt_ids


def build_sequence(traj: dict, tokenizer, system: str, case: dict) -> dict | None:
    turns = traj["turn_rl"]
    messages = traj["messages"]
    if not turns or messages is None:
        return None
    last = max((i for i, t in enumerate(turns) if t.get("gen_ids")), default=None)
    if last is None:
        return None
    info_last = turns[last]
    prompt = render_prompt_ids(tokenizer, messages[:info_last["n_messages"]], case["tools"])
    if len(prompt) != info_last["prompt_len"] or sha256_ids(prompt) != info_last["prompt_sha"]:
        return {"error": "last prompt does not reproduce", "turn_status": []}
    seq = prompt + list(info_last["gen_ids"])
    mask = [0] * len(seq)
    behaviour = [0.0] * len(seq)
    status = []
    for index, info in enumerate(turns):
        ids = info.get("gen_ids") or []
        if not ids:
            status.append("no_tokens")
            continue
        start = info["prompt_len"]
        end = start + len(ids)
        if index < last:
            if end > len(seq) or sha256_ids(seq[:start]) != info["prompt_sha"]:
                status.append("mismatch_prefix")
                continue
            if seq[start:end] != list(ids):
                status.append("mismatch_tokens")
                continue
        lps = info.get("gen_logprobs")
        if lps is None or any(v is None for v in lps):
            status.append("no_logprobs")
            continue
        mask[start:end] = [1] * len(ids)
        behaviour[start:end] = [float(v) for v in lps]
        status.append("ok")
    return {"input_ids": seq, "loss_mask": mask, "behaviour_logprobs": behaviour,
            "turn_status": status, "target_tokens": sum(mask),
            "turns_ok": status.count("ok"),
            "turns_mismatch": sum(s.startswith("mismatch") for s in status)}


def compare_with_sft_mask(traj: dict, tokenizer, system: str, case: dict, seq: dict) -> dict:
    from verify_qwen35_wire_cpu import analyze
    result = traj["result"]
    if result.get("final_answer") is None or traj["messages"] is None:
        return {"checked": False}
    row = {"case_id": case["case_id"], "source": case["source"], "tools": case["tools"],
           "messages": [{**m, "loss": m["role"] == "assistant"} for m in traj["messages"][1:]]}
    metrics, record = analyze(row, tokenizer, system, return_tokens=True)
    sft_ids = record["input_ids"]
    sft_mask = [int(x != -100) for x in record["labels"]]
    rl_ids, rl_mask = seq["input_ids"], seq["loss_mask"]
    same_prefix = sft_ids[:len(rl_ids)] == rl_ids
    subset = all(not m or (i < len(sft_mask) and sft_mask[i]) for i, m in enumerate(rl_mask))
    only_sft = [i for i, m in enumerate(sft_mask) if m and not (i < len(rl_mask) and rl_mask[i])]
    think = tokenizer("<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"]
    think_positions = set()
    for i in range(len(sft_ids) - len(think) + 1):
        if sft_ids[i:i + len(think)] == think:
            think_positions.update(range(i, i + len(think)))
    extra = [i for i in only_sft if i not in think_positions]
    errors = [e for e in metrics["errors"] if "content missing from render" not in e]
    return {"checked": True, "same_token_ids": same_prefix and len(sft_ids) - len(rl_ids) in (0, 1),
            "rl_subset_of_sft": subset, "sft_only_tokens": len(only_sft),
            "sft_only_not_think_prefix": len(extra), "sft_analyze_errors": errors,
            "turn_status": seq["turn_status"]}
