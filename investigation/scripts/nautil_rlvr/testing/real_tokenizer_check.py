from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path

from nautil_harness_v2_2 import run_case

from nautil_rlvr.common import read_jsonl, sha256_ids
from nautil_rlvr.config import load as load_config
from nautil_rlvr.reward import RewardConfig, judge_requests
from nautil_rlvr.rollout import render_prompt_ids
from nautil_rlvr.sequences import build_sequence, compare_with_sft_mask

HEAD = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def teacher_bodies(tokenizer, system, row) -> list[str]:
    msgs = [{"role": "system", "content": system}] + [{k: v for k, v in m.items() if k != "loss"} for m in row["messages"]]
    text = tokenizer.apply_chat_template(msgs, tools=row["tools"], tokenize=False, add_generation_prompt=False)
    return [b.split("<|im_end|>", 1)[0] for b in text.split(HEAD)[1:]]


def replay(tokenizer, end_id, system, case, bodies, protocol) -> dict:
    it = iter(bodies)
    turn_rl = []
    captured = {}

    def gen(model, tok, messages, tools, device, limit):
        captured["m"] = messages
        prompt = render_prompt_ids(tokenizer, messages, tools)
        text = next(it)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"] + [end_id]
        turn_rl.append({"prompt_len": len(prompt), "prompt_sha": sha256_ids(prompt), "n_messages": len(messages),
                        "gen_ids": ids, "gen_logprobs": [0.0] * len(ids)})
        return text, len(prompt), False
    result = run_case(None, tokenizer, system, case, None, protocol["context_limit"], protocol["max_turns"],
                      generate_fn=gen)
    return {"case_id": case["case_id"], "source": case["source"], "result": result,
            "messages": captured["m"], "turn_rl": turn_rl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    cfg = load_config(args.config)
    paths = cfg["paths"]
    tok = AutoTokenizer.from_pretrained(paths["tokenizer"], local_files_only=True)
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    system = Path(paths["system_prompt"]).read_text().strip()
    protocol = json.loads(Path(paths["rollout_protocol"]).read_text())
    cases = {c["case_id"]: c for c in read_jsonl(paths["train_bundle"])}
    refs = {r["case_id"]: r for r in read_jsonl(paths["train_refs"])}
    teachers = {r["case_id"]: r for r in read_jsonl(paths["train_teacher"])}
    room = protocol["context_limit"] - protocol["max_tokens_per_turn"]
    opening = {}
    for cid, case in cases.items():
        opening[cid] = len(render_prompt_ids(tok, [{"role": "system", "content": system},
                                                   {"role": "user", "content": case["user_message"]}], case["tools"]))
    too_long = sorted(c for c, n in opening.items() if n >= room)
    pick = random.Random(0).sample(sorted(cases), args.cases)
    rows, sizes = [], {"fetch": [], "hypothesis": [], "closure": []}
    rcfg = RewardConfig.from_dict({"phase": 2})
    for cid in pick:
        bodies = teacher_bodies(tok, system, teachers[cid])
        traj = replay(tok, end_id, system, cases[cid], bodies, protocol)
        seq = build_sequence(traj, tok, system, cases[cid])
        cmp_ = compare_with_sft_mask(traj, tok, system, cases[cid], seq)
        for dim, (text, n) in judge_requests(traj["result"], cases[cid], refs[cid], rcfg).items():
            sizes[dim].append(len(tok(text, add_special_tokens=False)["input_ids"]))
        rows.append({"case_id": cid, "status": traj["result"]["status"], "turns": len(traj["turn_rl"]),
                     "turn_status": seq["turn_status"], "tokens": len(seq["input_ids"]),
                     "target_tokens": seq["target_tokens"], **{k: cmp_[k] for k in cmp_ if k != "turn_status"}})
    ok_turns = sum(r["turn_status"].count("ok") for r in rows)
    all_turns = sum(len(r["turn_status"]) for r in rows)
    judge_prompt_overhead = {d: len(tok((Path(__file__).resolve().parents[1] / "prompts" / f).read_text(),
                                        add_special_tokens=False)["input_ids"])
                             for d, f in (("fetch", "fetch_judge_v1.txt"), ("hypothesis", "hypothesis_judge_v1.txt"),
                                          ("closure", "closure_judge_v1.txt"))}
    out = {"cases_checked": len(rows), "turns_token_exact": ok_turns, "turns_total": all_turns,
           "all_same_token_ids_as_sft": all(r.get("same_token_ids") for r in rows),
           "all_rl_mask_subset_of_sft": all(r.get("rl_subset_of_sft") for r in rows),
           "sft_only_tokens_not_think_prefix": sum(r.get("sft_only_not_think_prefix", 0) for r in rows),
           "sft_analyze_errors": sum(len(r.get("sft_analyze_errors") or []) for r in rows),
           "opening_tokens": {"min": min(opening.values()), "median": statistics.median(opening.values()),
                              "max": max(opening.values())},
           "excluded_opening_too_long": too_long,
           "judge_user_tokens": {d: {"mean": statistics.mean(v) if v else None, "max": max(v) if v else None,
                                     "calls_per_traj": len(v) / len(rows)} for d, v in sizes.items()},
           "judge_system_prompt_tokens": judge_prompt_overhead,
           "sequence_tokens_mean": statistics.mean(r["tokens"] for r in rows),
           "target_tokens_mean": statistics.mean(r["target_tokens"] for r in rows),
           "per_case": rows}
    args.output.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_case"}, indent=1))


if __name__ == "__main__":
    main()
