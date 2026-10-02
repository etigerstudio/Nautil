#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import time
import traceback
from pathlib import Path


CALL = re.compile(r"<tool_call>\s*<function=([A-Za-z_][A-Za-z_0-9]*)>\s*(.*?)\s*</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=([A-Za-z_][A-Za-z_0-9]*)>\s*(.*?)\s*</parameter>", re.S)
EVIDENCE_ID = re.compile(r"\bE[1-9][0-9]*\.[1-9][0-9]*\b")


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_call(content: str) -> tuple[str, dict | None, list[str]]:
    errors = []
    matches = list(CALL.finditer(content))
    if not matches:
        if "<tool_call>" in content or "<function=" in content:
            errors.append("malformed tool-call markup")
        return content, None, errors
    if len(matches) != 1:
        errors.append("multiple tool calls in one assistant turn")
    match = matches[0]
    if match.group(1) != "request_evidence":
        errors.append(f"unknown tool: {match.group(1)}")
    if content[match.end():].strip():
        errors.append("text after tool call")
    if errors:
        return content, None, errors
    params = PARAM.findall(match.group(2))
    arguments = {name: value.strip() for name, value in params}
    if len(params) != len(arguments):
        errors.append("duplicate parameter")
    try:
        ids = json.loads(arguments.get("evidence_ids", "null"))
    except json.JSONDecodeError:
        ids = None
    reason = arguments.get("reason", "")
    if not isinstance(ids, list) or not 1 <= len(ids) <= 12 or \
       not all(isinstance(cid, str) and EVIDENCE_ID.fullmatch(cid) for cid in ids):
        errors.append("invalid evidence_ids argument")
        ids = []
    if len(ids) != len(set(ids)):
        errors.append("duplicate ID within one call")
    if not reason or len(reason) < 10:
        errors.append("missing or too-short investigation reason")
    return content[:match.start()].rstrip(), {"evidence_ids": ids, "reason": reason}, errors


def generate_turn(model, tokenizer, messages: list[dict], tools: list[dict],
                  device: torch.device, context_limit: int) -> tuple[str, int, bool]:
    import torch
    rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    prompt = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    prompt_tokens = prompt.shape[1]
    if prompt_tokens >= context_limit:
        return "", prompt_tokens, True
    generated = prompt
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(end_id, int) or end_id < 0:
        raise ValueError("Qwen assistant end token unavailable")
    while generated.shape[1] < context_limit:
        remaining = context_limit - generated.shape[1]
        with torch.inference_mode():
            output = model.generate(input_ids=generated,
                max_new_tokens=min(8192, remaining), do_sample=False,
                eos_token_id=end_id, pad_token_id=end_id, use_cache=True)
        if output.shape[1] <= generated.shape[1]:
            raise RuntimeError("generation produced no new tokens")
        generated = output
        if generated[0, -1].item() == end_id:
            body_ids = generated[0, prompt_tokens:-1]
            return tokenizer.decode(body_ids, skip_special_tokens=False), prompt_tokens, False
    body_ids = generated[0, prompt_tokens:]
    return tokenizer.decode(body_ids, skip_special_tokens=False), prompt_tokens, True


def run_case(model, tokenizer, system: str, case: dict, device: torch.device,
             context_limit: int, max_turns: int, generate_fn=None) -> dict:
    if generate_fn is None:
        generate_fn = generate_turn
    store = {item["evidence_id"]: item for item in case["evidence_items"]}
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": case["user_message"]}]
    initial_record = case["user_message"].split("## Index of the remaining", 1)[0]
    disclosed = set(EVIDENCE_ID.findall(initial_record))
    fetches, errors, turns = [], [], []
    began = time.perf_counter()
    final = None
    status = "turn_limit"
    for turn_number in range(1, max_turns + 1):
        try:
            generation = generate_fn(model, tokenizer, messages, case["tools"],
                                     device, context_limit)
            if len(generation) == 4:
                text, prompt_tokens, context_full, generation_metrics = generation
            else:
                text, prompt_tokens, context_full = generation
                generation_metrics = None
        except Exception as exc:
            status = "runtime_error"
            errors.append(f"turn {turn_number}: {type(exc).__name__}: {exc}")
            turns.append({"turn": turn_number, "runtime_error": traceback.format_exc()})
            break
        if context_full:
            reason = (generation_metrics or {}).get("finish_reason", "context_limit")
            status = reason if reason in {"case_token_budget", "turn_token_budget", "wall_budget"} else "context_limit"
            errors.append(f"{status} reached in turn {turn_number}")
            turns.append({"turn": turn_number, "assistant": text,
                          "prompt_tokens": prompt_tokens, "context_full": True,
                          "generation": generation_metrics})
            break
        prose, arguments, call_errors = parse_call(text)
        errors.extend(f"turn {turn_number}: {error}" for error in call_errors)
        turn = {"turn": turn_number, "assistant": prose if arguments else text,
                "assistant_raw": text,
                "prompt_tokens": prompt_tokens, "call": arguments,
                "call_errors": call_errors, "generation": generation_metrics}
        if arguments is None:
            final = text
            status = "final" if not call_errors else "malformed_final"
            messages.append({"role": "assistant", "content": text})
            turns.append(turn)
            break
        call_id = f"call_{turn_number:03d}"
        messages.append({"role": "assistant", "content": prose,
                         "tool_calls": [{"id": call_id, "type": "function",
                                         "function": {"name": "request_evidence",
                                                      "arguments": arguments}}]})
        requested = arguments["evidence_ids"]
        unknown = [cid for cid in requested if cid not in store]
        if unknown or not requested or call_errors:
            payload = {"error": "invalid request_evidence arguments or unknown IDs",
                       "requested": requested, "unknown": unknown,
                       "schema_errors": call_errors}
            errors.append(f"turn {turn_number}: invalid fetch {requested}")
        else:
            fresh = [cid for cid in requested if cid not in disclosed]
            payload = {"evidence_items": [store[cid] for cid in fresh]}
            disclosed.update(fresh)
        tool_content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        messages.append({"role": "tool", "tool_call_id": call_id,
                         "name": "request_evidence", "content": tool_content})
        turn["tool_return"] = payload
        fetches.append(requested)
        turns.append(turn)
    final_citations = set(EVIDENCE_ID.findall(final or ""))
    if status == "turn_limit":
        errors.append(f"assistant turn budget {max_turns} exhausted without a final answer")
    if final is not None and not final_citations:
        errors.append("final answer contains no evidence citation")
    conclusion_marker = None
    if final and "CASE NOT CLOSED" in final:
        conclusion_marker = "CASE NOT CLOSED"
    elif final and "CASE CLOSED" in final:
        conclusion_marker = "CASE CLOSED"
    if final and conclusion_marker is None:
        errors.append("final answer lacks CASE CLOSED / CASE NOT CLOSED marker")
    return {"case_id": case["case_id"], "source": case["source"],
            "initial_user_brief": case["user_message"],
            "status": status, "turns": turns, "final_answer": final,
            "assistant_turns": len(turns), "turn_budget": max_turns,
            "conclusion_marker": conclusion_marker,
            "fetch_rounds": len(fetches), "requested_evidence_ids": fetches,
            "unique_fetched_evidence_ids": sorted({cid for request in fetches for cid in request}),
            "disclosed_evidence_ids": sorted(disclosed),
            "hypothesis_update_turns": sum("Hypothesis ledger" in item.get("assistant", "") or
                                           "Evidence and hypothesis update" in item.get("assistant", "")
                                           for item in turns),
            "final_cited_evidence_ids": sorted(final_citations),
            "final_citations_all_disclosed":
                bool(final_citations) and final_citations <= disclosed if final else None,
            "errors": errors, "elapsed_seconds": round(time.perf_counter() - began, 2)}


def render_markdown(result: dict) -> str:
    out = [f"# {result['case_id']} — {result['status']}", "",
           f"Fetch rounds: {result['fetch_rounds']}; elapsed: {result['elapsed_seconds']} s", "",
           "## Initial user brief", "", result["initial_user_brief"], ""]
    for turn in result["turns"]:
        out += [f"## Assistant turn {turn['turn']}", "",
                turn.get("assistant", "[generation runtime error]"), ""]
        if turn.get("runtime_error"):
            out += ["```text", turn["runtime_error"], "```", ""]
        if turn.get("call") is not None:
            out += ["### request_evidence", "", "```json",
                    json.dumps(turn["call"], ensure_ascii=False, indent=2), "```", ""]
            out += ["### Tool return", "", "```json",
                    json.dumps(turn["tool_return"], ensure_ascii=False, indent=2), "```", ""]
    if result["errors"]:
        out += ["## Harness findings", "", *result["errors"], ""]
    return "\n".join(out) + "\n"


def main() -> None:
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--adapter", type=Path,
                    help="Omit for the untouched base model; supply PEFT adapter for post-SFT")
    ap.add_argument("--system-prompt", type=Path, required=True)
    ap.add_argument("--bundles", type=Path, required=True)
    ap.add_argument("--bundle-sha256", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--persistent-dir", type=Path, required=True)
    ap.add_argument("--max-turns", type=int, default=128)
    ap.add_argument("--context-limit", type=int, default=32768)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--case-id", help="Optional single held-out case for a rollout smoke test")
    args = ap.parse_args()
    if sha256(args.bundles) != args.bundle_sha256:
        raise ValueError("held-out rollout bundle hash changed")
    cases = [json.loads(line) for line in args.bundles.read_text().splitlines() if line]
    if len(cases) != 8 or len({case["case_id"] for case in cases}) != 8:
        raise ValueError("expected eight distinct held-out cases")
    if args.case_id:
        cases = [case for case in cases if case["case_id"] == args.case_id]
        if len(cases) != 1:
            raise ValueError("smoke-test case is not in the fixed held-out set")
    system = args.system_prompt.read_text().strip()
    torch.manual_seed(20260925)
    torch.cuda.manual_seed_all(20260925)
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    device = torch.device("cuda", args.gpu_index)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    config = AutoConfig.from_pretrained(str(args.model), local_files_only=True).text_config
    base = Qwen3_5ForCausalLM.from_pretrained(str(args.model), config=config,
        dtype=torch.bfloat16, device_map={"": args.gpu_index}, low_cpu_mem_usage=True,
        local_files_only=True, attn_implementation="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    else:
        model = base
    model.eval()
    output_dir = args.output_root / args.tag
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for case in cases:
        result = run_case(model, tokenizer, system, case, device,
                          args.context_limit, args.max_turns)
        torch.cuda.empty_cache()
        results.append(result)
        (output_dir / f"{case['case_id']}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        (output_dir / f"{case['case_id']}.md").write_text(render_markdown(result))
        print(json.dumps({"case_id": result["case_id"], "status": result["status"],
                          "fetch_rounds": result["fetch_rounds"],
                          "final_citations_all_disclosed": result["final_citations_all_disclosed"],
                          "errors": len(result["errors"]),
                          "seconds": result["elapsed_seconds"]}), flush=True)
    summary = {"schema_version": "nautil.sft.live_rollout.v1", "tag": args.tag,
               "model_condition": "post_sft_lora" if args.adapter else "untouched_base",
               "adapter": str(args.adapter) if args.adapter else None,
               "bundle_sha256": args.bundle_sha256,
               "cases": len(results), "final_answers": sum(x["status"] == "final" for x in results),
               "valid_final_citations": sum(x["final_citations_all_disclosed"] is True for x in results),
               "marked_conclusions": sum(x["conclusion_marker"] is not None for x in results),
               "hypothesis_update_turns": sum(x["hypothesis_update_turns"] for x in results),
               "fetch_rounds": sum(x["fetch_rounds"] for x in results),
               "errors": sum(len(x["errors"]) for x in results),
               "per_case": [{k: x[k] for k in ("case_id", "source", "status", "fetch_rounds",
                                                "hypothesis_update_turns", "conclusion_marker",
                                                "final_citations_all_disclosed", "errors")}
                            for x in results]}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    archive = args.output_root / f"{args.tag}.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(output_dir, arcname=args.tag)
    archive_sha256 = sha256(archive)
    args.persistent_dir.mkdir(parents=True, exist_ok=True)
    target = args.persistent_dir / f"rollouts_{args.tag}.tar"
    pending = args.persistent_dir / f".{target.name}.partial"
    shutil.copyfile(archive, pending)
    os.replace(pending, target)
    if sha256(target) != archive_sha256:
        raise ValueError("persistent rollout archive hash changed")
    print(json.dumps({"summary": summary, "persistent_archive": str(target),
                      "archive_sha256": archive_sha256}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
