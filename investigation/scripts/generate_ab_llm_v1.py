#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from nautil_common import paths

ROOT = paths.DATA
RUN = paths.RUN
OUT = RUN / "results/ab_llm_v1"
AUDIT = RUN / "docs/SFT_POOL_AUDIT_20260924.json"
TOOLS = paths.CONFIGS / "sft_evidence_tool_v1.json"
SYSTEM = """You are an expert editor creating an auditable, single-investigator SFT trajectory from a frozen evidence package and a trusted reference report. The reference is editor-only. The simulated investigator initially sees only E1 full text and the remaining evidence titles. Return a JSON object with keys rounds and final. rounds must contain 3 to 7 objects, each with note, fetch_ids, reason. Each fetch_ids list contains 1 to 12 new evidence IDs, with no repeats across rounds. In the first note declare stable competing hypotheses H1 and H2 based on the opening brief; later notes explicitly update their support, identify uncertainty, and say what the next fetch discriminates. Notes may cite only E1 and IDs fetched in PRIOR rounds. The reason must justify the request based on what is already visible; never assert the contents of an unread item. Choose informative evidence groups in multiple rounds, not arbitrary batches. The final must start CASE CLOSED or CASE NOT CLOSED, answer the task question, cite only fetched or E1 IDs, address alternatives, and honestly preserve causal uncertainty. Use CASE CLOSED only when expected_closure begins determined; use CASE NOT CLOSED when it begins undetermined, even if the agency administratively closed its investigation. Never invent observations, measurements, or tool output. No hidden editor-only facts before they are fetched. English prose, concise but substantive."""
EID = re.compile(r"\bE[1-9][0-9]*\.[1-9][0-9]*\b")
HID = re.compile(r"\bH[1-9][0-9]*\b")


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_source(case_id):
    rows = {x["case_id"]: x for x in json.loads(AUDIT.read_text())["rows"]}
    row = rows[case_id]
    package = json.loads((ROOT / row["package_path"]).read_text())
    if row["family"] == "host":
        trace = json.loads((paths.DATASET / row["split"] / case_id / "trajectory/trajectory.json").read_text())
        gold = "Trusted multi-agent closing report:\n" + str(trace["result"].get("content") or "")
        reference = {"expected_closure": "host-level report", "withheld_identifiers": []}
    else:
        reference = json.loads((ROOT / row["reference_path"]).read_text())
        gold = "Trusted official analysis:\n" + str(reference.get("official_analysis_verbatim") or "") + "\n\nTrusted closure:\n" + str(reference.get("expected_closure") or "")
    editor_user = "FROZEN PACKAGE:\n" + json.dumps(package, ensure_ascii=False) + "\n\n" + gold
    return row, package, reference, editor_user


def opening_brief(package):
    record = [x for x in package["evidence_items"] if x["evidence_id"].startswith("E1.")]
    assert record
    indexed = [x for x in package["evidence_items"] if x not in record]
    shown = "\n".join(f'- {x["evidence_id"]} | {x["neutral_title"]} | {x.get("kind") or x.get("locator") or ""} | source {x["source_id"]}\n  {x["text"]}' for x in record)
    index = "\n".join(f'- {x["evidence_id"]} | {x["neutral_title"]} | {x.get("kind") or str(x.get("locator", "")).rsplit(" | ", 1)[-1]} | source {x["source_id"]}' for x in indexed)
    return (f'# Case brief — {package["case_id"]}\n\n## Task question\n{package["task_question"]}\n\n'
            f'## Initial context\n{package["initial_context"]}\n\n'
            f'## Case record E1 ({len(record)} items, full text)\n{shown}\n\n'
            f'## Index of the remaining {len(indexed)} evidence items (titles only)\n{index}\n\n'
            'Fetch exact text with request_evidence when needed. Finish with a direct answer '
            'that separates the supported mechanism, discounted alternatives, and unresolved details.')


def estimate(editor_user, cap):
    n = math.ceil((len(SYSTEM) + len(editor_user)) / 2.5)
    return {"input_tokens_upper_proxy": n, "output_tokens_cap": cap,
            "openai_reference_usd_upper": round(n * 2 / 1_000_000 + cap * 10 / 1_000_000, 4),
            "api_price_verified": False}


def api_key(name):
    for line in paths.ENV_FILE.read_text().splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise ValueError(f"{name} missing")


def fetch_model(editor_user, model, base, key_name, cap, timeout):
    body = {"model": model, "messages": [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": editor_user}], "reasoning_effort": "medium",
            "max_tokens": cap, "response_format": {"type": "json_object"}}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + api_key(key_name),
            "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def compile_plan(case_id, row, package, reference, plan, raw_path,
                 origin="gpt6sol_ab_llm_v1"):
    rounds = plan.get("rounds")
    if not isinstance(rounds, list) or not 3 <= len(rounds) <= 16:
        raise ValueError("round count outside 3..16")
    by_id = {x["evidence_id"]: x for x in package["evidence_items"]}
    disclosed = {x for x in by_id if x.startswith("E1.")}
    messages = [{"role": "user", "content": opening_brief(package), "loss": False}]
    initial_hids = None
    for number, step in enumerate(rounds, 1):
        note = str(step.get("note") or "")
        ids = step.get("fetch_ids") or []
        reason = str(step.get("reason") or "")
        if not note.strip() or not 1 <= len(ids) <= 12 or len(ids) != len(set(ids)) or len(reason) < 10:
            raise ValueError(f"round {number}: empty note, bad ids, or short reason")
        if any(x not in by_id for x in ids):
            raise ValueError(f"round {number}: unknown evidence ID")
        if set(EID.findall(note + ' ' + reason)) - disclosed:
            raise ValueError(f"round {number}: forward citation in note/reason")
        hids = set(HID.findall(note))
        if initial_hids is None:
            initial_hids = hids
            if len(hids) < 2:
                raise ValueError("first note lacks H1/H2")
        elif hids - initial_hids:
            raise ValueError(f"round {number}: undeclared hypothesis")
        cid = f"call_{number:03d}"
        args = {"evidence_ids": ids, "reason": reason}
        messages.append({"role": "assistant", "content": note, "tool_calls": [{"id": cid,
                         "type": "function", "function": {"name": "request_evidence", "arguments": args}}], "loss": True})
        repeated = [x for x in ids if x in disclosed]
        new = [x for x in ids if x not in disclosed]
        disclosed.update(new)
        tool_result = {"evidence_items": [by_id[x] for x in new]}
        if repeated:
            tool_result["already_delivered"] = repeated
        tool = compact(tool_result)
        messages.append({"role": "tool", "tool_call_id": cid, "name": "request_evidence",
                         "content": tool, "loss": False})
    final = str(plan.get("final") or "")
    if not final.startswith(("CASE CLOSED", "CASE NOT CLOSED")):
        raise ValueError("final lacks explicit closure status")
    expected = str(reference.get("expected_closure") or "").split(":", 1)[0].strip().lower()
    if expected == "determined" and final.startswith("CASE NOT CLOSED"):
        raise ValueError("final closure marker conflicts with determined reference")
    if expected == "undetermined" and not final.startswith("CASE NOT CLOSED"):
        raise ValueError("final closure marker conflicts with undetermined reference")
    if set(EID.findall(final)) - disclosed:
        raise ValueError("final cites unread evidence")
    messages.append({"role": "assistant", "content": final, "loss": True})
    trace = {"schema_version": "nautil.sft.v1",
             "system_prompt_ref": "investigation/prompts/sft_investigator_v1.txt",
             "sample_id": f"{row['family']}-{case_id}-{origin}", "case_id": case_id,
             "source": row["family"], "package_hash": package["package_hash"],
             "mode": "evidence_requested", "origin": origin,
             "review_status": "requires_case_review", "training_approved": False,
             "tools": json.loads(TOOLS.read_text()), "messages": messages,
             "labels": {"expected_closure": reference.get("expected_closure")},
             "provenance": {"package": row["package_path"],
                            "review_only_reference_path": row.get("reference_path"),
                            "raw_teacher_response": str(raw_path.relative_to(ROOT)),
                            "model_visible_fields": ["tools", "messages"]},
             "split": "review_draft"}
    visible = compact({"tools": trace["tools"], "messages": messages}).casefold()
    leaks = [x for x in reference.get("withheld_identifiers") or []
             if x and len(str(x)) >= 4 and str(x).casefold() in visible]
    if leaks:
        raise ValueError(f"withheld identifiers in model-visible trace: {leaks[:4]}")
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--base", default="https://example.com/v1")
    ap.add_argument("--model", default="gpt-6-sol")
    ap.add_argument("--key-var", default="API_KEY")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    row, package, reference, editor_user = load_source(args.case)
    cost = estimate(editor_user, args.max_tokens)
    print(json.dumps({"case": args.case, "estimate": cost}, ensure_ascii=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    prompt_path = OUT / f"{args.case}.editor_prompt.json"
    prompt_path.write_text(json.dumps({"system": SYSTEM, "user": editor_user}, ensure_ascii=False, indent=2) + "\n")
    if args.dry_run:
        return
    started = time.monotonic()
    response = fetch_model(editor_user, args.model, args.base, args.key_var, args.max_tokens, args.timeout)
    content = (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    raw_path = OUT / f"{args.case}.teacher_raw.json"
    raw_path.write_text(json.dumps({"case_id": args.case, "model": args.model,
                         "prompt_sha256": hashlib.sha256(editor_user.encode()).hexdigest(),
                         "estimate": cost, "elapsed_seconds": round(time.monotonic() - started, 2),
                         "usage": response.get("usage"), "teacher_text": content}, ensure_ascii=False, indent=2) + "\n")
    plan = json.loads(content)
    trace = compile_plan(args.case, row, package, reference, plan, raw_path)
    target = OUT / f"{args.case}.trace.jsonl"
    target.write_text(json.dumps(trace, ensure_ascii=False) + "\n")
    print(json.dumps({"case": args.case, "rounds": len(plan["rounds"]),
                      "usage": response.get("usage"), "trace": str(target.relative_to(ROOT))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
