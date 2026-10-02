#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from generate_ab_llm_v1 import load_source
from validate_hypothesis_ledger_v1 import validate
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from nautil_common import paths

ROOT = paths.DATA
RUN = paths.RUN
AUDIT = RUN / "results/sft_gpt6sol_scale60_v2_audit/audit.jsonl"
TRACES = RUN / "results/sft_scale60_evidence_recompiled_v2/traces"
OUT = RUN / "results/sft_final_only_revise_v1"
BASE = "https://example.com/v1/chat/completions"
MODEL = "gpt-6-sol"
KEY_ENV = "API_KEY"
SYSTEM = """You are making ONE narrow edit to a review-only Nautil investigation: revise only the final assistant answer. All earlier investigator notes, hypothesis ledgers, tool calls, and tool returns are immutable. The current draft scored +3 with no major logical or structural error; improve only final-answer issues reported by the auditor. Keep every valid factual claim and important uncertainty from the original final. Never add a fact or more certain cause unless one of the listed trainee-visible E items supports it. Cite exact individual E IDs already disclosed, and make the requested causal answer clear. Do not use agency administrative status as a substitute for causal determination. If an auditor issue concerns an earlier round, do not pretend to fix it in the final; mention it in editor_notes only. Return exactly one JSON object with final_answer and editor_notes. If no safe final change is possible, repeat the original final_answer verbatim and explain in editor_notes. No markdown wrapper around the JSON."""


def key() -> str:
    for line in paths.ENV_FILE.read_text().splitlines():
        if line.startswith(KEY_ENV + "="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    raise ValueError(f"{KEY_ENV} missing")


def case_ids() -> list[str]:
    rows = [json.loads(line) for line in AUDIT.read_text().splitlines() if line.strip()]
    return [item["case_id"] for item in rows if item.get("score") == 3
            and item.get("hard_failure") is False]


def editor(case_id: str, *, audit_record: dict | None = None,
           trace_dir: Path | None = None) -> tuple[dict, dict]:
    trace = json.loads(((trace_dir or TRACES) / f"{case_id}.jsonl").read_text())
    if trace.get("training_approved") is not False:
        raise ValueError("input trace is not review-only")
    row, package, reference, _ = load_source(case_id)
    visible = [item for item in package["evidence_items"] if item["evidence_id"].startswith("E1.")]
    for message in trace["messages"]:
        if message["role"] == "tool":
            visible.extend(json.loads(message["content"])["evidence_items"])
    audit = audit_record or next(item for item in map(json.loads, AUDIT.read_text().splitlines())
                                 if item["case_id"] == case_id)
    if row["family"] == "host":
        t = json.loads((paths.DATASET / row["split"] / case_id /
                        "trajectory/trajectory.json").read_text())
        trusted = {"closing_report": t["result"].get("content") or ""}
    else:
        trusted = {"expected_closure": reference.get("expected_closure"),
                   "official_analysis": reference.get("official_analysis_verbatim"),
                   "official_conclusion": reference.get("official_conclusion_verbatim")}
    prompt = {"case_id": case_id, "task_question": package["task_question"],
              "original_final_answer": trace["messages"][-1]["content"],
              "disclosed_evidence": visible,
              "auditor_issues": audit.get("issues") or [],
              "trusted_reference_editor_only": trusted}
    return trace, prompt


def estimate(prompt: dict) -> dict:
    chars = len(SYSTEM) + len(json.dumps(prompt, ensure_ascii=False))
    proxy = math.ceil(chars / 2.5 * 1.2)
    return {"input_token_proxy": proxy, "output_assumption_not_cap": 2000,
            "usd_reference_2k_output": round((proxy*2 + 2000*10)/1_000_000, 4),
            "usd_reference_6000_output": round((proxy*2 + 6000*10)/1_000_000, 4)}


def compile_saved(case_id: str, trace: dict, prompt_hash: str, raw_path: Path, reused: bool) -> dict:
    raw = json.loads(raw_path.read_text())
    result = {"case_id": case_id, "raw": str(raw_path.relative_to(ROOT)),
              "finish_reason": raw.get("finish_reason"), "usage": raw.get("usage"),
              "reused_saved_teacher_response": reused}
    if raw.get("prompt_sha256") != prompt_hash or raw.get("case_id") != case_id:
        return result | {"status": "cached_raw_mismatch"}
    if raw.get("finish_reason") == "length":
        return result | {"status": "truncated_output"}
    try:
        proposal = json.loads(raw.get("teacher_text") or "")
        final = proposal["final_answer"]
        if not isinstance(final, str) or len(final.strip()) < 50:
            raise ValueError("final_answer missing or too short")
        original_final = trace["messages"][-1]["content"]
        original_marker = re.match(r"CASE (?:NOT )?CLOSED", original_final)
        marker_restored = False
        if original_marker and not re.match(r"CASE (?:NOT )?CLOSED", final):
            if re.search(r"\bCASE (?:NOT )?CLOSED\b", final):
                raise ValueError("new final has a conflicting or misplaced closure marker")
            final = original_marker.group() + " — " + final.lstrip()
            marker_restored = True
        revised = copy.deepcopy(trace)
        revised["messages"][-1]["content"] = final
        if revised["messages"][:-1] != trace["messages"][:-1]:
            raise AssertionError("a non-final message changed")
        revised["sample_id"] += "-final-only-revised-v1"
        revised["origin"] = "gpt6sol_final_only_v1"
        revised["review_status"] = "final_only_revision_requires_rescore"
        revised["training_approved"] = False
        revised["provenance"]["final_revision_teacher_raw"] = str(raw_path.relative_to(ROOT))
        revised["provenance"]["final_revision_editor_notes"] = proposal.get("editor_notes")
        if marker_restored:
            revised["provenance"]["original_closure_marker_restored_in_final_only_edit"] = True
        metrics = validate(revised)
        target = OUT / "traces" / f"{case_id}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(revised, ensure_ascii=False) + "\n")
        temporary.replace(target)
        return result | {"status": "final_only_review_draft", "trace": str(target.relative_to(ROOT)),
                         "final_changed": final != trace["messages"][-1]["content"],
                         "original_closure_marker_restored": marker_restored,
                         "frozen_tool_calls": metrics["tool_calls"]}
    except Exception as exc:
        return result | {"status": "revision_rejected", "error": f"{type(exc).__name__}: {str(exc)[:260]}"}


def run(case_id: str, api_key: str, sem: threading.Semaphore,
        audit_record: dict | None = None, trace_dir: Path | None = None) -> dict:
    try:
        trace, prompt = editor(case_id, audit_record=audit_record, trace_dir=trace_dir)
    except Exception as exc:
        return {"case_id": case_id, "status": "input_error", "error": f"{type(exc).__name__}: {str(exc)[:250]}"}
    user_text = json.dumps(prompt, ensure_ascii=False)
    prompt_hash = hashlib.sha256((SYSTEM + user_text).encode()).hexdigest()
    raw_path = OUT / "raw" / f"{case_id}.json"
    if raw_path.exists():
        return compile_saved(case_id, trace, prompt_hash, raw_path, True)
    inflight = OUT / "inflight" / f"{case_id}.json"
    if inflight.exists():
        return {"case_id": case_id, "status": "uncertain_inflight_needs_manual_retry"}
    body = {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text}], "response_format": {"type": "json_object"}}
    request = urllib.request.Request(BASE, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    started = time.monotonic()
    with sem:
        inflight.parent.mkdir(parents=True, exist_ok=True)
        inflight.write_text(json.dumps({"case_id": case_id, "prompt_sha256": prompt_hash,
                                        "started_unix": time.time()}) + "\n")
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            inflight.unlink(missing_ok=True)
            return {"case_id": case_id, "status": "http_error", "http_status": exc.code,
                    "error": exc.read(300).decode(errors="replace")[:250]}
        except Exception as exc:
            inflight.unlink(missing_ok=True)
            return {"case_id": case_id, "status": "transport_error",
                    "error": f"{type(exc).__name__}: {str(exc)[:250]}"}
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    if isinstance(content, list):
        content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"case_id": case_id, "model": MODEL,
        "model_returned": data.get("model"), "prompt_sha256": prompt_hash,
        "finish_reason": choice.get("finish_reason"), "usage": data.get("usage"),
        "elapsed_seconds": round(time.monotonic()-started, 2), "teacher_text": content},
        ensure_ascii=False, indent=2) + "\n")
    temporary.replace(raw_path)
    inflight.unlink(missing_ok=True)
    return compile_saved(case_id, trace, prompt_hash, raw_path, False)


def main() -> None:
    global AUDIT, TRACES, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--audit", type=Path, default=AUDIT)
    ap.add_argument("--trace-dir", type=Path, default=TRACES)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()
    AUDIT = args.audit.resolve()
    TRACES = args.trace_dir.resolve()
    OUT = args.output_dir.resolve()
    if not OUT.is_relative_to(ROOT):
        raise ValueError("output-dir must be inside the workspace")
    ids = case_ids()
    estimates = []
    for cid in ids:
        try:
            _, prompt = editor(cid)
            estimates.append({"case_id": cid, **estimate(prompt)})
        except Exception as exc:
            estimates.append({"case_id": cid, "error": f"{type(exc).__name__}: {str(exc)[:250]}"})
    summary = {"cases": len(ids), "usd_reference_2k_output": round(sum(x.get("usd_reference_2k_output", 0) for x in estimates), 4),
               "usd_reference_6000_output": round(sum(x.get("usd_reference_6000_output", 0) for x in estimates), 4),
               "output_cap_sent": False, "api_price_verified": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pre_run_estimate.json").write_text(json.dumps({"summary": summary, "cases": estimates},
                                                    ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"estimate": summary, "execute": args.execute}, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    api_key = key()
    sem = threading.Semaphore(paths.CONCURRENCY)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(paths.CONCURRENCY, len(ids)) or 1) as pool:
        futures = {pool.submit(run, cid, api_key, sem): cid for cid in ids}
        for future in concurrent.futures.as_completed(futures):
            cid = futures[future]
            try:
                item = future.result()
            except Exception as exc:
                item = {"case_id": cid, "status": "local_error",
                        "error": f"{type(exc).__name__}: {str(exc)[:260]}"}
            results[cid] = item
            print(json.dumps({k: item.get(k) for k in ("case_id", "status", "error", "final_changed")},
                             ensure_ascii=False), flush=True)
    ordered = [results[cid] for cid in ids]
    (OUT / "status.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered))
    accepted = [item["case_id"] for item in ordered if item["status"] == "final_only_review_draft"]
    (OUT / "audit_manifest.json").write_text(json.dumps({"case_ids": accepted}, ensure_ascii=False, indent=2) + "\n")
    (OUT / "summary.json").write_text(json.dumps({"cases": len(ids),
        "statuses": dict(Counter(item["status"] for item in ordered)),
        "nonfinal_messages_changed": 0, "training_approved_changed": False},
        ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
