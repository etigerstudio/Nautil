#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import audit_sft_scale60_evidence_v2 as rubric
from run_sft_audit_repair_v2 import OUT, ROOT, RUN, compact, atomic_json
from nautil_common import paths
from validate_hypothesis_ledger_v1 import validate as validate_trace

AUDIT = rubric.audit
AUDIT.MODEL = "gpt-6-sol"
OUT_DIR = OUT / "blind_score"
MODEL = "gpt-6-sol"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def keys() -> list[tuple[str, str]]:
    return AUDIT.keys()


def blind_editor(case_id: str, trace: dict, meta: dict) -> dict:
    payload = {"case_id": case_id,
               "trusted_reference_reviewer_only": AUDIT.reviewer_reference(case_id, meta),
               "candidate_model_visible_trace": {"tools": trace["tools"], "messages": trace["messages"]}}
    if set(payload) != {"case_id", "trusted_reference_reviewer_only", "candidate_model_visible_trace"}:
        raise AssertionError("blind editor payload changed")
    if set(payload["candidate_model_visible_trace"]) != {"tools", "messages"}:
        raise AssertionError("blind scorer sees non-model-visible candidate fields")
    return payload


def parse_saved(case_id: str, raw_path: Path, prompt_hash: str) -> dict:
    raw = json.loads(raw_path.read_text())
    if raw.get("case_id") != case_id or raw.get("prompt_sha256") != prompt_hash:
        return {"case_id": case_id, "status": "cached_prompt_mismatch"}
    result = {"case_id": case_id, "route": raw.get("route"),
              "raw": str(raw_path.relative_to(ROOT)), "usage": raw.get("usage"),
              "finish_reason": raw.get("finish_reason"), "model_requested": raw.get("model_requested"),
              "model_returned": raw.get("model_returned")}
    if raw.get("finish_reason") == "length":
        return result | {"status": "truncated_output"}
    try:
        parsed = json.loads(raw.get("teacher_text") or "")
        if (parsed.get("case_id") != case_id or not isinstance(parsed.get("score"), int)
                or not -5 <= parsed["score"] <= 5 or parsed.get("action") not in {"keep", "revise", "skip"}
                or not isinstance(parsed.get("hard_failure"), bool)
                or not isinstance(parsed.get("issues"), list)):
            raise ValueError("missing or invalid case_id, score, action, hard_failure, or issues")
        return result | {"status": "scored", "score": parsed["score"], "action": parsed["action"],
                         "hard_failure": parsed["hard_failure"],
                         "hard_failure_reason": parsed.get("hard_failure_reason"),
                         "checks": parsed.get("checks"), "issues": parsed["issues"],
                         "verdict": parsed.get("verdict")}
    except Exception as exc:
        return result | {"status": "invalid_scorer_json", "error": f"{type(exc).__name__}: {str(exc)[:250]}"}


def score_one(case_id: str, trace_path: Path, meta: dict, route_name: str,
              api_key: str, semaphore: threading.Semaphore) -> dict:
    status_path = OUT_DIR / "status" / f"{case_id}.json"
    if status_path.exists():
        return json.loads(status_path.read_text())
    try:
        trace = json.loads(trace_path.read_text())
        validate_trace(trace)
        if trace["case_id"] != case_id or trace["training_approved"] is not False:
            raise ValueError("invalid candidate trace")
        payload = blind_editor(case_id, trace, meta)
    except Exception as exc:
        result = {"case_id": case_id, "status": "local_structure_rejected",
                  "error": f"{type(exc).__name__}: {str(exc)[:250]}"}
        atomic_json(status_path, result)
        return result
    user_text = compact(payload)
    prompt_hash = hashlib.sha256((AUDIT.SYSTEM+user_text).encode()).hexdigest()
    editor_path = OUT_DIR / "editor_inputs" / f"{case_id}.json"
    atomic_json(editor_path, payload)
    raw_path = OUT_DIR / "raw" / f"{case_id}.json"
    if raw_path.exists():
        result = parse_saved(case_id, raw_path, prompt_hash)
        atomic_json(status_path, result)
        return result
    inflight = OUT_DIR / "inflight" / f"{case_id}.json"
    if inflight.exists():
        result = {"case_id": case_id, "status": "uncertain_inflight_no_blind_retry"}
        atomic_json(status_path, result)
        return result
    body = {"model": MODEL, "messages": [{"role": "system", "content": AUDIT.SYSTEM},
            {"role": "user", "content": user_text}], "response_format": {"type": "json_object"}}
    request = urllib.request.Request("https://example.com/v1/chat/completions",
        data=compact(body).encode(), method="POST",
        headers={"Authorization": "Bearer "+api_key, "Content-Type": "application/json",
                 "HTTP-Referer": "https://localhost/nautil", "X-Title": "Nautil Blind Audit v2"})
    with semaphore:
        atomic_json(inflight, {"case_id": case_id, "route": route_name,
                               "prompt_sha256": prompt_hash, "started_unix": time.time()})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            inflight.unlink(missing_ok=True)
            result = {"case_id": case_id, "route": route_name, "status": "http_error",
                      "http_status": exc.code, "error": exc.read(350).decode(errors="replace")[:250]}
            atomic_json(status_path, result)
            return result
        except Exception as exc:
            inflight.unlink(missing_ok=True)
            result = {"case_id": case_id, "route": route_name, "status": "transport_error_uncertain",
                      "error": f"{type(exc).__name__}: {str(exc)[:250]}"}
            atomic_json(status_path, result)
            return result
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    if isinstance(content, list):
        content = "".join(x.get("text", "") for x in content if isinstance(x, dict))
    atomic_json(raw_path, {"case_id": case_id, "route": route_name,
                           "model_requested": MODEL, "model_returned": data.get("model"),
                           "provider_generation_id": data.get("id"),
                           "prompt_sha256": prompt_hash,
                           "elapsed_seconds": round(time.monotonic()-started, 2),
                           "usage": data.get("usage"), "finish_reason": choice.get("finish_reason"),
                           "teacher_text": content})
    inflight.unlink(missing_ok=True)
    result = parse_saved(case_id, raw_path, prompt_hash)
    atomic_json(status_path, result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    gate = OUT / "repair_phase_complete.json"
    if not gate.exists():
        raise ValueError("all 100 repair tasks must reach terminal status before blind scoring")
    repair_summary = json.loads(gate.read_text())
    if repair_summary.get("cases") != 100 or not repair_summary.get("repair_phase_complete"):
        raise ValueError("repair phase completion gate is invalid")
    candidates = repair_summary["compiled_case_ids"]
    if len(set(candidates)) != len(candidates):
        raise ValueError("duplicate compiled case IDs")
    meta = {row["case_id"]: row for row in
            json.loads((RUN / "docs/SFT_POOL_AUDIT_20260924.json").read_text())["rows"]}
    input_proxy = 0
    for cid in candidates:
        trace = json.loads((OUT / "traces" / f"{cid}.jsonl").read_text())
        payload = blind_editor(cid, trace, meta[cid])
        input_proxy += math.ceil((len(AUDIT.SYSTEM)+len(compact(payload)))/2.5*1.2)
    estimate = {"cases": len(candidates), "input_token_proxy": input_proxy,
                "usd_reference_3k_output": round((input_proxy*2+len(candidates)*3000*10)/1_000_000, 4),
                "usd_reference_6k_output": round((input_proxy*2+len(candidates)*6000*10)/1_000_000, 4),
                "output_cap_sent": False, "retail_price_verified": False}
    atomic_json(OUT_DIR / "pre_run_estimate.json", estimate)
    print(compact({"blind_score_candidates": len(candidates), "repair_cases_terminal": 100,
                   "model": MODEL, "uses_old_scores_or_patch_notes": False,
                   "estimate": estimate, "execute": args.execute}), flush=True)
    if not args.execute:
        return
    configured = keys()
    semaphores = {name: threading.Semaphore(paths.CONCURRENCY) for name, _ in configured}
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=paths.CONCURRENCY) as pool:
        futures = {}
        for index, cid in enumerate(candidates):
            name, key = configured[index % 2]
            future = pool.submit(score_one, cid, OUT / "traces" / f"{cid}.jsonl",
                                 meta[cid], name, key, semaphores[name])
            futures[future] = cid
        for future in concurrent.futures.as_completed(futures):
            cid = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"case_id": cid, "status": "local_error",
                          "error": f"{type(exc).__name__}: {str(exc)[:250]}"}
                atomic_json(OUT_DIR / "status" / f"{cid}.json", result)
            results[cid] = result
            print(compact({k: result.get(k) for k in ("case_id", "status", "score", "action", "hard_failure")}), flush=True)
    ordered = [results[cid] for cid in candidates]
    (OUT_DIR / "audit.jsonl").write_text("".join(compact(item) + "\n" for item in ordered))
    summary = {"repair_cases_terminal": 100, "blind_score_candidates": len(candidates),
               "statuses": dict(Counter(item["status"] for item in ordered)),
               "scores": dict(Counter(str(item.get("score")) for item in ordered)),
               "blind_score_phase_complete": True}
    atomic_json(OUT_DIR / "summary.json", summary)
    print(compact({"blind_score_summary": summary}), flush=True)


if __name__ == "__main__":
    main()
