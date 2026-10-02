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

import assess_sft_audit_issues_v2 as assessment
import repair_sft_audit_v2 as repair
from generate_sft_trajectories_v1 import env
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from nautil_common import paths

ROOT = paths.DATA
RUN = paths.RUN
OUT = RUN / "results/sft_audit_repair_v2"
ROUTES = paths.CONFIGS / "api_routes.json"
PRICE_IN = 2.0
PRICE_OUT = 10.0


def compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class RoutePool:
    def __init__(self, specs: list[dict], values: dict[str, str]):
        self.specs = {spec["name"]: spec for spec in specs}
        self.values = values
        self.active = {name: 0 for name in self.specs}
        self.pause_until = {name: 0.0 for name in self.specs}
        self.cv = threading.Condition()
        for spec in specs:
            if not values.get(spec["key_env"]):
                raise ValueError(f"missing key for route {spec['name']}")

    def acquire(self, exclude: str | None = None) -> tuple[dict, str]:
        with self.cv:
            while True:
                now = time.monotonic()
                candidates = [spec for spec in self.specs.values()
                              if spec["name"] != exclude and self.active[spec["name"]] < spec["max_concurrent"]
                              and now >= self.pause_until[spec["name"]]]
                if not candidates and exclude:
                    candidates = [spec for spec in self.specs.values()
                                  if self.active[spec["name"]] < spec["max_concurrent"]
                                  and now >= self.pause_until[spec["name"]]]
                if candidates:
                    spec = min(candidates, key=lambda x: (self.active[x["name"]]/x["max_concurrent"],
                                                           self.active[x["name"]], x["name"]))
                    self.active[spec["name"]] += 1
                    return spec, self.values[spec["key_env"]]
                self.cv.wait(timeout=1)

    def release(self, name: str, *, backoff_seconds: int = 0) -> None:
        with self.cv:
            self.active[name] -= 1
            if backoff_seconds:
                self.pause_until[name] = max(self.pause_until[name], time.monotonic()+backoff_seconds)
            self.cv.notify_all()


def estimate(records: list[dict]) -> dict:
    repair_proxy = 0
    score_proxy = 0
    from audit_sft_scale60_evidence_v2 import audit as reviewer
    reviewer.MODEL = "gpt-6-sol"
    for record in records:
        payload = repair.make_editor(record)
        repair_proxy += math.ceil((len(repair.SYSTEM)+len(compact(payload)))/2.5*1.2)
        trace = json.loads((ROOT / record["original_trace_path"]).read_text())
        reference = reviewer.reviewer_reference(record["case_id"], record["pool_row"])
        blind = {"case_id": record["case_id"], "trusted_reference_reviewer_only": reference,
                 "candidate_model_visible_trace": {"tools": trace["tools"], "messages": trace["messages"]}}
        score_proxy += math.ceil((len(reviewer.SYSTEM)+len(compact(blind)))/2.5*1.2)
    return {"cases": len(records), "repair_input_proxy": repair_proxy,
            "repair_reference_usd_at_4k_output": round((repair_proxy*PRICE_IN+len(records)*4000*PRICE_OUT)/1_000_000, 4),
            "repair_reference_usd_at_12k_output": round((repair_proxy*PRICE_IN+len(records)*12000*PRICE_OUT)/1_000_000, 4),
            "blind_score_input_proxy": score_proxy,
            "blind_score_reference_usd_at_3k_output": round((score_proxy*PRICE_IN+len(records)*3000*PRICE_OUT)/1_000_000, 4),
            "blind_score_reference_usd_at_6k_output": round((score_proxy*PRICE_IN+len(records)*6000*PRICE_OUT)/1_000_000, 4),
            "output_cap_sent": False, "api_price_verified": False,
            "retry_cost_not_included": True}


def source_records() -> list[dict]:
    base = [json.loads(line) for line in (OUT / "candidates.jsonl").read_text().splitlines() if line.strip()]
    issue_map = {x["case_id"]: x for x in map(json.loads,
        (OUT / "issue_assessment.jsonl").read_text().splitlines())}
    records = repair.attach_pool_rows(base)
    return [{**record, "issue_assessment": issue_map[record["case_id"]]} for record in records]


def call_teacher(record: dict, prompt: dict, attempt: int, routes: RoutePool,
                 exclude_route: str | None = None) -> dict:
    cid = record["case_id"]
    raw_path = OUT / "raw" / f"{cid}.attempt{attempt}.json"
    user_text = compact(prompt)
    prompt_hash = hashlib.sha256((repair.SYSTEM+user_text).encode()).hexdigest()
    if raw_path.exists():
        cached = json.loads(raw_path.read_text())
        if cached.get("prompt_sha256") != prompt_hash:
            return {"case_id": cid, "attempt": attempt, "status": "cached_prompt_mismatch"}
        return cached
    inflight = OUT / "inflight" / f"{cid}.attempt{attempt}.json"
    if inflight.exists():
        return {"case_id": cid, "attempt": attempt, "status": "uncertain_inflight_no_blind_retry"}
    spec, key = routes.acquire(exclude_route)
    body = {"model": spec["wire_model"],
            "messages": [{"role": "system", "content": repair.SYSTEM},
                         {"role": "user", "content": user_text}],
            "response_format": {"type": "json_object"}}
    request = urllib.request.Request(spec["base_url"].rstrip("/")+"/chat/completions",
        data=compact(body).encode(), method="POST",
        headers={"Authorization": "Bearer "+key, "Content-Type": "application/json"})
    atomic_json(inflight, {"case_id": cid, "attempt": attempt,
                           "route": spec["name"], "prompt_sha256": prompt_hash,
                           "started_unix": time.time()})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=360) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(400).decode(errors="replace")
        inflight.unlink(missing_ok=True)
        routes.release(spec["name"], backoff_seconds=20 if exc.code in {429, 503} else 0)
        return {"case_id": cid, "attempt": attempt, "route": spec["name"],
                "status": "http_error", "http_status": exc.code,
                "error": detail[:250], "elapsed_seconds": round(time.monotonic()-started, 2)}
    except Exception as exc:
        inflight.unlink(missing_ok=True)
        routes.release(spec["name"], backoff_seconds=10)
        return {"case_id": cid, "attempt": attempt, "route": spec["name"],
                "status": "transport_error_uncertain", "error":
                f"{type(exc).__name__}: {str(exc)[:250]}",
                "elapsed_seconds": round(time.monotonic()-started, 2)}
    routes.release(spec["name"])
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    if isinstance(content, list):
        content = "".join(x.get("text", "") for x in content if isinstance(x, dict))
    raw = {"case_id": cid, "attempt": attempt, "route": spec["name"],
           "model_requested": spec["wire_model"], "model_returned": data.get("model"),
           "finish_reason": choice.get("finish_reason"), "usage": data.get("usage"),
           "prompt_sha256": prompt_hash, "elapsed_seconds": round(time.monotonic()-started, 2),
           "teacher_text": content}
    atomic_json(raw_path, raw)
    inflight.unlink(missing_ok=True)
    return raw


def repair_one(record: dict, routes: RoutePool) -> dict:
    cid = record["case_id"]
    prior_status = OUT / "status" / f"{cid}.json"
    if prior_status.exists():
        return json.loads(prior_status.read_text())
    original = json.loads((ROOT / record["original_trace_path"]).read_text())
    base_prompt = repair.make_editor(record)
    last_error = None
    last_route = None
    last_teacher_text = None
    attempts = []
    for attempt in (1, 2):
        prompt = base_prompt if attempt == 1 else {**base_prompt,
            "directed_retry": {"previous_error": last_error,
                               "previous_teacher_output": last_teacher_text,
                               "instruction": "Correct only the recorded technical/compilation error. Keep source-grounded local edits and the same case ID."}}
        raw = call_teacher(record, prompt, attempt, routes, last_route)
        last_teacher_text = raw.get("teacher_text")
        attempts.append({k: raw.get(k) for k in ("attempt", "route", "status", "error", "finish_reason", "usage")})
        last_route = raw.get("route")
        if raw.get("status") == "uncertain_inflight_no_blind_retry" or raw.get("status") == "transport_error_uncertain":
            result = {"case_id": cid, "status": raw["status"], "attempts": attempts}
            atomic_json(prior_status, result)
            return result
        if raw.get("status") == "http_error":
            last_error = f"HTTP {raw.get('http_status')}: {raw.get('error')}"
            if attempt == 1:
                continue
            break
        if raw.get("status") == "cached_prompt_mismatch":
            last_error = "cached prompt hash mismatch"
            break
        if raw.get("finish_reason") == "length":
            last_error = "finish_reason=length: provider truncated output"
            if attempt == 1:
                continue
            break
        raw_path = OUT / "raw" / f"{cid}.attempt{attempt}.json"
        try:
            proposal = json.loads(raw.get("teacher_text") or "")
        except Exception as exc:
            last_error = f"invalid JSON: {type(exc).__name__}: {str(exc)[:180]}"
            if attempt == 1:
                continue
            break
        patch_path = OUT / "patches" / f"{cid}.attempt{attempt}.json"
        atomic_json(patch_path, proposal)
        if not proposal.get("operations") and not proposal.get("new_evidence"):
            reason = str(proposal.get("editor_summary") or "no source-grounded edit proposed")
            result = {"case_id": cid, "status": "unfixable_or_no_change", "reason": reason,
                      "patch": str(patch_path.relative_to(ROOT)), "attempts": attempts}
            atomic_json(prior_status, result)
            return result
        try:
            trace, changes, evidence_sources = repair.compile_patch(record, proposal, raw_path)
            trace["provenance"]["repair_patch_path"] = str(patch_path.relative_to(ROOT))
            trace_path = OUT / "traces" / f"{cid}.jsonl"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = trace_path.with_suffix(".tmp")
            temporary.write_text(compact(trace) + "\n")
            temporary.replace(trace_path)
            issue_result = assessment.assess_after(record, original, trace, proposal)
            atomic_json(OUT / "issue_results" / f"{cid}.json", issue_result)
            atomic_json(OUT / "patch_audit" / f"{cid}.json",
                {"case_id": cid, "original_trace": record["original_trace_path"],
                 "original_trace_sha256": record["original_trace_sha256"],
                 "model_raw": str(raw_path.relative_to(ROOT)), "patch": str(patch_path.relative_to(ROOT)),
                 "changes": changes, "new_evidence_sources": evidence_sources,
                 "issue_result": str((OUT / 'issue_results' / f'{cid}.json').relative_to(ROOT))})
            result = {"case_id": cid, "status": "compiled_review_draft",
                      "trace": str(trace_path.relative_to(ROOT)),
                      "patch": str(patch_path.relative_to(ROOT)),
                      "changed_operations": len(changes), "new_evidence": len(evidence_sources),
                      "attempts": attempts}
            atomic_json(prior_status, result)
            return result
        except Exception as exc:
            last_error = f"compiler: {type(exc).__name__}: {str(exc)[:250]}"
            if attempt == 1:
                continue
            break
    result = {"case_id": cid, "status": "repair_failed", "reason": last_error,
              "attempts": attempts}
    atomic_json(prior_status, result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    records = source_records()
    if len(records) != 100:
        raise ValueError("expected exactly 100 candidates")
    estimates = estimate(records)
    atomic_json(OUT / "pre_run_estimate.json", estimates)
    print(compact({"estimate": estimates, "execute": args.execute}), flush=True)
    if not args.execute:
        return
    specs = json.loads(ROUTES.read_text())["routes"]
    routes = RoutePool(specs, env())
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=paths.CONCURRENCY) as pool:
        futures = {pool.submit(repair_one, record, routes): record["case_id"] for record in records}
        for future in concurrent.futures.as_completed(futures):
            cid = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"case_id": cid, "status": "local_error",
                          "reason": f"{type(exc).__name__}: {str(exc)[:250]}"}
                atomic_json(OUT / "status" / f"{cid}.json", result)
            results[cid] = result
            print(compact({k: result.get(k) for k in ("case_id", "status", "reason", "changed_operations")}), flush=True)
    ordered = [results[record["case_id"]] for record in records]
    (OUT / "repair_status.jsonl").write_text("".join(compact(result) + "\n" for result in ordered))
    summary = {"cases": 100, "statuses": dict(Counter(result["status"] for result in ordered)),
               "repair_phase_complete": True, "blind_scoring_started": False,
               "compiled_case_ids": [result["case_id"] for result in ordered if result["status"] == "compiled_review_draft"]}
    atomic_json(OUT / "repair_phase_complete.json", summary)
    print(compact({"repair_phase_summary": summary["statuses"]}), flush=True)


if __name__ == "__main__":
    main()
