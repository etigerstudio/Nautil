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
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from nautil_common import paths

ROOT = paths.DATA
RUN = paths.RUN
CASES = paths.CASES / "auto/scale_regrouped_v2"
MANIFEST = paths.CONFIGS / "ab_pilot14_v1.json"
AUDIT_POOL = RUN / "docs/SFT_POOL_AUDIT_20260924.json"
SOURCE = RUN / "results/sft_lossless_dedup_v1_1"
TRACE_DIR = SOURCE
TRACE_SUFFIX = ".lossless.jsonl"
OUT = RUN / "results/sft_audit14_v1"
MODEL = "claude-sonnet-5"
ENDPOINT = "https://example.com/v1/chat/completions"
PRICE_INPUT = 2.0
PRICE_OUTPUT = 10.0
OUTPUT_CAP = 4000
REASONING_EFFORT: str | None = None
NO_OUTPUT_CAP = False

SYSTEM = """You are an independent auditor of a proposed single-investigator SFT trajectory. The trusted report is reviewer-only; it is NOT part of the trainee's visible context. Evaluate the trajectory that the trainee would see: active evidence selection, factual tool returns, stable hypothesis updates, honest causal closure, and source-case alignment. The report may contain PDF headers or factual sections despite an analysis field name. Do not demand perfection: minor phrasing, anonymization, or missing peripheral citations can still be usable. If the trace and report are different incidents, or a decisive final claim is absent from all disclosed evidence, recommend skip with the exact reason. Never infer a fact that neither the tool transcript nor trusted report says. Give one overall integer score from -5 to +5: +5 excellent; +4 strong; +3 usable after small edits; +2 useful but nontrivial repair needed; +1 weak salvageable skeleton; 0 uncertain; negative scores mean do not use without rebuilding; -4 or -5 for source mismatch or deeply unsupported outcome. Action is keep/revise/skip as a recommendation only: keep generally requires >=3 and no critical issue; revise is a fixable 0..2; skip is negative or a hard source/decisive-support failure. Treat an agency's administrative closure separately from whether the cause is established. Output exactly one JSON object with keys case_id, score, action, critical_issue, checks, strengths, issues, verdict. checks is an object with source_alignment, evidence_grounding, active_fetch, hypothesis_trajectory, honest_closure, each pass/concern/fail plus one short reason. issues is a list of objects {severity, round, finding, evidence_ids, suggested_fix}; severity minor/major/critical. Keep strengths to at most three short strings and issues to at most six, with trace evidence IDs or exact short text to substantiate each material criticism. Do not approve training; this is a reviewer draft."""


def compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def key() -> str:
    for line in paths.ENV_FILE.read_text().splitlines():
        if line.startswith("API_KEY="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    raise ValueError("API_KEY missing")


def keys() -> list[tuple[str, str]]:
    return [("api", key())]


def provider_keys(provider: str) -> list[tuple[str, str, int]]:
    return [(name, value, paths.CONCURRENCY) for name, value in keys()]


def cases() -> list[str]:
    return json.loads(MANIFEST.read_text())["case_ids"]


def pool_rows() -> dict[str, dict]:
    return {row["case_id"]: row for row in json.loads(AUDIT_POOL.read_text())["rows"]}


def reviewer_reference(case_id: str, meta: dict) -> dict:
    if meta["family"] == "host":
        path = paths.DATASET / meta["split"] / case_id / "trajectory/trajectory.json"
        trajectory = json.loads(path.read_text())
        return {"source_family": "host", "trusted_closing_report": trajectory["result"].get("content") or ""}
    reference = json.loads((ROOT / meta["reference_path"]).read_text())
    return {"source_family": meta["family"],
            "dataset": reference["source"]["dataset"],
            "source_document": reference["source"].get("document"),
            "expected_closure": reference.get("expected_closure"),
            "official_analysis_verbatim": reference.get("official_analysis_verbatim"),
            "official_conclusion_verbatim": reference.get("official_conclusion_verbatim")}


def editor(case_id: str, meta: dict) -> dict:
    trace = json.loads((TRACE_DIR / f"{case_id}{TRACE_SUFFIX}").read_text())
    assert trace["training_approved"] is False
    visible = {"tools": trace["tools"], "messages": trace["messages"]}
    return {"case_id": case_id, "trusted_reference_reviewer_only": reviewer_reference(case_id, meta),
            "candidate_model_visible_trace": visible}


def estimate(case_id: str, meta: dict) -> dict:
    payload = compact(editor(case_id, meta))
    input_proxy = math.ceil((len(SYSTEM) + len(payload)) / 2.5 * 1.25)
    usd = (PRICE_INPUT * input_proxy + PRICE_OUTPUT * OUTPUT_CAP) / 1_000_000
    return {"case_id": case_id, "prompt_chars": len(SYSTEM) + len(payload),
            "input_token_conservative_proxy": input_proxy,
            "output_token_cap": OUTPUT_CAP,
            "usd_reference_ceiling": round(usd, 4)}


def parse_saved_audit(case_id: str, raw_path: Path, prompt_hash: str) -> dict:
    record = json.loads(raw_path.read_text())
    if record.get("case_id") != case_id or record.get("prompt_sha256") != prompt_hash:
        return {"case_id": case_id, "error": "cached_audit_prompt_mismatch",
                "raw_response_path": str(raw_path.relative_to(ROOT))}
    if record.get("finish_reason") == "length":
        return {"case_id": case_id, "error": "truncated_output",
                "raw_response_path": str(raw_path.relative_to(ROOT)), "usage": record.get("usage")}
    try:
        parsed = json.loads(record.get("teacher_text") or "")
        if parsed.get("case_id") != case_id or not isinstance(parsed.get("score"), int) or not -5 <= parsed["score"] <= 5:
            raise ValueError("case id or -5..5 score invalid")
        if parsed.get("action") not in {"keep", "revise", "skip"}:
            raise ValueError("action invalid")
        parsed["audit_model"] = MODEL
        parsed["audit_route"] = record.get("route") or "api"
        parsed["raw_response_path"] = str(raw_path.relative_to(ROOT))
        parsed["usage"] = record.get("usage")
        return parsed
    except Exception as exc:
        return {"case_id": case_id, "error": f"produced_response_parse_failure: {exc}",
                "raw_response_path": str(raw_path.relative_to(ROOT)), "usage": record.get("usage")}


def request(case_id: str, meta: dict, api_key: str, timeout: int,
            semaphore: threading.Semaphore | None = None, route: str = "api") -> dict:
    prompt = compact(editor(case_id, meta))
    prompt_hash = hashlib.sha256((SYSTEM + prompt).encode()).hexdigest()
    raw_path = OUT / "raw" / f"{case_id}.json"
    if raw_path.exists():
        return parse_saved_audit(case_id, raw_path, prompt_hash)
    inflight = OUT / "inflight" / f"{case_id}.json"
    if inflight.exists():
        return {"case_id": case_id, "error": "uncertain_inflight_needs_manual_retry"}
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}}
    if not NO_OUTPUT_CAP:
        body["max_tokens"] = OUTPUT_CAP
    if REASONING_EFFORT:
        body["reasoning_effort"] = REASONING_EFFORT
    req = urllib.request.Request(ENDPOINT, data=compact(body).encode(), method="POST",
            headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json",
                     "HTTP-Referer": "https://localhost/nautil", "X-Title": "Nautil SFT Audit"})
    started = time.monotonic()
    guard = semaphore or threading.Semaphore(1)
    with guard:
        inflight.parent.mkdir(parents=True, exist_ok=True)
        inflight.write_text(json.dumps({"case_id": case_id, "route": route,
                                        "prompt_sha256": prompt_hash,
                                        "started_unix": time.time()}) + "\n")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as exc:
            inflight.unlink(missing_ok=True)
            detail = exc.read(500).decode(errors="replace")
            return {"case_id": case_id, "audit_route": route,
                    "error": f"HTTP {exc.code}: {detail[:300]}",
                    "elapsed_seconds": round(time.monotonic()-started, 2)}
        except Exception as exc:
            inflight.unlink(missing_ok=True)
            return {"case_id": case_id, "audit_route": route,
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "elapsed_seconds": round(time.monotonic()-started, 2)}
    choice = (raw.get("choices") or [{}])[0]
    content = choice.get("message", {}).get("content")
    if isinstance(content, list):
        content = "".join(x.get("text", "") for x in content if isinstance(x, dict))
    record = {"case_id": case_id, "route": route, "model_requested": MODEL, "model_returned": raw.get("model"),
              "provider_generation_id": raw.get("id"), "prompt_sha256": prompt_hash,
              "elapsed_seconds": round(time.monotonic()-started, 2), "usage": raw.get("usage"),
              "finish_reason": choice.get("finish_reason"),
              "teacher_text": content}
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(raw_path)
    inflight.unlink(missing_ok=True)
    return parse_saved_audit(case_id, raw_path, prompt_hash)


def main() -> None:
    global OUT, OUTPUT_CAP, REASONING_EFFORT, MODEL, MANIFEST, TRACE_DIR, TRACE_SUFFIX, NO_OUTPUT_CAP, ENDPOINT
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--provider", choices=["api"], default="api")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"])
    ap.add_argument("--output-dir", default="sft_audit14_v1")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--case-manifest", type=Path, default=MANIFEST)
    ap.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    ap.add_argument("--trace-suffix", default=TRACE_SUFFIX)
    ap.add_argument("--no-output-cap", action="store_true")
    args = ap.parse_args()
    workers = args.workers or paths.CONCURRENCY
    if not 1000 <= args.max_tokens <= 16000:
        raise ValueError("max-tokens outside 1000..16000")
    if "/" in args.output_dir or args.output_dir.startswith("."):
        raise ValueError("output-dir must be a simple results directory name")
    OUT = RUN / "results" / args.output_dir
    OUTPUT_CAP = args.max_tokens
    REASONING_EFFORT = args.reasoning_effort
    MODEL = args.model
    MANIFEST = args.case_manifest
    TRACE_DIR = args.trace_dir
    TRACE_SUFFIX = args.trace_suffix
    NO_OUTPUT_CAP = args.no_output_cap
    ids = cases()
    meta = pool_rows()
    estimates = [estimate(cid, meta[cid]) for cid in ids]
    total = {"cases": len(ids), "model": MODEL, "workers": workers,
             "provider": args.provider,
             "reasoning_effort": REASONING_EFFORT,
             "input_token_conservative_proxy": sum(x["input_token_conservative_proxy"] for x in estimates),
             "output_token_scenario": OUTPUT_CAP * len(ids),
             "output_cap_sent": not NO_OUTPUT_CAP,
             "usd_reference_at_output_scenario": round(sum(x["usd_reference_ceiling"] for x in estimates), 4)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pre_run_estimate.json").write_text(json.dumps({"total": total, "cases": estimates}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"estimate": total, "execute": args.execute}, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    available_keys = provider_keys(args.provider)
    semaphores = {name: threading.Semaphore(workers) for name, _, _ in available_keys}
    result = {}
    checkpoint_dir = OUT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for index, cid in enumerate(ids):
            route, api_key, _ = available_keys[index % len(available_keys)]
            future = pool.submit(request, cid, meta[cid], api_key, args.timeout,
                                 semaphores[route], route)
            futures[future] = cid
        for future in concurrent.futures.as_completed(futures):
            cid = futures[future]
            try:
                item = future.result()
            except Exception as exc:
                item = {"case_id": cid, "error": f"local_exception: {type(exc).__name__}: {exc}"}
            result[cid] = item
            checkpoint = checkpoint_dir / f"{cid}.json"
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps(item, ensure_ascii=False) + "\n")
            temporary.replace(checkpoint)
            print(json.dumps({"case_id": cid, "score": item.get("score"), "action": item.get("action"),
                              "error": item.get("error"), "usage": item.get("usage")}, ensure_ascii=False), flush=True)
    ordered = [result[cid] for cid in ids]
    (OUT / "audit.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ordered))
    summary = {"cases": len(ids), "completed": sum("score" in x for x in ordered),
               "errors": [x for x in ordered if "error" in x],
               "score_distribution": dict(Counter(x["score"] for x in ordered if "score" in x)),
               "action_distribution": dict(Counter(x["action"] for x in ordered if "action" in x)),
               "training_approved_changed": False}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
