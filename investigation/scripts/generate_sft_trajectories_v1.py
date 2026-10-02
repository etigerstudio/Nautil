#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from nautil_common import paths

ROOT = paths.DATA
RUN = paths.RUN
from generate_ab_llm_v1 import load_source, compile_plan
from validate_sft_format_v1 import validate

SYSTEM = """You are an editor constructing one auditable, single-investigator multi-round SFT trajectory from a frozen evidence package and a trusted report. The trusted report is editor-only, never part of the investigator's initial view. Return exactly one JSON object with keys rounds and final. Every round MUST have a nonempty note, 1-12 NEW fetch_ids, and a reason: never put a no-fetch conclusion in rounds; put it in final. Start with stable H1/H2 and optionally H3. Every subsequent note updates each live hypothesis from evidence already returned, distinguishing supported, weakened and still open. A note MUST NOT cite or assert the contents of an ID that will be fetched in that same round or a later round. It may use only E1 and items returned in PRIOR rounds; the title-only index does not reveal item content. The reason is the discriminating question the requested IDs can answer, without asserting their unseen results. Choose coherent groups, never repeat IDs, and never invent tool output. Use 3-12 fetch rounds for report cases and 4-16 for host cases, only as many as needed; do not impose two rounds on a complex host. If the reviewer-only expected_closure starts 'undetermined:', the final MUST start CASE NOT CLOSED and state the target cause is unresolved; if it starts 'determined:', the final may start CASE CLOSED only if every decisive factual claim is disclosed by E1 or fetched IDs. Before writing final, ensure every E ID you cite was fetched in an earlier round (or was in E1); if not, include it in an appropriate earlier fetch. Never equate an agency's administrative closure with causal certainty. Output JSON only. The compiler, not you, produces tool messages from the frozen package."""
PRICE_IN = 2.0
PRICE_OUT = 10.0


def compact(x: object) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def env() -> dict[str, str]:
    result = {}
    for line in paths.ENV_FILE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            result[k.strip()] = v.strip().strip('"').strip("'")
    return result


def estimate(editor_user: str) -> dict:
    input_proxy = math.ceil((len(SYSTEM) + len(editor_user)) / 2.5 * 1.2)
    expected_output = 6000
    return {"input_token_proxy": input_proxy, "output_tokens_assumption_not_cap": expected_output,
            "usd_reference_expected": round((PRICE_IN * input_proxy + PRICE_OUT * expected_output) / 1_000_000, 4),
            "usd_reference_if_16000_output": round((PRICE_IN * input_proxy + PRICE_OUT * 16000) / 1_000_000, 4)}


def compile_saved(case_id: str, spec: dict, output: Path, raw_path: Path,
                  row: dict, package: dict, reference: dict, prompt_hash: str,
                  *, reused: bool) -> dict:
    raw = json.loads(raw_path.read_text())
    if raw.get("case_id") != case_id or raw.get("prompt_sha256") != prompt_hash:
        return {"case_id": case_id, "route": spec["name"], "status": "cached_raw_mismatch",
                "error": "cached case ID or prompt hash differs from this batch"}
    result = {"case_id": case_id, "route": raw.get("route") or spec["name"],
              "raw": str(raw_path.relative_to(ROOT)), "finish_reason": raw.get("finish_reason"),
              "usage": raw.get("usage"), "elapsed_seconds": raw.get("elapsed_seconds"),
              "reused_saved_teacher_response": reused}
    if raw.get("finish_reason") == "length":
        return result | {"status": "truncated_output"}
    try:
        plan = json.loads(raw.get("teacher_text") or "")
        trace = compile_plan(case_id, row, package, reference, plan, raw_path,
                             origin=f"gpt6sol_{result['route']}_scale_plan_v1")
        trace["review_status"] = "plan_draft_needs_explicit_ledger_and_semantic_review"
        trace["training_approved"] = False
        metrics = validate(trace)
        target = output / "traces" / f"{case_id}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(trace, ensure_ascii=False) + "\n")
        temporary.replace(target)
        return result | {"status": "compiled_review_draft", "trace": str(target.relative_to(ROOT)),
                         "rounds": metrics["tool_calls"], "fetched_items": metrics["fetched_items"]}
    except Exception as exc:
        return result | {"status": "compiler_or_json_rejected",
                         "error": f"{type(exc).__name__}: {str(exc)[:260]}"}


def call(case_id: str, spec: dict, key: str, output: Path, sem: threading.Semaphore,
         reuse_raw_path: str | None = None, retry_uncertain: bool = False) -> dict:
    try:
        row, package, reference, editor_user = load_source(case_id)
    except Exception as exc:
        return {"case_id": case_id, "route": spec["name"], "status": "input_error", "error": str(exc)[:240]}
    prompt_hash = hashlib.sha256((SYSTEM + editor_user).encode()).hexdigest()
    raw_path = output / "raw" / f"{case_id}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists() and reuse_raw_path:
        source = (ROOT / reuse_raw_path).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file():
            return {"case_id": case_id, "route": spec["name"], "status": "cache_source_error",
                    "error": "reuse_raw_path is missing or outside workspace"}
        shutil.copy2(source, raw_path)
    if raw_path.exists():
        return compile_saved(case_id, spec, output, raw_path, row, package, reference,
                             prompt_hash, reused=True)
    inflight = output / "inflight" / f"{case_id}.json"
    if inflight.exists() and not retry_uncertain:
        return {"case_id": case_id, "route": spec["name"],
                "status": "uncertain_inflight_needs_manual_retry",
                "error": "previous process stopped after dispatch but before saving a response"}
    body = {"model": spec["wire_model"],
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": editor_user}],
            "response_format": {"type": "json_object"}}
    req = urllib.request.Request(spec["base_url"].rstrip("/") + "/chat/completions",
            data=compact(body).encode(), method="POST",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    started = time.monotonic()
    with sem:
        inflight.parent.mkdir(parents=True, exist_ok=True)
        inflight.write_text(json.dumps({"case_id": case_id, "route": spec["name"],
            "prompt_sha256": prompt_hash, "started_unix": time.time()}) + "\n")
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                response_json = json.load(response)
        except urllib.error.HTTPError as exc:
            error = exc.read(500).decode(errors="replace")
            inflight.unlink(missing_ok=True)
            return {"case_id": case_id, "route": spec["name"], "status": "http_error",
                    "http_status": exc.code, "error": error[:260], "elapsed_seconds": round(time.monotonic()-started, 2)}
        except Exception as exc:
            inflight.unlink(missing_ok=True)
            return {"case_id": case_id, "route": spec["name"], "status": "transport_error",
                    "error": f"{type(exc).__name__}: {str(exc)[:220]}", "elapsed_seconds": round(time.monotonic()-started, 2)}
    choice = (response_json.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    if isinstance(content, list):
        content = "".join(x.get("text", "") for x in content if isinstance(x, dict))
    temporary = raw_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"case_id": case_id, "route": spec["name"], "model_requested": spec["wire_model"],
        "model_returned": response_json.get("model"), "finish_reason": choice.get("finish_reason"),
        "usage": response_json.get("usage"), "elapsed_seconds": round(time.monotonic()-started, 2),
        "prompt_sha256": prompt_hash,
        "teacher_text": content}, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(raw_path)
    inflight.unlink(missing_ok=True)
    return compile_saved(case_id, spec, output, raw_path, row, package, reference,
                         prompt_hash, reused=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--retry-uncertain", action="store_true",
                    help="Explicitly retry an in-flight request whose prior completion is unknown")
    args = ap.parse_args()
    manifest = json.loads(args.batch.read_text())
    route_cfg = json.loads((ROOT / manifest["route_config"]).read_text())
    routes = {x["name"]: x for x in route_cfg["routes"]}
    items = manifest["items"]
    if len({x["case_id"] for x in items}) != len(items):
        raise ValueError("duplicate case ID in batch")
    privacy_lane = manifest.get("privacy_lane")
    if privacy_lane in {"private_host_restricted", "public_nonmedical"}:
        families = {row["case_id"]: row["family"] for row in
                    json.loads((RUN / "docs/SFT_POOL_AUDIT_20260924.json").read_text())["rows"]}
        for item in items:
            family = families[item["case_id"]]
            if privacy_lane == "private_host_restricted" and (family != "host" or item["route"] != "api"):
                raise ValueError(f"privacy lane violation before network: {item['case_id']}")
            if privacy_lane == "public_nonmedical" and family == "host":
                raise ValueError(f"private host case entered a public route batch: {item['case_id']}")
    route_counts = {}
    for item in items:
        spec = routes[item["route"]]
        route_counts[spec["name"]] = route_counts.get(spec["name"], 0) + 1
    output = ROOT / manifest["output"]
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in items:
        try:
            _, _, _, editor_user = load_source(item["case_id"])
            cost = estimate(editor_user)
            rows.append({"case_id": item["case_id"], "route": item["route"], **cost})
        except Exception as exc:
            rows.append({"case_id": item["case_id"], "route": item["route"], "error": str(exc)[:200]})
    estimate_summary = {"cases": len(items), "routes": route_counts,
        "usd_reference_expected": round(sum(x.get("usd_reference_expected", 0) for x in rows), 4),
        "usd_reference_if_16000_output_each": round(sum(x.get("usd_reference_if_16000_output", 0) for x in rows), 4),
        "output_cap_sent": False, "pricing_note": "OpenAI public $2/$10 per million proxy; actual API prices may differ."}
    (output / "pre_run_estimate.json").write_text(json.dumps({"summary": estimate_summary, "cases": rows}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"estimate": estimate_summary, "execute": args.execute}, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    values = env()
    effective = {name: spec["max_concurrent"]
                 for name, spec in routes.items()}
    sems = {name: threading.Semaphore(limit) for name, limit in effective.items()}
    for item in items:
        if not values.get(routes[item["route"]]["key_env"]):
            raise ValueError(f"missing key for {item['route']}")
    results = {}
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(items), sum(effective.values()))) as pool:
        futures = {pool.submit(call, item["case_id"], routes[item["route"]],
                               values[routes[item["route"]]["key_env"]], output, sems[item["route"]],
                               item.get("reuse_raw_path"), args.retry_uncertain): item["case_id"]
                   for item in items}
        for future in concurrent.futures.as_completed(futures):
            cid = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"case_id": cid, "status": "local_error", "error": f"{type(exc).__name__}: {exc}"}
            results[cid] = result
            checkpoint = checkpoint_dir / f"{cid}.json"
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n")
            temporary.replace(checkpoint)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    ordered = [results[x["case_id"]] for x in items]
    (output / "status.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ordered))
    summary = {"cases": len(items), "compiled_review_drafts": sum(x["status"] == "compiled_review_draft" for x in ordered),
               "statuses": {status: sum(x["status"] == status for x in ordered) for status in sorted({x["status"] for x in ordered})},
               "training_approved_changed": False}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
