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
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from nautil_common import paths

ROOT = paths.DATA
RUN = paths.RUN
SRC = RUN / "results/sft_scale60_v1"
REPEATED = RUN / "results/sft_scale60_repeated_fetch_v1"
OUT = RUN / "results/sft_scale60_evidence_ledger_v2"
ROUTES = paths.CONFIGS / "api_routes.json"
EID = re.compile(r"\bE[1-9][0-9]*\.[1-9][0-9]*\b")
HID = re.compile(r"\bH[1-9][0-9]*\b")
STATUSES = {"open", "favored", "weakened"}
from validate_hypothesis_ledger_v1 import validate as validate_ledger
from generate_ab_llm_v1 import load_source

SYSTEM = """You are grounding an existing multi-round investigator draft in its actual tool evidence. Return exactly one JSON object with hypotheses and rounds. hypotheses is an array of {id, claim}; keep the H IDs from the first note and define stable competing claims. Each round is {round, note_citation_insertions, states}. note_citation_insertions is an array of {anchor, evidence_ids}: anchor must be an EXACT, UNIQUE substring of that round's original_note, and the compiler inserts the cited IDs immediately after it without changing any original words. Ground every material observed fact in the note with an ID; do not add a citation that merely has the same topic but does not support the claim. Each states array contains one {id, status, basis} for every H ID, with status exactly open, favored or weakened. A status change must cite one or more actual supporting or contradicting E IDs in basis. Prefer provisional open status when evidence does not discriminate, but a reasoned favored/weakened state is allowed without proof beyond doubt. For round N use only opening E1 and tool items returned AFTER rounds 1 through N-1. The tool results listed after round N must NOT support that round's note or state. The original case notes and fetches are fixed; if a material claim cannot be grounded by visible evidence, mark that round's states conservatively and report an error in an optional problems array. Never use later tool returns, final answer, or a report hidden from the investigator to backfill an earlier round. Do not cite an ID unless its exact text supports the statement. Return JSON only."""


def env() -> dict[str, str]:
    result = {}
    for line in paths.ENV_FILE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            result[k.strip()] = v.strip().strip('"').strip("'")
    return result


def inputs() -> list[dict]:
    status = [json.loads(line) for line in (SRC / "status.jsonl").read_text().splitlines() if line.strip()]
    original = [x for x in status if x["status"] == "compiled_review_draft"]
    routes_by_case = {x["case_id"]: x["route"] for x in status}
    repeated = [json.loads(line) for line in (REPEATED / "status.jsonl").read_text().splitlines() if line.strip()]
    original.extend({**x, "route": routes_by_case[x["case_id"]]}
                    for x in repeated if x["status"] == "compiled_review_draft")
    return original


def make_editor(trace: dict) -> tuple[list[dict], dict]:
    turns = []
    _, package, _, _ = load_source(trace["case_id"])
    opening_evidence = [item for item in package["evidence_items"] if item["evidence_id"].startswith("E1.")]
    disclosed = {item["evidence_id"] for item in opening_evidence}
    for n, pos in enumerate(range(1, len(trace["messages"]) - 1, 2), 1):
        assistant = trace["messages"][pos]
        note = assistant["content"]
        reason = assistant["tool_calls"][0]["function"]["arguments"]["reason"]
        citations = sorted(set(EID.findall(note)))
        if set(citations) - disclosed:
            raise ValueError(f"source note {n} cites unread evidence")
        returned_result = json.loads(trace["messages"][pos + 1]["content"])
        returned = returned_result["evidence_items"]
        turns.append({"round": n, "original_note": note, "request_reason": reason,
                      "visible_before_ids": sorted(disclosed), "note_citations": citations,
                      "tool_return_after": returned,
                      "already_delivered_after": returned_result.get("already_delivered", [])})
        disclosed.update(item["evidence_id"] for item in returned)
    hids = sorted(set(HID.findall(turns[0]["original_note"])))
    if len(hids) < 2:
        raise ValueError("source first note lacks H1/H2")
    return turns, {"case_id": trace["case_id"], "hypothesis_ids": hids,
                   "opening_evidence": opening_evidence, "turns": turns}


def insert_citations(original: str, insertions: object, visible: set[str], number: int) -> str:
    if not isinstance(insertions, list):
        raise ValueError(f"round {number}: note_citation_insertions must be a list")
    positions = []
    for item in insertions:
        if not isinstance(item, dict):
            raise ValueError(f"round {number}: bad citation insertion")
        anchor, ids = item.get("anchor"), item.get("evidence_ids")
        if not isinstance(anchor, str) or not anchor:
            raise ValueError(f"round {number}: empty citation anchor")
        matches = list(re.finditer(re.escape(anchor), original, re.IGNORECASE))
        if len(matches) != 1:
            raise ValueError(f"round {number}: citation anchor not exact and unique")
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or any(eid not in visible for eid in ids):
            raise ValueError(f"round {number}: citation includes unread or malformed ID")
        positions.append((matches[0].end(), ids))
    if len({position for position, _ in positions}) != len(positions):
        raise ValueError(f"round {number}: multiple citations at same anchor")
    annotated = original
    for position, ids in sorted(positions, reverse=True):
        annotated = annotated[:position] + " [" + ", ".join(ids) + "]" + annotated[position:]
    if number > 1 and not EID.search(annotated):
        raise ValueError(f"round {number}: fact-bearing update has no evidence citation")
    return annotated


def compile_ledger_v2(trace: dict, turns: list[dict], proposed: dict, raw_path: Path,
                      source_path: Path, origin: str) -> dict:
    expected_ids = sorted(set(HID.findall(turns[0]["original_note"])))
    definitions = proposed.get("hypotheses")
    if not isinstance(definitions, list) or sorted(x.get("id") for x in definitions) != expected_ids:
        raise ValueError("hypothesis definitions differ from first turn")
    claims = {item["id"]: str(item.get("claim") or "").strip() for item in definitions}
    if any(len(claims[hid]) < 8 or EID.search(claims[hid]) for hid in expected_ids):
        raise ValueError("empty or evidence-specific stable claim")
    rounds = proposed.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != len(turns):
        raise ValueError("wrong number of hypothesis rounds")
    result = copy.deepcopy(trace)
    previous = {hid: "open" for hid in expected_ids}
    for turn, item in zip(turns, rounds):
        number = turn["round"]
        if item.get("round") != number:
            raise ValueError(f"round number mismatch at {number}")
        visible = set(turn["visible_before_ids"])
        note = insert_citations(turn["original_note"], item.get("note_citation_insertions"), visible, number)
        states = item.get("states")
        if not isinstance(states, list) or sorted(x.get("id") for x in states) != expected_ids:
            raise ValueError(f"round {number}: missing hypothesis state")
        by_id = {state["id"]: state for state in states}
        lines = ["## Current hypotheses"]
        lines.extend(f"{hid} — {claims[hid]}" for hid in expected_ids)
        lines += ["", "## Evidence and hypothesis update", note, "", "## Hypothesis ledger"]
        for hid in expected_ids:
            state = by_id[hid]
            status = state.get("status")
            basis = str(state.get("basis") or "").strip()
            basis_refs = set(EID.findall(basis))
            if status not in STATUSES or len(basis) < 12:
                raise ValueError(f"round {number} {hid}: bad status/basis")
            if basis_refs - visible:
                raise ValueError(f"round {number} {hid}: basis cites unread evidence")
            if number > 1 and status != previous[hid] and not basis_refs:
                raise ValueError(f"round {number} {hid}: status changed without cited evidence")
            lines.append(f"{hid} [{previous[hid]} → {status}] — {basis}")
            previous[hid] = status
        lines += ["", "## Next discriminating check", turn["request_reason"]]
        result["messages"][2 * number - 1]["content"] = "\n".join(lines)
    result["sample_id"] = str(trace["sample_id"]) + "-evidence-ledger-v2"
    result["origin"] = origin
    result["review_status"] = ("teacher_grounding_warning_requires_semantic_review"
                               if proposed.get("problems") else "evidence_grounded_ledger_requires_semantic_review")
    result["training_approved"] = False
    result.setdefault("provenance", {})["ledger_teacher_raw"] = str(raw_path.relative_to(ROOT))
    result["provenance"]["source_internal_trace"] = str(source_path.relative_to(ROOT))
    if proposed.get("problems"):
        result["provenance"]["ledger_grounding_warnings"] = proposed["problems"]
    return result


def estimate(editor: dict) -> dict:
    size = len(SYSTEM) + len(json.dumps(editor, ensure_ascii=False, separators=(",", ":")))
    input_proxy = math.ceil(size / 2.5 * 1.2)
    return {"input_proxy": input_proxy, "output_assumption_not_cap": 5000,
            "usd_reference_expected": round((2*input_proxy + 10*5000)/1_000_000, 4),
            "usd_reference_if_12000_output": round((2*input_proxy + 10*12000)/1_000_000, 4)}


def compile_saved_ledger(item: dict, raw_path: Path, trace: dict, turns: list[dict],
                         source_path: Path, prompt_hash: str, *, reused: bool) -> dict:
    cid = item["case_id"]
    raw = json.loads(raw_path.read_text())
    if raw.get("case_id") != cid or raw.get("prompt_sha256") != prompt_hash:
        return {"case_id": cid, "route": item["route"], "status": "cached_raw_mismatch",
                "error": "cached case ID or prompt hash differs from this evidence-aware editor input"}
    result = {"case_id": cid, "route": raw.get("route") or item["route"],
              "raw": str(raw_path.relative_to(ROOT)), "finish_reason": raw.get("finish_reason"),
              "usage": raw.get("usage"), "elapsed_seconds": raw.get("elapsed_seconds"),
              "reused_saved_teacher_response": reused}
    if raw.get("finish_reason") == "length":
        return result | {"status": "truncated_output"}
    try:
        proposed = json.loads(raw.get("teacher_text") or "")
        structured = compile_ledger_v2(trace, turns, proposed, raw_path,
                                       source_path=source_path,
                                       origin=f"gpt6sol_{result['route']}_evidence_ledger_v2")
        structured["training_approved"] = False
        metrics = validate_ledger(structured)
        target = OUT / "traces" / f"{cid}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(structured, ensure_ascii=False) + "\n")
        temporary.replace(target)
        return result | {"status": "structured_review_draft", "trace": str(target.relative_to(ROOT)),
                         "tool_calls": metrics["tool_calls"], "state_updates": metrics["explicit_state_updates"],
                         "grounding_warnings": len(structured["provenance"].get("ledger_grounding_warnings") or [])}
    except Exception as exc:
        return result | {"status": "ledger_rejected", "error": f"{type(exc).__name__}: {str(exc)[:260]}"}


def run(item: dict, spec: dict, key: str, sem: threading.Semaphore,
        retry_uncertain: bool = False) -> dict:
    cid = item["case_id"]
    source_path = ROOT / item["trace"]
    try:
        trace = json.loads(source_path.read_text())
        turns, editor = make_editor(trace)
    except Exception as exc:
        return {"case_id": cid, "route": item["route"], "status": "source_error", "error": str(exc)[:250]}
    user_text = json.dumps(editor, ensure_ascii=False)
    prompt_hash = hashlib.sha256((SYSTEM + user_text).encode()).hexdigest()
    raw_path = OUT / "raw" / f"{cid}.json"
    if raw_path.exists():
        return compile_saved_ledger(item, raw_path, trace, turns, source_path, prompt_hash, reused=True)
    inflight = OUT / "inflight" / f"{cid}.json"
    if inflight.exists() and not retry_uncertain:
        return {"case_id": cid, "route": item["route"],
                "status": "uncertain_inflight_needs_manual_retry",
                "error": "previous process stopped after dispatch but before saving a response"}
    body = {"model": spec["wire_model"], "messages": [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text}], "response_format": {"type": "json_object"}}
    req = urllib.request.Request(spec["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    started = time.monotonic()
    with sem:
        inflight.parent.mkdir(parents=True, exist_ok=True)
        inflight.write_text(json.dumps({"case_id": cid, "route": item["route"],
            "prompt_sha256": prompt_hash, "started_unix": time.time()}) + "\n")
        try:
            with urllib.request.urlopen(req, timeout=240) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            inflight.unlink(missing_ok=True)
            return {"case_id": cid, "route": item["route"], "status": "http_error", "http_status": exc.code,
                    "error": exc.read(300).decode(errors="replace")[:250]}
        except Exception as exc:
            inflight.unlink(missing_ok=True)
            return {"case_id": cid, "route": item["route"], "status": "transport_error", "error": f"{type(exc).__name__}: {str(exc)[:250]}"}
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    if isinstance(content, list):
        content = "".join(x.get("text", "") for x in content if isinstance(x, dict))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"case_id": cid, "route": item["route"], "model": spec["wire_model"],
        "model_returned": data.get("model"), "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"), "elapsed_seconds": round(time.monotonic()-started, 2),
        "prompt_sha256": prompt_hash,
        "teacher_text": content}, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(raw_path)
    inflight.unlink(missing_ok=True)
    return compile_saved_ledger(item, raw_path, trace, turns, source_path, prompt_hash, reused=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--retry-uncertain", action="store_true")
    args = ap.parse_args()
    items = inputs()
    specs = {x["name"]: x for x in json.loads(ROUTES.read_text())["routes"]}
    estimates = []
    for item in items:
        try:
            trace = json.loads((ROOT / item["trace"]).read_text())
            _, editor = make_editor(trace)
            estimates.append({"case_id": item["case_id"], "route": item["route"], **estimate(editor)})
        except Exception as exc:
            estimates.append({"case_id": item["case_id"], "route": item["route"], "error": str(exc)[:200]})
    summary = {"cases": len(items), "by_route": dict(Counter(x["route"] for x in items)),
               "usd_reference_expected": round(sum(x.get("usd_reference_expected", 0) for x in estimates), 4),
               "usd_reference_if_12000_output_each": round(sum(x.get("usd_reference_if_12000_output", 0) for x in estimates), 4),
               "output_cap_sent": False, "pricing_note": "OpenAI $2/$10 per million reference; actual API prices unverified."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pre_run_estimate.json").write_text(json.dumps({"summary": summary, "cases": estimates}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"estimate": summary, "execute": args.execute}, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    values = env()
    effective = {name: spec["max_concurrent"]
                 for name, spec in specs.items()}
    sems = {name: threading.Semaphore(limit) for name, limit in effective.items()}
    for item in items:
        if not values.get(specs[item["route"]]["key_env"]):
            raise ValueError(f"missing key for {item['route']}")
    results = {}
    checkpoint_dir = OUT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(items), sum(effective.values()))) as pool:
        futures = {pool.submit(run, item, specs[item["route"]], values[specs[item["route"]]["key_env"]],
                               sems[item["route"]], args.retry_uncertain): item["case_id"] for item in items}
        for future in concurrent.futures.as_completed(futures):
            cid = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"case_id": cid, "status": "local_error", "error": str(exc)[:260]}
            results[cid] = result
            checkpoint = checkpoint_dir / f"{cid}.json"
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n")
            temporary.replace(checkpoint)
            print(json.dumps({k: result.get(k) for k in ("case_id", "route", "status", "error", "finish_reason", "tool_calls")}, ensure_ascii=False), flush=True)
    ordered = [results[item["case_id"]] for item in items]
    (OUT / "status.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ordered))
    final = {"cases": len(items), "statuses": dict(Counter(x["status"] for x in ordered)),
             "effective_route_limits": effective, "training_approved_changed": False}
    (OUT / "summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": final}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
