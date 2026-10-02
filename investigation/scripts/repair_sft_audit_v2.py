#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from nautil_common import paths

ROOT = paths.DATA
RUN = paths.RUN
OUT = RUN / "results/sft_audit_repair_v2"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nautil_common.case_contract import package_digest, validate_case
from generate_ab_llm_v1 import opening_brief
from build_sft_evidence_demo import EvidenceEnvironment
from validate_hypothesis_ledger_v1 import validate as validate_trace

SYSTEM = """You are repairing one review-only Nautil multi-round investigation, starting from the ORIGINAL +3 trace, not the previous final-only edit. The trusted report and audit issues are editor-only. Return exactly one JSON object with case_id, new_evidence, operations, issue_resolutions, and editor_summary. Prefer exact, local operations: replace_text on a unique old substring in a round_content, reason or final; set_fetch on a numbered round when evidence needs to move or be added. Edit the earliest affected round and every later hypothesis-ledger line whose before/after state depends on it. Keep the same declared stable H IDs and definitions unless an audit issue specifically requires a correction, and preserve all valid investigation content. A round's note and state may use only E1 plus tool evidence returned in PRIOR rounds. Each changed factual claim and changed hypothesis state must cite supporting E IDs visible at that prefix. The reason for a tool call must be a question or discriminating check, not knowledge of unseen tool output. Do not fabricate tool returns; the compiler reconstructs them. Use operations shaped like {"op":"replace_text","target":"round_content","round":2,"old":"exact old phrase","new":"corrected phrase [E2.1]"} or {"op":"set_fetch","round":2,"old_ids":["E2.1"],"new_ids":["E2.1","E3.1"],"reason":"Check the relevant observation."}. A set_fetch operation has old_ids, new_ids and reason. A replace_text operation has target (round_content, reason, or final), round when applicable, old exact substring, and new replacement text. If an essential observation is genuinely absent from the package, you may add up to three new_evidence entries using reserved IDs and exact factual quotes from the supplied official_analysis_verbatim, or exact host expert-tool-output strings. Each entry needs evidence_id, neutral_title, kind, source_id, source_field and source_quote. Do not use the official conclusion as a source quote, and do not expose an editorial answer as a fact. The compiler will verify exact source location and add the quote verbatim to a new package; include a set_fetch operation before any new ID is cited. If a claim cannot be grounded in the original source, remove or qualify it, or return an empty operations list with a clear unfixable reason. Every issue_resolutions entry must be {id,status,explanation}, using each supplied audit issue ID exactly once; status must be fixed, partially_fixed, not_fixable, or already_acceptable. Keep active multi-round evidence selection, hypothesis progression, and an honest final causal answer. Output JSON only; no markdown and no full rewritten transcript."""


def compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def reserved_ids(package: dict) -> list[str]:
    major = max(int(match.group(1)) for item in package["evidence_items"]
                if (match := re.fullmatch(r"E([0-9]+)\.[0-9]+", item["evidence_id"])))
    return [f"E{major+1}.{n}" for n in range(1, 4)]


def _host_strings(value: object, path: str = "result.calls") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, list):
        return [entry for index, child in enumerate(value)
                for entry in _host_strings(child, f"{path}[{index}]")]
    if isinstance(value, dict):
        return [entry for key, child in value.items()
                for entry in _host_strings(child, f"{path}.{key}")]
    return []


def source_quote_location(record: dict, proposal: dict) -> dict:
    field = proposal.get("source_field")
    quote = proposal.get("source_quote")
    if not isinstance(quote, str) or len(quote.strip()) < 20:
        raise ValueError("new evidence requires an exact factual quote of at least 20 characters")
    row = record["pool_row"]
    if row["family"] == "host":
        if field != "host_expert_tool_output":
            raise ValueError("host additions must come from expert-tool output, not the closing report")
        source_path = paths.DATASET / row["split"] / row["case_id"] / "trajectory/trajectory.json"
        trajectory = json.loads(source_path.read_text())
        matches = [(path, text.index(quote)) for path, text in _host_strings(trajectory["result"]["calls"])
                   if quote in text]
    else:
        if field != "official_analysis_verbatim":
            raise ValueError("report additions must use official_analysis_verbatim factual text")
        source_path = ROOT / row["reference_path"]
        reference = json.loads(source_path.read_text())
        text = str(reference.get(field) or "")
        matches = [(field, match.start()) for match in re.finditer(re.escape(quote), text)]
    if len(matches) != 1:
        raise ValueError(f"new evidence quote must occur exactly once in the allowed original source; found {len(matches)}")
    return {"source_path": str(source_path.relative_to(ROOT)), "source_field": matches[0][0],
            "char_offset": matches[0][1], "quote_sha256": sha256_bytes(quote.encode()),
            "source_quote": quote}


def add_evidence(record: dict, package: dict, additions: object) -> tuple[dict, list[dict]]:
    if not isinstance(additions, list) or len(additions) > 3:
        raise ValueError("new_evidence must be a list of at most three items")
    new_package = copy.deepcopy(package)
    allowed_ids = reserved_ids(package)
    existing_source_ids = {item["source_id"] for item in package["evidence_items"]}
    provenance = []
    for index, entry in enumerate(additions):
        if not isinstance(entry, dict) or entry.get("evidence_id") != allowed_ids[index]:
            raise ValueError("new evidence ID must match the reserved sequential ID")
        title = entry.get("neutral_title")
        kind = entry.get("kind")
        source_id = entry.get("source_id")
        if not isinstance(title, str) or len(title.strip()) < 8 or not isinstance(kind, str) or not kind.strip():
            raise ValueError("new evidence needs a neutral title and kind")
        if source_id not in existing_source_ids:
            raise ValueError("new evidence source_id must be one already declared in the package")
        location = source_quote_location(record, entry)
        quote = location["source_quote"]
        new_package["evidence_items"].append({"evidence_id": entry["evidence_id"],
            "neutral_title": title.strip(), "kind": kind.strip(),
            "source_id": source_id, "text": quote})
        provenance.append({"evidence_id": entry["evidence_id"], **location})
    if additions:
        new_package["package_hash"] = package_digest(new_package)
        validate_case(new_package)
    return new_package, provenance


def replace_exact(text: str, old: object, new: object, target: str) -> str:
    if not isinstance(old, str) or not old or not isinstance(new, str) or not new.strip():
        raise ValueError(f"{target}: replace_text needs nonempty old/new strings")
    if text.count(old) != 1:
        raise ValueError(f"{target}: old substring must occur exactly once, found {text.count(old)}")
    return text.replace(old, new, 1)


def compile_patch(record: dict, proposal: dict, raw_path: Path) -> tuple[dict, list[dict], list[dict]]:
    cid = record["case_id"]
    if proposal.get("case_id") != cid:
        raise ValueError("patch case ID mismatch")
    operations = proposal.get("operations")
    if not isinstance(operations, list) or len(operations) > 40:
        raise ValueError("operations must be a list with at most 40 local edits")
    issue_ids = {issue["id"] for issue in record["issues"]}
    resolutions = proposal.get("issue_resolutions")
    if not isinstance(resolutions, list) or {x.get("id") for x in resolutions if isinstance(x, dict)} != issue_ids:
        raise ValueError("every prior audit issue must have exactly one disposition")
    if len(resolutions) != len(issue_ids) or any(x.get("status") not in
       {"fixed", "partially_fixed", "not_fixable", "already_acceptable"} for x in resolutions):
        raise ValueError("invalid issue disposition")
    original_path = ROOT / record["original_trace_path"]
    if sha256_bytes(original_path.read_bytes()) != record["original_trace_sha256"]:
        raise ValueError("original trace changed since preflight")
    original = json.loads(original_path.read_text())
    package = json.loads((ROOT / record["package_path"]).read_text())
    augmented, evidence_sources = add_evidence(record, package, proposal.get("new_evidence") or [])
    trace = copy.deepcopy(original)
    trace["review_status"] = "audit_repair_v2_requires_new_blind_score"
    trace["training_approved"] = False
    changes = []
    rounds = (len(trace["messages"]) - 2) // 2
    if not operations and not evidence_sources:
        raise ValueError("no local repair proposed; record as unfixable or already acceptable")
    for number, operation in enumerate(operations, 1):
        if not isinstance(operation, dict):
            raise ValueError("operation is not a JSON object")
        kind = operation.get("op")
        if kind == "replace_text":
            target = operation.get("target")
            if target == "final":
                index = len(trace["messages"]) - 1
                before = trace["messages"][index]["content"]
                trace["messages"][index]["content"] = replace_exact(before, operation.get("old"),
                                                                       operation.get("new"), target)
            elif target in {"round_content", "reason"}:
                turn = operation.get("round")
                if not isinstance(turn, int) or not 1 <= turn <= rounds:
                    raise ValueError("invalid round number in replace_text")
                index = 2*turn-1
                if target == "round_content":
                    before = trace["messages"][index]["content"]
                    trace["messages"][index]["content"] = replace_exact(before, operation.get("old"),
                                                                           operation.get("new"), target)
                else:
                    args = trace["messages"][index]["tool_calls"][0]["function"]["arguments"]
                    before = args["reason"]
                    args["reason"] = replace_exact(before, operation.get("old"), operation.get("new"), target)
            else:
                raise ValueError("replace_text target must be round_content, reason, or final")
            changes.append({"number": number, "op": kind, "target": target,
                            "round": operation.get("round"),
                            "old_sha256": sha256_bytes(before.encode()),
                            "new_sha256": sha256_bytes((trace["messages"][index]["content"] if target != "reason"
                                                         else args["reason"]).encode())})
        elif kind == "set_fetch":
            turn = operation.get("round")
            if not isinstance(turn, int) or not 1 <= turn <= rounds:
                raise ValueError("invalid round number in set_fetch")
            args = trace["messages"][2*turn-1]["tool_calls"][0]["function"]["arguments"]
            old_ids = args["evidence_ids"]
            new_ids = operation.get("new_ids")
            reason = operation.get("reason")
            if operation.get("old_ids") != old_ids or not isinstance(new_ids, list) or not 1 <= len(new_ids) <= 12:
                raise ValueError("set_fetch must give exact old_ids and 1..12 new_ids")
            if len(set(new_ids)) != len(new_ids) or not all(isinstance(x, str) for x in new_ids):
                raise ValueError("set_fetch has duplicate or malformed IDs")
            if not isinstance(reason, str) or len(reason.strip()) < 10:
                raise ValueError("set_fetch needs a substantive reason")
            prior_reason = args["reason"]
            args["evidence_ids"] = new_ids
            args["reason"] = reason
            changes.append({"number": number, "op": kind, "round": turn,
                            "old_ids": old_ids, "new_ids": new_ids,
                            "reason_changed": reason != prior_reason})
        else:
            raise ValueError(f"unsupported patch operation: {kind}")
    for turn in range(1, rounds+1):
        note = trace["messages"][2*turn-1]["content"]
        reason = trace["messages"][2*turn-1]["tool_calls"][0]["function"]["arguments"]["reason"]
        marker = "## Next discriminating check\n"
        if marker in note:
            before, _, old_check = note.partition(marker)
            if old_check.strip() != reason:
                trace["messages"][2*turn-1]["content"] = before + marker + reason
                changes.append({"op": "auto_sync_next_check_to_native_reason", "round": turn,
                                "old_check": old_check.strip(), "new_check": reason})
    original_final = original["messages"][-1]["content"]
    revised_final = trace["messages"][-1]["content"]
    original_marker = re.match(r"CASE (?:NOT )?CLOSED", original_final)
    if original_marker and not re.match(r"CASE (?:NOT )?CLOSED", revised_final):
        if re.search(r"\bCASE (?:NOT )?CLOSED\b", revised_final):
            raise ValueError("edited final has a misplaced or contradictory closure marker")
        trace["messages"][-1]["content"] = original_marker.group() + " — " + revised_final.lstrip()
        changes.append({"op": "auto_restore_original_closure_marker_in_final",
                        "marker": original_marker.group()})
    if evidence_sources:
        package_path = OUT / "packages" / f"{cid}.json"
        package_path.parent.mkdir(parents=True, exist_ok=True)
        package_path.write_text(json.dumps(augmented, ensure_ascii=False, indent=2) + "\n")
        trace["package_hash"] = augmented["package_hash"]
        trace.setdefault("provenance", {})["new_package"] = str(package_path.relative_to(ROOT))
        trace["messages"][0]["content"] = opening_brief(augmented)
    env = EvidenceEnvironment(augmented)
    for turn in range(1, rounds+1):
        assistant = trace["messages"][2*turn-1]
        call = assistant["tool_calls"][0]
        args = call["function"]["arguments"]
        trace["messages"][2*turn] = {"role": "tool", "tool_call_id": call["id"],
            "name": "request_evidence", "content": env.fetch(args["evidence_ids"]), "loss": False}
    trace["sample_id"] = str(original["sample_id"]) + "-audit-repair-v2"
    trace["origin"] = "gpt6sol_audit_driven_local_patch_v2"
    trace.setdefault("provenance", {})["repair_teacher_raw"] = str(raw_path.relative_to(ROOT))
    trace["provenance"]["original_trace_path"] = record["original_trace_path"]
    trace["provenance"]["original_trace_sha256"] = record["original_trace_sha256"]
    trace["provenance"]["source_evidence_extracts"] = evidence_sources
    trace["provenance"]["issue_resolutions"] = resolutions
    validate_trace(trace)
    return trace, changes, evidence_sources


def make_editor(record: dict) -> dict:
    trace = json.loads((ROOT / record["original_trace_path"]).read_text())
    package = json.loads((ROOT / record["package_path"]).read_text())
    row = record["pool_row"]
    if row["family"] == "host":
        source_path = paths.DATASET / row["split"] / row["case_id"] / "trajectory/trajectory.json"
        source = json.loads(source_path.read_text())
        trusted = {"closing_report": source["result"].get("content") or ""}
    else:
        reference = json.loads((ROOT / row["reference_path"]).read_text())
        trusted = {"official_analysis_verbatim": reference.get("official_analysis_verbatim"),
                   "official_conclusion_verbatim": reference.get("official_conclusion_verbatim"),
                   "expected_closure": reference.get("expected_closure")}
    return {"case_id": record["case_id"], "original_trace": {"tools": trace["tools"],
             "messages": trace["messages"]},
            "frozen_package": package, "trusted_reference_editor_only": trusted,
            "audit_issues": record["issues"],
            "preliminary_issue_assessment": record.get("issue_assessment"),
            "earliest_affected_round": record["earliest_affected_round"],
            "reserved_new_evidence_ids": reserved_ids(package)}


def attach_pool_rows(records: list[dict]) -> list[dict]:
    pool = {x["case_id"]: x for x in json.loads((RUN / "docs/SFT_POOL_AUDIT_20260924.json").read_text())["rows"]}
    return [{**record, "pool_row": pool[record["case_id"]]} for record in records]
