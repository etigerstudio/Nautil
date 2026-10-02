#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from nautil_common import paths

ROOT = paths.DATA
RUN = paths.RUN
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sft_evidence_demo import EvidenceEnvironment, opening_brief, validate_evidence_args

EID_RE = re.compile(r"\bE[1-9][0-9]*\.[1-9][0-9]*\b")
HID_RE = re.compile(r"\bH[1-9][0-9]*\b")
TOOL_CONTRACT = paths.CONFIGS / "sft_evidence_tool_v1.json"


def package_path(row: dict) -> Path:
    provenance = row.get("provenance") or {}
    path = provenance.get("new_package") or provenance.get("package")
    if not path:
        raise ValueError("sample has no package provenance path")
    return ROOT / path


def validate(row: dict) -> dict:
    if row.get("schema_version") != "nautil.sft.v1":
        raise ValueError("wrong schema version")
    prompt_ref = ROOT / str(row.get("system_prompt_ref") or "")
    if not prompt_ref.is_file():
        raise ValueError("frozen system prompt is missing")
    tools = json.loads(TOOL_CONTRACT.read_text())
    if row.get("tools") != tools:
        raise ValueError("tool contract mismatch")
    schema = tools[0]["function"]["parameters"]
    if schema.get("required") != ["evidence_ids", "reason"]:
        raise ValueError("request_evidence schema differs from the fixed contract")
    package = json.loads(package_path(row).read_text())
    if package["case_id"] != row.get("case_id") or package["package_hash"] != row.get("package_hash"):
        raise ValueError("case ID or package hash mismatch")
    messages = row.get("messages") or []
    if len(messages) < 4 or len(messages) % 2:
        raise ValueError("expected user + assistant/tool pairs + assistant final")
    if messages[0] != {"role": "user", "content": opening_brief(package), "loss": False}:
        raise ValueError("opening brief differs from source package")
    env = EvidenceEnvironment(package)
    initial_hypotheses: set[str] | None = None
    calls = 0
    for pos in range(1, len(messages) - 1, 2):
        assistant = messages[pos]
        tool_message = messages[pos + 1]
        if assistant.get("role") != "assistant" or assistant.get("loss") is not True:
            raise ValueError(f"message {pos}: assistant working turn must be trained")
        content = assistant.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"message {pos}: public working note is missing")
        hypotheses = set(HID_RE.findall(content))
        if initial_hypotheses is None:
            initial_hypotheses = hypotheses
            if len(initial_hypotheses) < 2:
                raise ValueError("initial working note needs at least two competing hypotheses")
        elif hypotheses - initial_hypotheses:
            raise ValueError(f"message {pos}: undeclared hypothesis IDs {sorted(hypotheses - initial_hypotheses)}")
        unread = set(EID_RE.findall(content)) - env.disclosed
        if unread:
            raise ValueError(f"message {pos}: note cites unread evidence {sorted(unread)}")
        calls_here = assistant.get("tool_calls") or []
        if len(calls_here) != 1 or calls_here[0].get("type") != "function":
            raise ValueError(f"message {pos}: expected exactly one native tool call")
        call = calls_here[0]
        if call.get("function", {}).get("name") != "request_evidence":
            raise ValueError(f"message {pos}: unexpected tool")
        args = call["function"]["arguments"]
        if not isinstance(args, dict):
            raise ValueError(f"message {pos}: Qwen v1 source format needs object-valued tool arguments")
        validate_evidence_args(args)
        expected_content = env.fetch(args["evidence_ids"])
        if tool_message != {"role": "tool", "tool_call_id": call["id"],
                            "name": "request_evidence", "content": expected_content, "loss": False}:
            raise ValueError(f"message {pos + 1}: tool result is not the deterministic package response")
        calls += 1
    final = messages[-1]
    if final.get("role") != "assistant" or final.get("loss") is not True or final.get("tool_calls"):
        raise ValueError("final answer must be a directly supervised assistant message")
    final_content = final.get("content")
    if not isinstance(final_content, str) or not any(x in final_content for x in ("CASE CLOSED", "CASE NOT CLOSED")):
        raise ValueError("final answer lacks explicit closure status")
    expected = str((row.get("labels") or {}).get("expected_closure") or "").split(":", 1)[0].strip().lower()
    if expected == "determined" and "CASE NOT CLOSED" in final_content:
        raise ValueError("final closure marker conflicts with determined reference")
    if expected == "undetermined" and "CASE NOT CLOSED" not in final_content:
        raise ValueError("final closure marker conflicts with undetermined reference")
    unread = set(EID_RE.findall(final_content)) - env.disclosed
    if unread:
        raise ValueError(f"final answer cites unread evidence {sorted(unread)}")
    return {"case_id": row["case_id"], "tool_calls": calls,
            "assistant_targets": sum(m["role"] == "assistant" and m["loss"] for m in messages),
            "fetched_items": len(env.disclosed) - sum(eid.startswith("E1.") for eid in env.by_id),
            "training_approved": bool(row.get("training_approved"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.jsonl:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            print(json.dumps(validate(row), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
