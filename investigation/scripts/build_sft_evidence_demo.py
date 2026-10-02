#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from nautil_common import paths

ROOT = paths.DATA
RUN = paths.RUN
PACKAGE_PATH = paths.CASES / "pilot/runtime/NTSB-10.json"
REFERENCE_PATH = paths.CASES / "pilot/review/NTSB-10/reference.json"
TOOLS_PATH = paths.CONFIGS / "sft_evidence_tool_v1.json"
OUT_JSONL = RUN / "docs/NTSB_NATIVE_TOOL_SAMPLE.jsonl"
OUT_MD = RUN / "docs/NTSB_NATIVE_TOOL_SAMPLE.md"
EID_RE = re.compile(r"\bE[1-9][0-9]*\.[1-9][0-9]*\b")
EXACT_EID_RE = re.compile(r"^E[1-9][0-9]*\.[1-9][0-9]*$")


def validate_evidence_args(args: object) -> None:
    if not isinstance(args, dict) or set(args) != {"evidence_ids", "reason"}:
        raise ValueError("request_evidence arguments require exactly evidence_ids and reason")
    ids, reason = args["evidence_ids"], args["reason"]
    if not isinstance(ids, list) or not 1 <= len(ids) <= 12 or any(
        not isinstance(eid, str) or not EXACT_EID_RE.fullmatch(eid) for eid in ids
    ) or len(set(ids)) != len(ids):
        raise ValueError("request_evidence evidence_ids violate the fixed tool contract")
    if not isinstance(reason, str) or len(reason) < 10:
        raise ValueError("request_evidence reason violates the fixed tool contract")


def compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def kind(item: dict) -> str:
    return str(item.get("kind") or str(item.get("locator", "")).rsplit(" | ", 1)[-1])


def render_items(items: list[dict]) -> str:
    return "\n".join(
        f'- {item["evidence_id"]} | {item["neutral_title"]} | '
        f'{item.get("kind") or item.get("locator") or ""} | source {item["source_id"]}\n  {item["text"]}'
        for item in items
    )


def opening_brief(package: dict) -> str:
    record = [x for x in package["evidence_items"] if x["evidence_id"].startswith("E1.")]
    assert record, "case has no opening record"
    indexed = [x for x in package["evidence_items"] if x not in record]
    index = "\n".join(
        f'- {x["evidence_id"]} | {x["neutral_title"]} | {kind(x)} | source {x["source_id"]}'
        for x in indexed
    )
    return (
        f'# Case brief — {package["case_id"]}\n\n'
        f'## Task question\n{package["task_question"]}\n\n'
        f'## Initial context\n{package["initial_context"]}\n\n'
        f'## Case record E1 ({len(record)} items, full text)\n{render_items(record)}\n\n'
        f'## Index of the remaining {len(indexed)} evidence items (titles only)\n{index}\n\n'
        'Fetch exact text with request_evidence when needed. Finish with a direct answer '
        'that separates the supported mechanism, discounted alternatives, and unresolved details.'
    )


class EvidenceEnvironment:
    def __init__(self, package: dict):
        self.by_id = {x["evidence_id"]: x for x in package["evidence_items"]}
        self.disclosed = {eid for eid in self.by_id if eid.startswith("E1.")}

    def fetch(self, ids: list[str]) -> str:
        unknown = [eid for eid in ids if eid not in self.by_id]
        if unknown:
            raise ValueError(f"unknown evidence: {unknown}")
        repeated = [eid for eid in ids if eid in self.disclosed]
        new = [eid for eid in ids if eid not in self.disclosed]
        self.disclosed.update(new)
        result = {"evidence_items": [self.by_id[eid] for eid in new]}
        if repeated:
            result["already_delivered"] = repeated
        return compact(result)


FETCHES = [
    {
        "evidence_ids": ["E3.2", "E3.3", "E3.4", "E4.2", "E4.3", "E4.4", "E5.4", "E5.5"],
        "reason": "Check what the pilot knew about tank switching and what the purchase and consumption records actually establish.",
    },
    {
        "evidence_ids": ["E6.9", "E6.10", "E6.11"],
        "reason": "The recovered tank fuel, selector position and leakage pattern distinguish selected-tank starvation from total exhaustion.",
    },
    {
        "evidence_ids": ["E6.4", "E7.1", "E7.2", "E7.3", "E7.4", "E7.5"],
        "reason": "Check whether the engine and fuel-system examinations reveal a competing mechanical cause.",
    },
]

FINAL_ANSWER = """## Determination
CASE CLOSED — The evidence supports fuel starvation: the selector was found on the empty right tank and the fuel lines were dry, while about 18 gallons remained in the left tank [E6.10, E6.11]. The right bladder was intact and the right wing showed no fuel-leak staining, making postimpact drainage a weaker explanation for the empty right tank [E6.9]. The pilot reported rough running, temporary recovery and a final loss of power; those symptoms establish the sequence, not the mechanism on their own [E1.3, E1.6, E1.9].

## Discounted alternatives
Total fuel exhaustion does not fit the roughly 18 gallons recovered from the left tank [E6.10]. Examinations found compression and internal continuity, working magnetos, an unobstructed carburetor screen and float, and a pump that produced suction and expulsion. They did not reveal a preimpact component fault that explains the loss of power [E7.1, E7.2, E7.3, E7.4, E7.5].

## Not determined
The exact fuel quantity at departure is not established. A photograph of a 38.2-gallon receipt names a different airplane, and the actual usable capacity of this modified airplane could not be established [E4.4, E5.5]. The pilot had asked whether the tanks needed switching and said he would switch on the next flight, but these messages do not prove what he did in flight or why fuel was not obtained from the left tank [E3.2, E3.3, E3.4]."""


def build(package: dict, reference: dict, tools: list[dict]) -> dict:
    env = EvidenceEnvironment(package)
    messages: list[dict] = [{"role": "user", "content": opening_brief(package), "loss": False}]
    for n, args in enumerate(FETCHES, 1):
        call_id = f"call_{n:03d}"
        messages.append({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": "request_evidence", "arguments": compact(args)}}],
            "loss": False,
        })
        messages.append({"role": "tool", "tool_call_id": call_id, "name": "request_evidence",
                         "content": env.fetch(args["evidence_ids"]), "loss": False})
    messages.append({"role": "assistant", "content": FINAL_ANSWER, "loss": True})
    return {
        "sample_id": "ntsb-NTSB-10-one-fetch-tool-v1",
        "source": "ntsb_pilot",
        "case_id": package["case_id"],
        "package_hash": package["package_hash"],
        "mode": "evidence_requested",
        "origin": "human_authored_from_official_report",
        "review_status": "pilot_eval_holdout",
        "training_approved": False,
        "tools": tools,
        "messages": messages,
        "labels": {"expected_closure": reference["expected_closure"]},
        "provenance": {
            "package": str(PACKAGE_PATH.relative_to(ROOT)),
            "reference": str(REFERENCE_PATH.relative_to(ROOT)),
            "tool_contract": str(TOOLS_PATH.relative_to(ROOT)),
            "fetch_actions_are_hindsight": True,
            "model_visible_fields": ["tools", "messages"],
        },
        "split": "prototype_only",
    }


def validate(row: dict, package: dict, reference: dict, tools: list[dict]) -> list[str]:
    problems: list[str] = []
    if row.get("package_hash") != package["package_hash"]:
        problems.append("package hash changed")
    if row.get("tools") != tools or len(tools) != 1 or tools[0]["function"]["name"] != "request_evidence":
        problems.append("tool contract differs from the single fetch tool")
        return problems
    if row.get("labels", {}).get("expected_closure") != reference["expected_closure"]:
        problems.append("reference label changed")
    schema = tools[0]["function"]["parameters"]
    if schema.get("required") != ["evidence_ids", "reason"]:
        problems.append("request_evidence schema differs from the fixed contract")
    messages = row.get("messages") or []
    if not messages or messages[0] != {"role": "user", "content": opening_brief(package), "loss": False}:
        problems.append("opening brief differs from renderer")
        return problems
    if len(messages) < 4 or (len(messages) - 2) % 2:
        problems.append("messages do not alternate fetch call and tool reply before the final answer")
        return problems
    env = EvidenceEnvironment(package)
    seen_calls: set[str] = set()
    for pos in range(1, len(messages) - 1, 2):
        try:
            assistant, tool_message = messages[pos:pos + 2]
            if assistant.get("role") != "assistant" or assistant.get("content") not in (None, ""):
                raise ValueError("fetch turn must be a native assistant tool call")
            if assistant.get("loss") is not False:
                raise ValueError("report-derived fetch selection must be loss-masked")
            calls = assistant.get("tool_calls") or []
            if len(calls) != 1 or calls[0].get("type") != "function":
                raise ValueError("assistant fetch turn must contain exactly one native function call")
            call = calls[0]
            if not call.get("id") or call["id"] in seen_calls:
                raise ValueError("missing or duplicate tool call ID")
            seen_calls.add(call["id"])
            if call["function"]["name"] != "request_evidence":
                raise ValueError("unknown tool")
            args = json.loads(call["function"]["arguments"])
            validate_evidence_args(args)
            if set(EID_RE.findall(args["reason"])) - env.disclosed:
                raise ValueError("fetch reason cites unread evidence")
            expected = {"role": "tool", "tool_call_id": call["id"], "name": "request_evidence",
                        "content": env.fetch(args["evidence_ids"]), "loss": False}
            if tool_message != expected:
                raise ValueError("tool result differs from replayed package data")
        except Exception as exc:
            problems.append(f"message {pos}: {exc}")
            break
    final = messages[-1]
    if final.get("role") != "assistant" or final.get("loss") is not True or final.get("tool_calls"):
        problems.append("the final assistant answer must be a direct trainable response")
    if final.get("content") != FINAL_ANSWER:
        problems.append("final answer differs from reviewed text")
    missing = set(EID_RE.findall(str(final.get("content") or ""))) - env.disclosed
    if missing:
        problems.append(f"final answer cites undisclosed evidence: {sorted(missing)}")
    visible = compact({"tools": row.get("tools"), "messages": messages})
    for value in reference.get("withheld_identifiers") or []:
        if value and len(str(value)) >= 4 and str(value).casefold() in visible.casefold():
            problems.append(f"withheld identifier leaked into model-visible fields: {value!r}")
    return problems


def markdown(row: dict) -> str:
    lines = [
        "# NTSB-10: a complete SFT sample with the single evidence tool",
        "",
        "This sample is arranged after the fact from the official report; tool returns come from the frozen evidence package. The three fetches are chosen after the fact",
        " and set to `loss=false`; the final analysis is `loss=true`. It only checks the protocol and answer format",
        " and is not a teacher trajectory for fetch strategy. `labels` and `provenance` are not part of the model input.",
        "",
        f"Case: `{row['case_id']}`; package hash: `{row['package_hash']}`;",
        f" {len(row['messages'])} messages, {sum(m['role'] == 'tool' for m in row['messages'])} fetches.",
        "",
        "## Full messages",
        "",
    ]
    for n, message in enumerate(row["messages"]):
        lines += [f"### {n}. {message['role']} | loss={str(message['loss']).lower()}", "",
                  "```json", json.dumps(message, ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="replay existing JSONL without writing")
    args = parser.parse_args()
    package = json.loads(PACKAGE_PATH.read_text())
    reference = json.loads(REFERENCE_PATH.read_text())
    tools = json.loads(TOOLS_PATH.read_text())
    if args.check:
        lines = OUT_JSONL.read_text().splitlines()
        if len(lines) != 1:
            raise SystemExit(f"expected one JSONL record, got {len(lines)}")
        row = json.loads(lines[0])
    else:
        row = build(package, reference, tools)
    problems = validate(row, package, reference, tools)
    if problems:
        raise SystemExit("\n".join(problems))
    if not args.check:
        OUT_JSONL.write_text(json.dumps(row, ensure_ascii=False) + "\n")
        OUT_MD.write_text(markdown(row))
    print(f"OK {row['case_id']}: {len(row['messages'])} messages; "
          f"{sum(m['role'] == 'tool' for m in row['messages'])} native fetches; "
          f"{sum(m.get('loss') is True for m in row['messages'])} loss-bearing turn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
