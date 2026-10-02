#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from nautil_common import paths

ROOT = paths.DATA
RUN = paths.RUN
OUT = RUN / "results/sft_audit_repair_v2"
EID = re.compile(r"\bE[1-9][0-9]*\.[1-9][0-9]*\b")


def kind(issue: dict) -> str:
    text = " ".join(str(issue.get(key) or "") for key in ("finding", "suggested_fix")).lower()
    if any(word in text for word in ("before it was fetched", "before it was disclosed", "not yet returned",
                                     "unread", "premature", "forward", "before the tool")):
        return "temporal_evidence"
    if any(word in text for word in ("hypothesis", "ledger", "favored", "weakened", "state change")):
        return "hypothesis_update"
    if any(word in text for word in ("missing evidence", "not fetched", "fetch", "request")):
        return "evidence_selection"
    if issue.get("earliest_round", 1) > issue.get("total_rounds", 0) or "final" in str(issue.get("round")).lower():
        return "final_answer"
    if any(word in text for word in ("final answer", "conclusion", "closure")):
        return "final_answer"
    return "factual_grounding"


def timeline(trace: dict) -> dict[str, int]:
    disclosed = {}
    for eid in EID.findall(trace["messages"][0]["content"]):
        if eid.startswith("E1."):
            disclosed[eid] = 0
    for turn, pos in enumerate(range(1, len(trace["messages"])-1, 2), 1):
        result = json.loads(trace["messages"][pos+1]["content"])
        for item in result["evidence_items"]:
            disclosed.setdefault(item["evidence_id"], turn)
    return disclosed


def assess_before(record: dict) -> dict:
    original = json.loads((ROOT / record["original_trace_path"]).read_text())
    first_disclosed = timeline(original)
    rounds = record["rounds"]
    assessed = []
    for issue in record["issues"]:
        annotated = {**issue, "total_rounds": rounds}
        category = kind(annotated)
        at = issue["earliest_round"]
        ids = sorted(set(issue.get("evidence_ids") or []) | set(EID.findall(str(issue.get("finding") or ""))))
        statuses = {eid: first_disclosed.get(eid) for eid in ids}
        later = [eid for eid, fetched_after in statuses.items()
                 if fetched_after is not None and fetched_after >= at and at <= rounds]
        target = original["messages"][-1]["content"] if at > rounds else original["messages"][2*at-1]["content"]
        assessed.append({"id": issue["id"], "audit": issue["audit"],
                         "category_hint": category, "earliest_affected_round": at,
                         "evidence_first_disclosed_after_round": statuses,
                         "audit_cites_evidence_not_yet_visible_at_issue_round": later,
                         "original_target_sha256": __import__("hashlib").sha256(target.encode()).hexdigest(),
                         "assessment_limit": "Mechanical timing/type triage only; source support and audit validity require teacher review."})
    return {"case_id": record["case_id"], "rounds": rounds, "issues": assessed}


def assess_after(record: dict, original: dict, revised: dict, proposal: dict) -> dict:
    before = assess_before(record)
    resolutions = {item["id"]: item for item in proposal.get("issue_resolutions") or []}
    outcomes = []
    for issue in before["issues"]:
        iid = issue["id"]
        at = issue["earliest_affected_round"]
        index = len(original["messages"])-1 if at > record["rounds"] else 2*at-1
        original_text = original["messages"][index]["content"]
        revised_text = revised["messages"][index]["content"]
        resolution = resolutions.get(iid) or {}
        claimed = resolution.get("status")
        changed = original_text != revised_text
        review = "needs_blind_semantic_review"
        if claimed == "fixed" and not changed:
            review = "claimed_fixed_without_target_text_change_check_fetch_or_dependent_rounds"
        if claimed == "not_fixable":
            review = "explicitly_unfixable"
        outcomes.append({"id": iid, "category_hint": issue["category_hint"],
                         "earliest_affected_round": at,
                         "teacher_disposition": claimed,
                         "teacher_explanation": resolution.get("explanation"),
                         "target_text_changed": changed,
                         "local_review_signal": review})
    return {"case_id": record["case_id"], "issues": outcomes,
            "all_issues_accounted_for": len(outcomes) == len(record["issues"]),
            "semantic_validation_deferred_to_blind_score": True}


def main() -> None:
    records = [json.loads(line) for line in (OUT / "candidates.jsonl").read_text().splitlines() if line.strip()]
    output = [assess_before(record) for record in records]
    (OUT / "issue_assessment.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in output))
    summary = {"cases": len(output), "issues": sum(len(item["issues"]) for item in output),
               "category_hints": dict(Counter(issue["category_hint"] for item in output for issue in item["issues"])),
               "semantic_truth_not_inferred": True}
    (OUT / "issue_assessment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
