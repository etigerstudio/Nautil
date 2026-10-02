#!/usr/bin/env python3
import json
import audit_sft_pilot14_v1 as audit

audit.SYSTEM = """You are an independent auditor of one review-only, single-investigator SFT trajectory. The trusted report is reviewer-only and is not part of the trainee's visible context. Judge the model-visible multi-round investigation and the trusted report. The user's acceptance rule is: a case is usable if it has NO MAJOR logical error and NO MAJOR structural error; minor imperfections are acceptable. Evidence citations are a structural requirement: every material observation used to update a hypothesis and every decisive final claim must have specific already-disclosed E IDs, and those items must actually support the claim. A cited ID that is merely topical but does not support the claim does not fix the error. Hard structural failures include missing causal conclusion, undeclared new hypotheses, forward citations, malformed tool calls or fabricated tool results. Hard logical failures include a decisive conclusion not supported by disclosed evidence, material contradiction with source facts, a different incident, or hiding unresolved cause as certain. A provisional favored/weakened state does NOT require proof beyond doubt; do not punish reasonable uncertainty or minor phrasing. Treat the question's causal answer separately from an agency's administrative case or phase status. If CASE NOT CLOSED is used to mean only that the causal question remains unresolved, assess the actual explanation and flag ambiguity only to the extent it changes meaning. Score as an integer from -5 to +5 for quality ranking: +5 excellent, +4 strong, +3 good, +2 usable but weaker, +1 marginal without a hard error, 0 uncertain, negative means a serious error. Score is not itself the gate. Recommend keep if no major logical or structural error, revise for a fixable hard error or unresolved audit concern, skip when the evidence/source cannot support a usable investigation without rebuilding. Do not approve training. Output exactly one JSON object with keys case_id, score, action, hard_failure, hard_failure_reason, checks, strengths, issues, verdict. checks must cover source_alignment, evidence_grounding, active_fetch, hypothesis_trajectory, honest_closure, each with pass/concern/fail and a short reason. issues is an array of {severity, round, finding, evidence_ids, suggested_fix}, with severity minor/major/critical. Give concrete evidence IDs or exact short text for material criticisms. Be concise; output JSON only."""

_base_editor = audit.editor


def editor_with_warnings(case_id: str, meta: dict) -> dict:
    prompt = _base_editor(case_id, meta)
    trace = json.loads((audit.TRACE_DIR / f"{case_id}{audit.TRACE_SUFFIX}").read_text())
    warnings = (trace.get("provenance") or {}).get("ledger_grounding_warnings") or []
    if warnings:
        prompt["teacher_grounding_warnings_reviewer_only"] = warnings
    return prompt


audit.editor = editor_with_warnings

if __name__ == "__main__":
    audit.main()
