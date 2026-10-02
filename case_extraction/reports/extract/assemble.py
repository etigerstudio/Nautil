from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import verify
from .adapters.base import RawCase, squash

KINDS = ("observation", "measurement", "statement", "test result", "examination conclusion")
NUMBER = re.compile(r"\d")
SAID = re.compile(r"\b(stated|reported|recalled|said|described|told|advised|claimed|complained)\b", re.I)
TESTED = re.compile(r"\b(test|assay|culture|analysis|sample|laborator|specimen|scan|imaging|radiograph)\w*\b", re.I)


def normalise_kind(kind: str, text: str) -> str:
    kind = squash(kind).lower()
    for allowed in KINDS:
        if kind == allowed or kind.startswith(allowed[:6]):
            return allowed
    if SAID.search(text):
        return "statement"
    if TESTED.search(text):
        return "test result"
    if NUMBER.search(text):
        return "measurement"
    return "observation"


def apply_review(items: list[dict[str, Any]], reference: dict[str, Any],
                 review: dict[str, Any]) -> dict[str, Any]:
    actions: dict[str, Any] = {"dropped_items": [], "dropped_rejections": [], "flagged_identifiers": [],
                               "determination_overridden": None, "missed_rejections": []}
    if not review.get("ok"):
        actions["review_failed"] = review.get("error")
        return actions
    leak_ids = set()
    for leak in review.get("leaks") or []:
        if not isinstance(leak, dict):
            continue
        if leak.get("kind") == "conclusion" and leak.get("evidence_id"):
            leak_ids.add(str(leak["evidence_id"]))
        elif leak.get("kind") == "identifier":
            actions["flagged_identifiers"].append(leak)
    if leak_ids:
        actions["dropped_items"] = sorted(leak_ids)
        items[:] = [i for i in items if i["evidence_id"] not in leak_ids]
    verdicts = {int(v["index"]): str(v.get("verdict", "")) for v in (review.get("rejected_explanations") or [])
                if isinstance(v, dict) and str(v.get("index", "")).lstrip("-").isdigit()}
    kept = []
    for index, item in enumerate(reference["rejected_explanations"]):
        verdict = verdicts.get(index, "unreviewed")
        item["review_verdict"] = verdict
        if verdict == "not_a_rejection":
            actions["dropped_rejections"].append(item["contradicted_mechanism"])
            continue
        kept.append(item)
    for number, item in enumerate(kept, 1):
        item["id"] = f"C{number}"
    reference["rejected_explanations"] = kept
    determination = review.get("determination") or {}
    if isinstance(determination, dict) and determination.get("agrees") is False:
        should = squash(str(determination.get("should_be", ""))).lower()
        if should in ("determined", "undetermined") and should != reference["determination"]:
            actions["determination_overridden"] = {"from": reference["determination"], "to": should,
                                                   "why": squash(str(determination.get("why", "")))}
            reference["determination"] = should
    actions["missed_rejections"] = [m for m in (review.get("missed_rejections") or []) if isinstance(m, dict)]
    return actions


def build(case_id: str, raw: RawCase, steps: dict[str, Any]) -> dict[str, Any]:
    items = steps["atomize"]["items"]
    reference = steps["reference"]
    actions = apply_review(items, reference, steps.get("review", {"ok": False, "error": "not run"}))
    id_map: dict[str, str] = {}
    groups: dict[str, int] = {}
    counters: dict[int, int] = {}
    for item in items:
        label = item["group_label"]
        if label not in groups:
            groups[label] = len(groups) + 1
        group = groups[label]
        counters[group] = counters.get(group, 0) + 1
        new_id = f"E{group}.{counters[group]}"
        id_map[item["evidence_id"]] = new_id
        item["evidence_id"] = new_id
    for entry in reference["rejected_explanations"]:
        entry["evidence_ids"] = [id_map.get(e, e) for e in entry["evidence_ids"] if e in id_map]

    package = {
        "case_id": case_id, "domain": raw.domain, "task_question": raw.task_question,
        "initial_context": steps["anonymize"]["initial_context"],
        "evidence_items": [{"evidence_id": i["evidence_id"], "neutral_title": i["neutral_title"],
                            "text": i["text"], "source_id": i["source_id"],
                            "kind": normalise_kind(i["kind"], i["text"])} for i in items],
        "version": "1.0.0-auto", "package_hash": "",
    }
    for item in package["evidence_items"]:
        item["text"] = verify.tidy(squash(verify.strip_control(item["text"])))
        item["neutral_title"] = verify.tidy(squash(verify.strip_control(item["neutral_title"])))
    package["initial_context"] = verify.tidy(squash(verify.strip_control(package["initial_context"])))
    identity = steps["anonymize"]["identity"]
    withheld = steps["anonymize"]["withheld_identifiers"]
    blob = verify.model_visible_text(package)
    ref = {
        "case_id": case_id,
        "source": {"dataset": raw.source_dataset, "native_id": raw.native_id, "url": raw.url,
                   "document": raw.document, "license": raw.license},
        "expected_closure": f"{reference['determination']}: {reference['closure_object']}",
        "official_conclusion_verbatim": reference["official_conclusion_verbatim"],
        "official_analysis_verbatim": steps["analysis_text"],
        "counterevidence": [{"id": c["id"], "contradicted_mechanism": c["contradicted_mechanism"],
                             "text": c["text"], "evidence_ids": c["evidence_ids"], "note": c["note"],
                             "confirmed_by_human": False, "confirmed_by_model_review": c.get("review_verdict") == "sound",
                             "review_verdict": c.get("review_verdict", "unreviewed"),
                             "quote_verbatim_in_analysis": c["quote_verbatim_in_analysis"]}
                            for c in reference["rejected_explanations"]],
        "alternatives_left_open": reference["alternatives_left_open"],
        "withheld_identifiers": withheld,
        "anonymization": [list(pair) for pair in steps["anonymize"]["replacements"]],
        "identity": identity,
        "withheld_sentences_in_factual_section": [
            {"locator": "factual block", "reason": squash(str(w.get("reason", ""))), "sentence": squash(str(w.get("sentence", "")))}
            for w in steps["atomize"]["withheld_sentences"]],
        "package_construction": {
            "pipeline": "Nautil case extraction v1 (model-extracted, model-reviewed, mechanically verified)",
            "blocks_used": sorted({i["group_label"] for i in items}),
            "blocks_by_role": steps.get("label", {}).get("roles", "source-declared"),
            "atom_count_by_source": {s: sum(1 for i in items if i["source_id"] == s)
                                     for s in sorted({i["source_id"] for i in items})},
            "coverage_by_block": steps["atomize"].get("coverage_by_block", {}),
            "review_actions": actions,
        },
        "stage2_confirmation": {
            "reviewer": "model review pass (no human confirmation)",
            "closure_confirmed": actions.get("determination_overridden") is None,
            "counterevidence_confirmed": f"{sum(c.get('review_verdict') == 'sound' for c in reference['rejected_explanations'])}"
                                         f" of {len(reference['rejected_explanations'])} judged sound by the review pass",
            "recognition_risk": "unknown until the recognition probe is run",
            "decision": "auto-include",
        },
    }
    checks = {
        "items": len(items),
        "quote_unverified": sum(not i["quote_verified"] for i in items),
        "low_overlap_items": sum(i["overlap"] < 0.6 for i in items),
        "conclusion_words_in_package": verify.conclusion_hits(blob),
        "withheld_still_present": verify.leakage(blob, withheld),
        "residual_withheld_tokens": verify.residual_withheld_tokens(blob, withheld) if raw.anonymize else [],
        "residual_identifiers": verify.residual_identifiers(blob) if raw.anonymize else {},
        "proper_nouns_remaining": steps["anonymize"].get("proper_nouns_remaining", [])[:25],
        "near_duplicates_dropped": len(steps["atomize"].get("near_duplicates_dropped", [])),
        "counterevidence": len(ref["counterevidence"]),
        "counterevidence_quote_unverified": sum(not c["quote_verbatim_in_analysis"] for c in ref["counterevidence"]),
        "counterevidence_without_evidence_ids": sum(not c["evidence_ids"] for c in ref["counterevidence"]),
        "mean_coverage": round(sum(steps["atomize"].get("coverage_by_block", {}).values()) /
                               max(1, len(steps["atomize"].get("coverage_by_block", {}))), 3),
    }
    return {"package": package, "reference": ref, "checks": checks, "review_actions": actions}


def provenance(case_id: str, raw: RawCase, steps: dict[str, Any], out: dict[str, Any]) -> str:
    checks = out["checks"]
    lines = [f"# {case_id} — automatic extraction", "",
             "## Source",
             f"- Dataset: {raw.source_dataset}; native id `{raw.native_id}`.",
             f"- Document: {raw.document}", f"- Licence: {raw.license}",
             f"- URL: {raw.url or '(none recorded)'}", "",
             "## How this package was made",
             "Produced by the Nautil case extraction pipeline (no human wrote or confirmed any part of it).",
             f"1. Blocks: {len(raw.blocks)} candidate blocks; roles "
             + ("assigned by the model" if raw.needs_block_labelling else "declared by the source layout") + ".",
             f"2. Atomisation: {checks['items']} evidence items; "
             f"{checks['quote_unverified']} without a verbatim quote found in their block; "
             f"mean word coverage of the factual blocks {checks['mean_coverage']}.",
             f"3. Sources: {len(steps['sources']['sources'])} origins after merging"
             + (f"; unmapped proposals: {steps['sources']['unmapped']}" if steps["sources"].get("unmapped") else "") + ".",
             f"4. Anonymisation: " + ("not applied (clinical case reports keep their wording; the identity is the paper)."
                                      if not raw.anonymize else
                                      f"{len(steps['anonymize']['replacements'])} strings replaced, "
                                      f"{steps['anonymize']['applied']} substitutions applied; "
                                      f"strings still present after the sweep: {checks['withheld_still_present'] or 'none'}."),
             f"5. Reference: determination `{out['reference']['expected_closure'].split(':')[0]}`; "
             f"{checks['counterevidence']} rejected explanations, "
             f"{checks['counterevidence_quote_unverified']} whose quote was not found verbatim in the analysis.",
             "6. Review pass: " + json.dumps(out["review_actions"], ensure_ascii=False)[:600],
             "", "## What a human still has to check",
             "- Every rejected explanation is model-asserted; `confirmed_by_human` is false throughout.",
             "- Coverage is measured by word overlap, not by meaning; a dropped fact can still hide behind a high number.",
             "- The recognition probe has not been run on this package yet."]
    if raw.extra.get("garbled"):
        lines.append(f"- Text-layer quality score of the source PDF cache: {raw.extra['garbled']} (0 clean, 1 unusable).")
    return "\n".join(lines) + "\n"
