from __future__ import annotations

import json
import random

TOOLS = [{"type": "function", "function": {
    "name": "request_evidence",
    "description": "Return the exact text of previously unread evidence items by ID. Choose IDs from the case index; the result arrives as a tool message.",
    "parameters": {"type": "object", "properties": {
        "evidence_ids": {"type": "array", "minItems": 1, "maxItems": 12, "uniqueItems": True,
                         "items": {"type": "string", "pattern": "^E[1-9][0-9]*\\.[1-9][0-9]*$"}},
        "reason": {"type": "string", "minLength": 10,
                   "description": "A concise statement of what this fetch will help decide."}},
        "required": ["evidence_ids", "reason"], "additionalProperties": False}}}]

SYSTEM = ("You investigate a case file. Keep stable H1/H2 hypotheses, fetch evidence with "
          "request_evidence, cite evidence IDs, and finish with CASE CLOSED or CASE NOT CLOSED.")
KEY_IDS = ["E2.2", "E2.5", "E2.7"]


def make_case(n: int, closed: bool, source: str) -> tuple[dict, dict, dict]:
    cid = f"MOCK-{n:04d}"
    rec = [{"evidence_id": "E1.1", "neutral_title": "event summary", "kind": "observation", "source_id": "S1",
            "text": "A pump stopped during a routine run and the line tripped."},
           {"evidence_id": "E1.2", "neutral_title": "operator statement", "kind": "statement", "source_id": "S1",
            "text": "The operator saw a pressure drop before the trip."}]
    items = []
    for j in range(1, 13):
        eid = f"E2.{j}"
        if eid in KEY_IDS:
            title = f"sensor log {j} (decisive)"
            text = (f"Log {j}: the bearing temperature trace supports H1 (bearing seizure)."
                    if closed else f"Log {j}: the trace is inconclusive between H1 and H2.")
        else:
            title = f"background item {j}"
            text = f"Background {j}: maintenance paperwork, unrelated to the trip."
        items.append({"evidence_id": eid, "neutral_title": title, "kind": "record", "source_id": "S2", "text": text})
    brief = (f"# Case brief — {cid}\n\n## Task question\nBased only on the supplied materials, what caused "
             f"the pump trip? Cite the evidence_id for each key claim.\n\n## Initial context\nA synthetic "
             f"test case.\n\n## Case record E1 (2 items, full text)\n" +
             "".join(f"- {i['evidence_id']} | {i['neutral_title']} | {i['kind']} | source S1\n  {i['text']}\n" for i in rec) +
             "\n## Index of the remaining 12 evidence items (titles only)\n" +
             "".join(f"- {i['evidence_id']} | {i['neutral_title']} | record | source S2\n" for i in items) +
             "\nFetch exact text with request_evidence when needed. Finish with a direct answer that separates "
             "the supported mechanism, discounted alternatives, and unresolved details.")
    case = {"case_id": cid, "source": source, "user_message": brief, "tools": TOOLS,
            "evidence_items": rec + items, "package_hash": f"mock{n}"}
    teacher_final = (("CASE CLOSED. Supported mechanism: bearing seizure (H1) shown by E2.2, E2.5, E2.7. "
                      "Discounted alternatives: H2 electrical fault, no support. Unresolved details: none.")
                     if closed else
                     ("CASE NOT CLOSED - the logs E2.2, E2.5, E2.7 are inconclusive between H1 and H2. "
                      "Unresolved details: no teardown report exists in the file."))
    ref = {"case_id": cid, "source": source, "category": source, "closure": "closed" if closed else "not_closed",
           "closure_provenance": "synthetic", "teacher_marker": "closed" if closed else "not_closed",
           "teacher_final_answer": teacher_final, "key_evidence_ids": list(KEY_IDS),
           "key_fetchable_ids": list(KEY_IDS), "official_conclusion": "Bearing seizure." if closed else None,
           "alternatives_left_open": [] if closed else ["electrical fault (H2)"]}
    teacher = {"case_id": cid, "source": source, "tools": TOOLS, "messages": [
        {"role": "user", "content": brief, "loss": False},
        {"role": "assistant", "content": "## Current hypotheses\nH1 - bearing seizure.\nH2 - electrical fault.",
         "tool_calls": [{"id": "call_001", "type": "function", "function": {"name": "request_evidence",
                         "arguments": {"evidence_ids": list(KEY_IDS), "reason": "Which hypothesis do the decisive logs support?"}}}],
         "loss": True},
        {"role": "tool", "tool_call_id": "call_001", "name": "request_evidence",
         "content": json.dumps({"evidence_items": [i for i in items if i["evidence_id"] in KEY_IDS]},
                               ensure_ascii=False, separators=(",", ":")), "loss": False},
        {"role": "assistant", "content": teacher_final, "loss": True}]}
    return case, ref, teacher


def make_dataset(n_cases: int = 6) -> tuple[list, list, list]:
    cases, refs, teachers = [], [], []
    for n in range(1, n_cases + 1):
        closed = n % 2 == 1
        source = "host" if n % 3 else "ntsb"
        c, r, t = make_case(n, closed, source)
        cases.append(c)
        refs.append(r)
        teachers.append(t)
    return cases, refs, teachers


VARIANTS = ["careful", "careful", "careful", "bulk", "never_close", "premature", "flip", "nomarker",
            "badcall", "badsep"]


def call_text(prose: str, ids: list[str], reason: str, sep: str = "\n\n") -> str:
    return (prose + sep + "<tool_call>\n<function=request_evidence>\n<parameter=evidence_ids>\n" +
            json.dumps(ids) + "\n</parameter>\n<parameter=reason>\n" + reason +
            "\n</parameter>\n</function>\n</tool_call>")


def state_block(h1: str, h2: str, update: str, nxt: str | None) -> str:
    h1 = h1 if "->" in h1 else f"open -> {h1}"
    h2 = h2 if "->" in h2 else f"open -> {h2}"
    out = (f"## Current hypotheses\nH1 - bearing seizure.\nH2 - electrical fault.\n\n"
           f"## Evidence and hypothesis update\n{update}\n\n"
           f"## Hypothesis ledger\nH1 [{h1}] - see cited items.\nH2 [{h2}] - see cited items.")
    if nxt:
        out += f"\n\n## Next discriminating check\n{nxt}"
    return out


def scripted_reply(variant: str, turn: int, returned_texts: list[str], closed_truth_hint: bool | None,
                   rng: random.Random) -> str:
    supports = any("supports H1" in t for t in returned_texts)
    inconclusive = any("inconclusive" in t for t in returned_texts)
    header = f"Strategy: {variant}\n\n" if turn == 1 else ""
    if variant in ("careful", "never_close", "badsep", "nomarker"):
        if turn == 1:
            return call_text(header + state_block("open", "open", "Initial hypotheses from E1.1 and E1.2.",
                                                  "Which hypothesis do the decisive logs support?"),
                             list(KEY_IDS), "Which hypothesis do the decisive logs support?",
                             sep="\n" if variant == "badsep" else "\n\n")
        h1 = "favored" if supports else "open"
        if variant == "nomarker":
            return "Summary: the logs E2.2, E2.5 and E2.7 were read. The investigation stops here."
        if variant == "never_close" or not supports:
            return ("CASE NOT CLOSED - the decisive logs E2.2, E2.5, E2.7 point to H1 but a teardown is lacking. "
                    "Tentative: bearing seizure (H1). Unresolved details: no teardown report."
                    if supports else
                    "CASE NOT CLOSED - the logs E2.2, E2.5, E2.7 are inconclusive between H1 and H2. "
                    "Unresolved details: no teardown report exists in the file.")
        return ("CASE CLOSED. Supported mechanism: bearing seizure (H1) shown by E2.2, E2.5, E2.7. "
                f"Discounted alternatives: H2 electrical fault, not supported. Ledger H1 [{h1}].")
    if variant == "bulk":
        all_ids = [f"E2.{j}" for j in range(1, 13)]
        if turn == 1:
            return call_text(header + state_block("open", "open", "Initial hypotheses.", "Read everything."),
                             all_ids[:6], "Read every item to be thorough.")
        if turn == 2:
            return call_text(state_block("open", "open", "Read the first batch of items E2.1 to E2.6.", "Read the rest."),
                             all_ids[6:], "Read every remaining item to be thorough.")
        if supports:
            return "CASE CLOSED. Supported mechanism: bearing seizure (H1) shown by E2.2, E2.5, E2.7."
        return ("CASE NOT CLOSED - the logs E2.2, E2.5, E2.7 are inconclusive between H1 and H2. "
                "Unresolved details: no teardown report exists in the file.")
    if variant == "premature":
        return (header + "CASE CLOSED. Supported mechanism: electrical fault (H2), because the operator saw a "
                "pressure drop (E1.2).")
    if variant == "flip":
        if turn <= 3:
            h1, h2 = ("weakened -> favored", "favored -> weakened") if turn % 2 else ("favored -> weakened", "weakened -> favored")
            return call_text(header + state_block(h1, h2, "Re-reading the event summary changes the picture.",
                                                  "Re-check the event summary."),
                             ["E1.1"], "Re-check the event summary once more.")
        return rng.choice(["CASE CLOSED. Electrical fault (H2) per E1.1.",
                           "CASE NOT CLOSED - unclear; E1.1 only."])
    if variant == "badcall":
        if turn == 1:
            ids = [f"E2.{j}" for j in range(1, 13)] + ["E1.1"]
            return call_text(header + "Fetching many items.", ids, "Fetch a lot of items at once.")
        return "CASE NOT CLOSED - nothing was read; E1.1 only."
    raise ValueError(variant)
