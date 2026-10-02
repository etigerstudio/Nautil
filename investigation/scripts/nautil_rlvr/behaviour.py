from __future__ import annotations

import re

from .common import EVIDENCE_ID, initial_record_ids
from .reward import sections, valid_fetch, invalid_fetch

HYP_LINE = re.compile(r"^H(\d+)\s+[—-]", re.M)
LEDGER = re.compile(r"H(\d+)\s*\[([^\]→>]+?)\s*(?:→|->)\s*([^\]]+?)\]")
OUT_STATES = ("weakened", "ruled out", "excluded", "eliminated", "rejected", "discounted")


def stats(result: dict, case: dict, ref: dict | None, turn_rl: list[dict] | None = None) -> dict:
    turns = result.get("turns") or []
    seen_h: set[str] = set()
    new_after_first = 0
    per_turn_counts = []
    leading_seq = []
    final_states: dict[str, str] = {}
    for i, t in enumerate(turns):
        text = t.get("assistant") or ""
        cur = sections(text).get("Current hypotheses", "")
        hs = set(HYP_LINE.findall(cur))
        if hs:
            per_turn_counts.append(len(hs))
            if i > 0:
                new_after_first += len(hs - seen_h)
            seen_h |= hs
        ledger = LEDGER.findall(text)
        for h, _before, after in ledger:
            final_states[h] = after.strip().lower()
        fav = sorted(h for h, _b, a in ledger if a.strip().lower().startswith("favo"))
        if ledger:
            leading_seq.append(",".join(fav) or "-")
    switches = sum(1 for a, b in zip(leading_seq, leading_seq[1:]) if a != b and a != "-" and b != "-")
    fetch_sizes = [len(t["call"]["evidence_ids"]) for t in turns if t.get("call") is not None]
    valid = [t for t in turns if valid_fetch(t)]
    seen = set(initial_record_ids(case["user_message"]))
    rereq = 0
    for t in valid:
        ids = t["call"]["evidence_ids"]
        rereq += sum(i in seen for i in ids)
        seen.update(ids)
    fetch_turns = sum(t.get("call") is not None or bool(t.get("call_errors")) for t in turns)
    gen_per_turn = [len(x.get("gen_ids") or []) for x in (turn_rl or []) if x.get("gen_ids")]
    final = result.get("final_answer")
    cited = set(EVIDENCE_ID.findall(final or ""))
    key = set((ref or {}).get("key_fetchable_ids") or [])
    fetched = set(result.get("unique_fetched_evidence_ids") or [])
    return {"hypothesis_count": len(seen_h),
            "hypothesis_count_max_per_turn": max(per_turn_counts) if per_turn_counts else 0,
            "new_hypotheses_after_turn1": new_after_first,
            "hypotheses_ruled_out": sum(1 for s in final_states.values() if s.startswith(OUT_STATES)),
            "leading_switches": switches,
            "fetches": len(fetch_sizes), "ids_per_fetch": (sum(fetch_sizes) / len(fetch_sizes)) if fetch_sizes else None,
            "rerequested_items": rereq,
            "invalid_fetch_turns": sum(invalid_fetch(t) for t in turns),
            "invalid_format_rate": (sum(invalid_fetch(t) for t in turns) / fetch_turns) if fetch_turns else 0.0,
            "key_recall": (len(fetched & key) / len(key)) if key else None,
            "key_precision": (len(fetched & key) / len(fetched)) if fetched and key else None,
            "tokens_per_turn": (sum(gen_per_turn) / len(gen_per_turn)) if gen_per_turn else None,
            "final_answer_chars": len(final) if final else None,
            "final_citations": len(cited),
            "citations_valid": result.get("final_citations_all_disclosed")}
